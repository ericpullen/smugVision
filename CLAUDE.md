# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

smugVision reads a SmugMug album, generates a caption and keyword tags for each image with a
**local Ollama vision model**, and PATCHes the result back to SmugMug. Location (GPS → place name)
and face recognition (named reference faces) are folded into the model prompt as context, so the
generated captions contain real people's names and place names.

Two front ends sit on the same core: a CLI (`smugvision`) and a Flask web UI (`smugvision-web`)
that previews changes before committing them.

## Commands

```bash
pip install -e ".[dev]"          # dev install
pip install -e ".[insightface]"  # optional, experimental face backend (dlib is the default)
ollama list                      # any vision model here works; set it as vision.model
                                 # the shipped default is defaults.py -> vision.model

smugvision-config                # interactive setup → ~/.smugvision/config.yaml
smugvision-get-tokens            # OAuth 1.0a dance → user_token / user_secret
smugvision-optimize-faces        # downscale reference faces (big speedup on encoding)

smugvision --gallery <album_key> --dry-run    # safest way to exercise the whole pipeline
smugvision --url "https://site.smugmug.com/.../n-XXXXX/album-name" --verbose
smugvision --gallery <key> --force-reprocess  # ignore the marker tag
smugvision-web --port 5050                    # web UI (127.0.0.1 only by default)

black .                          # line-length 100 (configured in pyproject.toml)
flake8 smugvision
mypy smugvision
```

### Testing reality check

`pytest` is configured (`testpaths = ["tests"]`, `--cov=smugvision`) but **there are no pytest tests
yet** — nothing in `tests/` defines a `test_*` function. `tests/*.py` and root `test_config.py` are
hand-run integration scripts that hit the live SmugMug API and a live Ollama server:

```bash
python tests/test_smugmug.py --gallery <key>      # API client / auth
python tests/test_vision.py path/to/image.jpg     # Ollama round trip
python tests/debug_face_recognition.py img.jpg    # face detection tuning
python tests/test_processor.py <key> --dry-run    # full pipeline
python test_config.py --non-interactive           # config loading
```

Running bare `pytest` collects nothing and reports 0% coverage — that is expected, not a broken
install. New real tests go in `tests/` as `test_*.py` with `test_*` functions; mock `SmugMugClient`
and `VisionModel` (both are injectable — see below) rather than hitting the network.

## Architecture

`ImageProcessor` (`processing/processor.py`) is the single orchestrator. Everything else is a
collaborator it constructs from config, and **every collaborator is constructor-injectable**
(`smugmug_client`, `vision_model`, `cache_manager`, `face_recognizer`) — that is the seam for tests
and for the web UI.

Per-image pipeline in `process_image()`:

1. **Marker-tag skip** — images already carrying `processing.marker_tag` (default `smugvision`) are
   skipped unless `force_reprocess`. This is the idempotency mechanism; the marker is appended to
   the keyword list by `MetadataFormatter.format_tags()`.
2. **Download to cache** — `CacheManager` lays images out as `<cache_dir>/<album_name>/<filename>`.
   `SmugMugClient.download_image(skip_if_exists=True)` returns `None` when the file is already
   cached, so the processor reconstructs the path itself — don't "fix" that None into an error.
3. **GPS: SmugMug API first, EXIF second.** SmugMug strips GPS from downloaded image bytes but
   exposes it on the API object, so `image.has_gps` wins and `extract_exif_location()` is only the
   fallback. Any change here silently breaks location on most photos.
4. **Location resolution** — `resolve_location_with_custom()` checks `~/.smugvision/locations.yaml`
   (user-defined places with radius + aliases, `utils/locations.py`) before reverse geocoding via
   Nominatim + an Overpass POI lookup. Custom aliases also become tags when
   `location.use_aliases_as_tags` is on.
5. **Face recognition** — `FaceRecognizer.get_person_names()` returns names above
   `min_confidence`. It returns the reference-folder names **verbatim**, underscores intact
   (`John_Doe`). Conversion to "John Doe" happens downstream, for display/tags/captions.
   `~/.smugvision/relationships.yaml` is matched on the *underscore* form, so raw names must
   survive into the vision layer or relationships silently match nothing.
6. **Prompt building + generation — ONE structured call.**
   `LlamaVisionModel.generate_metadata(image_path, caption_instruction, tags_instruction, *,
   temperature, max_tokens, location_context, person_names, total_faces, album_name)` builds one
   prompt from both instructions plus all context, encodes the image **once**, and makes **one**
   `ollama.Client.chat()` request constrained by a JSON schema, returning a `MetadataResult`.
   Context injection (album, location, people, and `relationships.yaml` via
   `_enhance_prompt_with_context()`) lives in the vision layer. `generate_caption()` and
   `generate_tags()` still exist with unchanged signatures as thin wrappers over
   `generate_metadata` — passing `""` for the other instruction skips that half of the reply.
   Escape hatches: `vision.single_call: false` restores the two-request path and
   `vision.structured_output: false` restores free-text output with heuristic parsing
   (`_parse_tags`, `_strip_thinking_tags`). Both legacy paths are live, not stubs.
7. **Formatting** → `MetadataFormatter` merges AI output with existing caption/keywords
   (`preserve_existing` joins old and new with ` | `), dedupes case-insensitively, appends marker.
8. **Write or dry-run**, then an explicit `gc.collect()` — image + dlib buffers are large and this
   is deliberate.

### Layers

- `smugmug/` — OAuth 1.0a client (`requests-oauthlib`), node-tree walking to turn a URL's
  `n-XXXXX` node ID + slug into an album key (`resolve_album_key`), `AlbumImage`/`Album` models.
- `vision/` — `VisionModel` ABC + `VisionModelFactory`. All Ollama vision models share one
  implementation, `LlamaVisionModel` (a misnomer — it is the generic Ollama adapter).
  **There is no allow-list**: `VisionModelFactory.create()` maps any model name to
  `LlamaVisionModel`, so using a new Ollama model needs **no code change**, just
  `vision.model` in config. `register_model(name, cls)` remains as an override hook for a
  genuinely different implementation and takes precedence over the default.
  `vision.validate_model` (default true) warns — never fails — when the model is missing from
  Ollama's tag list. `LlamaVisionModel` owns a real `ollama.Client(host=endpoint,
  timeout=timeout)`, so `vision.endpoint` and `vision.timeout` are actually honoured, and it
  threads `keep_alive` and `think` into every chat call. `_encode_image()` downscales the long
  edge to `vision.max_image_dimension` (default 1568; 0/None disables, never upscales), applies
  EXIF orientation, and re-encodes at `vision.jpeg_quality`.
  `_strip_thinking_tags()` still removes `<think>…</think>` for reasoning models on the free-text
  path — but the real lever is `vision.think: false`, which is the default precisely because a
  reasoning model can consume the whole `max_tokens` budget before emitting content. Remember
  `max_tokens` is now one budget for caption **and** tags together.
- `face/` — reference faces live as `reference_faces/<Person_Name>/*.jpg`. `FaceRecognizer` is a
  backend-agnostic coordinator (reference map, cache, name/confidence filtering); detection and
  embedding live in `face/backends/` (`FaceBackend` ABC, `DlibFaceBackend`,
  `InsightFaceBackend`, `create_backend`/`register_backend`). `face_recognition.backend` selects
  one; **dlib is the default** and InsightFace is an optional extra
  (`pip install -e ".[insightface]"`) that falls back to dlib with a logged error when its
  dependencies are missing. `FaceBackend.score()` returns a normalized confidence (>= 0.0 iff
  within that backend's threshold, 1.0 at a perfect match), so `min_confidence` means the same
  thing on every backend and the matching loop never sees a raw distance or similarity.
  Encodings are pickled to `~/.smugvision/cache/face_encodings` keyed by a path+size+mtime
  fingerprint; the filename carries the backend slug and the manifest records
  backend/embedding_model/dim/metric/normalized, so a cache is rejected wholesale if any of them
  disagree — the two backends can never compare 128-d dlib vectors against 512-d ArcFace ones.
  Bump `CACHE_VERSION` (now 2) whenever the encoding format changes. `face_recognition` is an
  optional import guarded by `FACE_RECOGNITION_AVAILABLE` — note it raises `SystemExit`, not
  `ImportError`, when its models are missing, which is why the guard catches both — and a missing
  reference dir degrades to "face recognition disabled" rather than failing.
- `utils/exif.py` is the live geocoding path (reads `~/.smugvision/geocoding_config.yaml`).
  `reverse_geocode()` is a caching wrapper over `_reverse_geocode_uncached()`: results are
  memoized for the life of the process against coordinates rounded to
  `GEOCODE_CACHE_PRECISION` (4 places, ~11m), capped at `GEOCODE_CACHE_MAX_ENTRIES` with
  FIFO eviction, lock-guarded. **Failures are cached too** — deliberately, so an
  unresolvable coordinate does not re-time-out once per photo; `clear_geocode_cache()`
  retries and `geocode_cache_info()` reports hits/misses. Anything that needs a live lookup
  must call `_reverse_geocode_uncached()` explicitly. There is **no** rate limiting despite
  what Nominatim's usage policy asks; the cache is what keeps request volume down.
  `utils/exif_optimized.py` is a faster alternative that **nothing imports** — dead code,
  not a dependency, and its separate `_geocode_cache` is now redundant too.
- `web/` — Flask app factory + two blueprints (`pages_bp`, `api_bp` at `/api`). `PreviewService`
  wraps the *same* `ImageProcessor` with `dry_run=True` and streams SSE progress; results are held
  in an in-memory job dict (max 5 jobs) and written only when `POST /api/commit` runs. Keep new
  processing logic in `processing/`, not in the service — the point of that design is CLI/UI parity.
- `config/` — `ConfigManager.load()` deep-merges the YAML over `config/defaults.py`, so adding a
  key to `DEFAULT_CONFIG` reaches existing users' configs automatically. Access is dot-notation
  (`config.get("vision.model")`). Search order: `~/.smugvision/config.yaml`, then `./config.yaml`.
  Only the four `smugmug.*` credentials are required. Behaviour keys worth knowing:
  `vision.think`, `vision.keep_alive`, `vision.single_call`, `vision.structured_output`,
  `vision.max_image_dimension`, `vision.jpeg_quality`, `vision.validate_model`,
  `face_recognition.backend` and the `face_recognition.insightface.*` sub-block.
  **A key in `DEFAULT_CONFIG` only takes effect if a call site forwards it.** `ImageProcessor`
  is the forwarding point for `vision.*` (into `VisionModelFactory.create`, which relays
  `**kwargs` to the constructor) and for `face_recognition.*` (into `FaceRecognizer`). Check
  there first when a new setting appears to do nothing.

### State lives outside the repo

`~/.smugvision/` holds `config.yaml`, `locations.yaml`, `relationships.yaml`,
`geocoding_config.yaml`, `reference_faces/`, `cache/`, and `smugvision.log`. Nothing under the repo
is authoritative at runtime; `config.yaml.example` and `locations.yaml.example` are templates only.

### The write path

`SmugMugClient.update_image_metadata(image_key, caption=None, keywords=None, title=None)` wants a
caption **string** and a keywords **list**; it builds the SmugMug-cased PATCH body itself and joins
the keywords into the comma-separated string the API expects. `None` leaves a field untouched.

Both callers now use that keyword form — `ImageProcessor.process_image()` and
`PreviewService.commit_changes()`. (`ImageProcessor` previously passed
`MetadataFormatter.create_update_payload()`'s dict as the second *positional* argument, so the
whole payload landed in `caption`; that is fixed, and the payload builder — which returned
SmugMug-cased `{'Caption': str, 'Keywords': list}` that never matched the parameter names — has
been deleted rather than left as a trap.) Do not "simplify" either call site to
`update_image_metadata(image_key, some_dict)` or to `**some_dict`.

### Config keys that are still not wired

Verify before documenting or "fixing" — these were true at the time of writing:

- `face_recognition.tolerance` / `.model` / `.detection_scale` — `FaceRecognizer` accepts all
  three, but `ImageProcessor` does not pass them, so the constructor defaults (0.6 / `cnn` /
  0.5) win. A `face_recognition.dlib:` sub-block *does* reach the backend, because the
  processor reads `face_recognition.<backend>` into `backend_options`.
- `processing.generate_captions` / `.generate_tags` — read by nothing; both are always
  produced. Honouring them is now cheap: `generate_metadata` treats an empty
  `caption_instruction` or `tags_instruction` as "skip that half".
- `tests/test_vision.py` hardcodes a model name rather than reading `vision.model`, so the
  repo's main hand-run smoke test fails with a model-not-found 404 on a machine that does not
  have that exact model. That failure is not evidence of a broken vision layer.

## Conventions

Google-style docstrings with Args/Returns/Raises on all public methods, type hints throughout,
100-char lines, `logger = logging.getLogger(__name__)` per module. Optional heavy dependencies
(`face_recognition`, `insightface`/`onnxruntime`, `geopy`, `exifread`, `pillow_heif`, `httpx`) are
imported in try/except behind an `X_AVAILABLE` flag and degrade gracefully — preserve that.
HEIC/HEIF support comes from `pillow_heif.register_heif_opener()` in `vision/llama.py`.
Conventional Commits for commit messages (`feat:`, `fix:`, `docs:`).

`DESIGN.md` is the long-form architecture and roadmap doc; the topic-specific `README_*.md` files
cover config, processor, face recognition, and SmugMug testing in more depth.
