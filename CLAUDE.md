# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

smugVision reads a SmugMug album, generates a caption and keyword tags for each image with a
**local Ollama vision model**, and PATCHes the result back to SmugMug. Location (GPS → place name)
and face recognition (named reference faces) are folded into the model prompt as context, so the
generated captions contain real people's names and place names.

Two front ends sit on the same core: a CLI (`smugvision`) and a Flask web UI (`smugvision-web`)
that previews changes before committing them.

Anything the model cannot see for itself is asserted by the user rather than guessed: notes,
location overrides, who is in a photo, and which pets are in it. All of that lives in
`~/.smugvision/hints.yaml` and `pets.yaml`, is injected into the prompt as ground truth, and
outranks the model's own reading of the image.

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
smugvision --gallery <key> --no-preserve-existing   # replace caption/keywords, do not merge
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
   the keyword list by `MetadataFormatter.format_tags()`. The test lives in
   `AlbumImage.has_marker_tag()` → `smugmug.models.keywords_contain()`, which is **separator
   aware**: SmugMug returns the whole keyword list as one semicolon-joined blob, so a plain
   `"smugvision" in image.keywords` matches nothing on every image already processed. One rule,
   one place — `ImageProcessor.needs_processing()` exposes it so a caller can filter an album
   *before* processing instead of processing everything and discarding the skips.
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
   A **people override** (`HintManager.resolve_people()`) replaces the recognised list outright
   when the user has said who is in the photo — applied here, not as a note, because the
   recognised names also feed the keywords, the relationships lookup and `result.detected_faces`,
   none of which a note can reach. It applies even with face recognition disabled entirely.
5b. **Pets** — `HintManager.resolve_pets()` names animals, which have no reference faces.
   `ImageProcessor._resolve_pets()` turns them into the description sentences from
   `~/.smugvision/pets.yaml` (`processing/pets.py`) and appends them to the hint text; the names
   go to `format_tags(pet_names=...)`. A pet deliberately never joins `person_names`: it must not
   be counted as a detected face or reach the "N people visible" arithmetic in the prompt. A pet
   ticked but since deleted from pets.yaml contributes nothing rather than a bare name.
6. **Prompt building + generation — ONE structured call.**
   `LlamaVisionModel.generate_metadata(image_path, caption_instruction, tags_instruction, *,
   temperature, max_tokens, location_context, person_names, total_faces, album_name, hints,
   title_instruction)` builds one
   prompt from both instructions plus all context, encodes the image **once**, and makes **one**
   `ollama.Client.chat()` request constrained by a JSON schema, returning a `MetadataResult`.
   Context injection (album, location, people, and `relationships.yaml` via
   `_enhance_prompt_with_context()`) lives in the vision layer. `generate_caption()` and
   `generate_tags()` still exist with unchanged signatures as thin wrappers over
   `generate_metadata` — passing `""` for the other instruction skips that half of the reply.
   Escape hatches: `vision.single_call: false` restores the two-request path and
   `vision.structured_output: false` restores free-text output with heuristic parsing
   (`_parse_tags`, `_strip_thinking_tags`). Both legacy paths are live, not stubs.
   `hints` is the resolved user-asserted text (global + album + image, plus any pet sentences) and
   is injected **last** so nothing later dilutes it. `title_instruction` is passed only when
   `processing.generate_titles` is on; the title is offered in the JSON schema but deliberately
   NOT required, so a model that ignores it costs nothing.
7. **Formatting** → `MetadataFormatter` merges AI output with existing caption/keywords
   (`preserve_existing` joins old and new with ` | `), dedupes case-insensitively, appends marker.
   `format_caption()` only appends its "Featuring X at Y" suffix when the caption does not already
   name those people/places. `format_tags()` takes `person_names`, `pet_names` and `location_tags`
   as separate arguments — keep them separate, that is what stops a pet being treated as a face.
8. **Write or dry-run**, then an explicit `gc.collect()` — image + dlib buffers are large and this
   is deliberate.

### Layers

- `smugmug/` — OAuth 1.0a client (`requests-oauthlib`), node-tree walking to turn a URL's
  `n-XXXXX` node ID + slug into an album key (`resolve_album_key`), `AlbumImage`/`Album` models.
  `models.split_keywords()` / `keywords_contain()` are the single source of truth for reading
  SmugMug's semicolon-joined keyword blob; `AlbumImage.has_marker_tag()` and
  `ImageProcessor._split_keywords()` both delegate to them, so the marker rule cannot drift.
  `get_album_image_keywords()` is a deliberately cheap read (`_filter=KeywordArray,Keywords,
  IsVideo`, ~0.24s for 13 images against ~1s for the full expansion) used only to answer "has
  this album been processed?" — it does not build `AlbumImage` objects.
  `_request()` uses `requests.request()` directly with a per-call OAuth1 signature and holds no
  shared session, which is why the proof-state scanner can run 8 albums concurrently.
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
  Jobs are in memory only: **restarting the server invalidates every open proof sheet.**

  - **Which images a run covers is settled when the job is created.** `images_to_proof()` drops
    videos and, unless `force_reprocess`, images already carrying the marker tag; both the count
    and the processing loop go through it, so they cannot disagree. `force_reprocess` therefore
    lives on the `PreviewJob`, and `process_preview()` takes no such argument — the SSE route
    still accepts the old query param and ignores it.
  - **Proof-state badges** — `GET /api/albums/proof-state?keys=a,b,c` reports `all` / `partial` /
    `none` / `empty` per album so the picker can show what has been done. Separate from
    `/api/galleries` because it costs one request per 100 images per album (~4s for 19 albums);
    the list draws first and badges fill in. Cached for `PROOF_STATE_CACHE_TTL_SECONDS` (300).
    **SmugMug's `ImagesLastUpdated` does NOT move when image metadata is PATCHed**, so there is no
    timestamp to validate against: a commit invalidates its own album, `images_to_proof()`
    re-derives state for free when a run starts, and the TTL is the only thing that heals a write
    made by another process.
  - **Pets** — `GET/PUT /api/pets`, `DELETE /api/pets/<name>` over `processing/pets.py`.
  - **Hints** — `PUT /api/hints` takes `text`, `location`, `people` and `pets` together; the
    proof sheet saves all four in one call before re-reading a frame.
  - `PreviewJob.origin_node` remembers the folder the picker was showing, so the proof sheet can
    return there after a write (`/?node=<id>`). A run started from a pasted URL has none.
- `processing/` — `hints.py` (`HintManager`) owns `~/.smugvision/hints.yaml`: notes in three
  scopes (global/album/image, which **accumulate**), plus `locations:`, `people:` and `pets:`
  sections where the **most specific scope wins outright**. `pets.py` (`PetManager`) owns
  `~/.smugvision/pets.yaml`, a name → sentence map for animals. Both reload on mtime change, so a
  hand edit or another process's write is picked up without a restart, and both write atomically
  (`tempfile.mkstemp` + `os.replace`) preserving the existing file mode.
  `HintManager.people_usage()` tallies how often each person has been picked, which is how the
  web UI seeds its pinned-people list — a better signal than reference-photo count.
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

`~/.smugvision/` holds `config.yaml`, `hints.yaml`, `pets.yaml`, `locations.yaml`,
`relationships.yaml`, `geocoding_config.yaml`, `reference_faces/`, `cache/`, and `smugvision.log`.
Nothing under the repo is authoritative at runtime; `config.yaml.example`, `locations.yaml.example`
and `pets.yaml.example` are templates only.

**Treat `hints.yaml` as user data.** It accumulates hundreds of hand-written per-photo assertions
that exist nowhere else and cannot be regenerated. Back it up before any test that writes to it,
and remember that `PUT /api/hints` with `text: ""` **clears** the note for that key — the
"Cleared image hint" log line is emitted unconditionally, so it is not evidence that something was
actually removed.

### The web UI, as a workflow

Step 1 picks an album, step 2 proofs it, and the only write is behind a latch and a dialog.

- The picker shows a **proof badge** per album (`✓ proofed`, `7 of 11 proofed`, `not proofed`,
  `no photos` for an album that holds only videos), and the selection panel counts the frames a
  run would actually cover — not the whole album, since already-tagged frames are left out.
- The proof sheet carries the album note, a location override, an album-scope people picker
  (collapsed, because naming a whole album is the rare case; it opens itself and says so when an
  override IS set) and pet chips. Each frame card repeats all four at image scope.
- The **people picker pins a few people as large tiles** and files the rest behind a drawer.
  Pinned names live in `localStorage` (`smugvision.favouritePeople`), seeded from
  `picker_count` on `/api/faces`. Selection state lives in a closure, not the DOM, so
  re-rendering after a pin change cannot drop a tick; anyone already ticked is hoisted out of the
  drawer when a picker is built.
- **"Save & re-read all N frames"** saves the album note and re-reads every frame sequentially
  (one local Ollama process; parallel would not be faster), holding the write path shut for the
  whole sweep and offering a stop that finishes the frame in flight.
- After a **clean** write the sheet returns to `/?node=<origin>` with the confirmation carried in
  the query string and then stripped. A partial write stays put — that is when you need to see
  what failed.

If you touch this UI: a hidden `<input type="checkbox">` needs its checked state drawn somewhere,
or ticking looks like nothing happened. That shipped as a bug once because the tests clicked the
input directly (`box.click()`) — the one path a user never takes. Click the label.

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

- `processing.generate_captions` / `.generate_tags` — read by nothing; both are always
  produced. Honouring them is now cheap: `generate_metadata` treats an empty
  `caption_instruction` or `tags_instruction` as "skip that half".
- `tests/test_vision.py` hardcodes `llama3.2-vision` rather than reading `vision.model`, so the
  repo's main hand-run smoke test fails with a model-not-found 404 on a machine that does not
  have that exact model. That failure is not evidence of a broken vision layer.

`face_recognition.tolerance` / `.model` / `.detection_scale` **are** now wired: the processor
reads the `face_recognition.<backend>` sub-block into `backend_options` and folds those three
legacy top-level keys in, with the sub-block winning on conflict. (Earlier revisions of this file
said otherwise.)

### Known-unfixed

- `vision.single_call: false` **and** `structured_output: false` together raise on any tags-only
  free-text response. Pre-existing; both legacy paths are otherwise live.

## Conventions

Google-style docstrings with Args/Returns/Raises on all public methods, type hints throughout,
100-char lines, `logger = logging.getLogger(__name__)` per module. Optional heavy dependencies
(`face_recognition`, `insightface`/`onnxruntime`, `geopy`, `exifread`, `pillow_heif`, `httpx`) are
imported in try/except behind an `X_AVAILABLE` flag and degrade gracefully — preserve that.
HEIC/HEIF support comes from `pillow_heif.register_heif_opener()` in `vision/llama.py`.
Conventional Commits for commit messages (`feat:`, `fix:`, `docs:`).

`DESIGN.md` is the long-form architecture and roadmap doc; the topic-specific `README_*.md` files
cover config, processor, face recognition, and SmugMug testing in more depth.
