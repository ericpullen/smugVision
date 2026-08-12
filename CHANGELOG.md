# Changelog

All notable changes to smugVision will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Vision-layer rewrite: one structured model call per image, a factory that accepts any
Ollama model, and an optional second face recognition backend.

### Added

- `feat(vision)`: `VisionModel.generate_metadata(image_path, caption_instruction,
  tags_instruction, *, temperature, max_tokens, location_context, person_names,
  total_faces, album_name) -> MetadataResult`. Builds one prompt from both instructions
  plus all context, encodes the image once, and makes a single `chat` request constrained
  by a JSON schema. Passing `""` for either instruction skips that half of the reply.
- `feat(vision)`: new config keys under `vision:` — `think`, `keep_alive`, `single_call`,
  `structured_output`, `max_image_dimension`, `jpeg_quality`, `validate_model`. See
  `config.yaml.example` for the annotated list.
- `feat(vision)`: `vision.keep_alive` (default `"30m"`) keeps the model resident between
  images, so only the first image of a run pays load time.
- `feat(vision)`: `vision.max_image_dimension` (default 1568) downscales the long edge
  before base64 encoding. On one real 3840x2880 JPEG the payload went from 5.59 MB to
  1.07 MB. Images are never upscaled; `0`/`null` disables.
- `feat(vision)`: `vision.validate_model` warns — never fails — when the configured model
  is absent from Ollama's tag list.
- `feat(vision)`: `VisionModelFactory.registered_models()`; `list_models()` now takes
  `endpoint`/`vision_only`/`timeout` and reports what Ollama actually serves.
- `feat(face)`: optional, **experimental** InsightFace (ArcFace ONNX) backend selected by
  `face_recognition.backend: "insightface"`, with its own `face_recognition.insightface`
  options block (`model_name`, `det_size`, `similarity_threshold`). Install with
  `pip install -e ".[insightface]"`. **dlib remains the default.** Accuracy versus dlib
  has not been benchmarked; only mechanical correctness was verified.
- `feat(face)`: `face/backends/` package — `FaceBackend` ABC, `DlibFaceBackend`,
  `InsightFaceBackend`, `create_backend`/`register_backend`. Every backend reports a
  normalized confidence (0.0 at its own threshold, 1.0 at a perfect match), so
  `min_confidence` means the same thing regardless of backend.
- `feat(face)`: `FaceRecognizer(..., *, backend=..., backend_options=...)`, both
  keyword-only and appended after `use_cache`; existing positional callers are unaffected.
- `feat(config)`: `face_recognition.backend` (default `"dlib"`) and the
  `face_recognition.insightface` sub-block.

### Changed

- `perf(vision)`: **one inference call per image instead of two**, and one base64 encode
  instead of two. On `gemma4:latest` the single structured call measured 1.76-2.77s warm
  versus 3.24-4.00s for the two-call paths; see `TIMING_ANALYSIS.md` for the full table
  and its caveats (one model, one machine, inference only).
- `feat(vision)`!: `VisionModelFactory` no longer enforces a hard-coded allow-list of
  model names. Any Ollama model name is accepted and mapped to the single Ollama adapter,
  so using a new model needs no code change. `register_model()` remains as an override
  hook for genuinely different implementations and takes precedence.
- `feat(vision)`: `vision.think` defaults to `false`. Reasoning models can otherwise
  consume the entire `max_tokens` budget before emitting content — reproduced at
  `max_tokens: 400` with `think: "low"`, which needed 1200 tokens to succeed.
- `docs`: `max_tokens` is now a single budget covering the caption **and** the tags,
  because they arrive in one reply.
- `refactor(face)`: `FaceRecognizer` is now a backend-agnostic coordinator; detection and
  embedding moved into `face/backends/`. Its public surface (constructor signature,
  `load_reference_faces`/`identify_faces`/`get_person_names`/`get_face_count`/
  `clear_cache`/`get_cache_info`, and the `.reference_faces`/`.tolerance`/`.model`/
  `.detection_scale` attributes) is unchanged; dlib results were verified bit-identical
  against a 14-image baseline.
- `refactor(vision)`: `LlamaVisionModel` is the generic Ollama adapter, not a
  Llama-specific one. The class name is historical.

### Fixed

- `fix(smugmug)`: **the write path.** `ImageProcessor` passed
  `MetadataFormatter.create_update_payload()`'s dict as the second *positional* argument
  to `update_image_metadata(image_key, caption=None, keywords=None, title=None)`, so the
  entire payload landed in `caption`. Both callers now use the keyword form.
- `fix(vision)`: `endpoint` and `timeout` were accepted, logged, and then ignored — every
  request went to hardcoded localhost with no timeout. The model now owns a real
  `ollama.Client(host=endpoint, timeout=timeout)` and all requests go through it.
- `fix(vision)`: `generate_tags()` never applied context enrichment, so
  `~/.smugvision/relationships.yaml` influenced captions but never tags.
- `fix(processing)`: `total_faces` was never supplied to the vision layer, leaving the
  "there are N people, one of them is X" prompt branch as dead code. It is now threaded
  through, and `result.faces_detected` reports faces *detected* rather than *recognized*.
- `fix(processing)`: `face_recognition.min_confidence` is now passed to
  `get_person_names()`; it was configured but never read. (`tolerance`, `model` and
  `detection_scale` are still not forwarded by `ImageProcessor` — see Known gaps.)
- `fix(vision)`: `_encode_image()` now applies EXIF orientation. Portrait-orientation
  photos were previously sent to the model rotated 90 degrees, since the re-encoded JPEG
  carries no EXIF block. Non-RGB modes (e.g. RGBA PNGs) are also flattened rather than
  failing on save.
- `fix(face)`: `face_recognition` calls `quit()` (raising `SystemExit`) rather than
  `ImportError` when its models are missing, so the guard did not catch it and importing
  `smugvision.face` could terminate the interpreter silently. Both are now caught.
- `fix(face)`: repeated `load_reference_faces()` calls accumulated duplicate encodings.
- `perf(face)`: detection results are memoized per file, so `get_person_names()` followed
  by `get_face_count()` on the same image no longer runs detection twice.
- `fix(deps)`: the declared floor is now `ollama>=0.5.0` (was `>=0.1.0`). Every request
  sends `think=`, which only exists in ollama-python 0.5.0+; dict `format` needs 0.4.0+.
  An older client already installed satisfied `>=0.1.0`, so `pip install -e .` left it in
  place and every image failed with `TypeError: Client.chat() got an unexpected keyword
  argument 'think'`. `_is_think_unsupported()` now also matches that client-side
  `TypeError`, so an old client degrades to a working call instead of erroring per image.
- `fix(web)`: `PreviewService.commit_changes()` sent `caption=""` when the model returned
  tags but no caption and `processing.preserve_existing` was false, wiping the user's
  existing SmugMug caption. It now passes `... or None` exactly like the CLI write path,
  and `None` leaves the field untouched.
- `fix(vision)`: with `vision.structured_output: false`, any response merely *containing*
  a `{...}` substring (a caption about a whiteboard, a receipt, a code screenshot) was
  treated as malformed JSON and failed the whole image, even when it was a perfectly
  parseable `CAPTION:`/`TAGS:` block. `_looks_like_json()` now requires the whole reply to
  read as JSON, so prose with braces reaches the free-text parser again.
- `fix(smugmug)`: `download_image()` matched the configured size case-sensitively
  (`f"{size}ImageUrl"`), so the shipped default `"medium"` never matched SmugMug's
  `MediumImageUrl` and silently fell through to `LargestImageUrl` — full-size originals
  for every image. The lookup (and the `Original`/`ArchivedUri` fallback) is now
  case-insensitive, and the shipped default reads `"Medium"`.
- `fix(config)`: `vision.endpoint` now defaults to `null` instead of the literal
  `http://localhost:11434`, and `ImageProcessor` no longer substitutes a literal fallback.
  Binding a real `ollama.Client` made the old default override `$OLLAMA_HOST` for everyone
  who had never set the key. The connection error also names `$OLLAMA_HOST` now.
- `docs`: corrected the documented default model, `max_tokens`, face detector default,
  the non-existent `processing.skip_videos` key, the config save location, the stale
  `python get_smugmug_tokens.py` / `python -m smugvision.config.manager --setup`
  invocations (in `QUICKSTART.md`, and now also `README.md` and `README_PROCESSOR.md` —
  `smugvision/config/manager.py` has no `__main__`/argparse, so `--setup` was a silent
  no-op), the flat reference-faces layout (it is one subdirectory per person), and
  the `VisionModelFactory.create_model(...)` snippet for a method that never existed.
- `docs`: `TIMING_ANALYSIS.md` and `README.md` claimed reverse-geocode results were
  cached. They are not: `utils/exif.py::reverse_geocode()` builds a fresh `Nominatim`
  geolocator per call and the only `_geocode_cache` lives in the unimported
  `utils/exif_optimized.py`. Both documents now say so.
- `docs`: the `README_FACE_RECOGNITION.md` usage example passed an unexpanded `~` path
  (silently loading zero reference faces) and hardcoded the `insightface` options block
  while the default backend is dlib. `FaceRecognizer.load_reference_faces()` now expands
  `~` as well.

### Known gaps

- `face_recognition.tolerance`, `face_recognition.model` and
  `face_recognition.detection_scale` are still **not forwarded** by `ImageProcessor` to
  `FaceRecognizer`, so the dlib defaults (0.6 / `cnn` / 0.5) apply no matter what those
  keys say. `FaceRecognizer` accepts all three; only the call site is missing. They can
  also be set through `backend_options` via a `face_recognition.dlib` sub-block, which
  the processor does read.
- `processing.generate_captions` and `processing.generate_tags` remain unread — the
  processor always produces both.

### Notes for upgraders

- `CACHE_VERSION` bumped 1 → 2 and the face encoding cache filename now carries the
  backend slug. One re-encode of your reference images happens on first run.
- `vision.single_call: false` and `vision.structured_output: false` restore the legacy
  two-request and free-text paths respectively. Both remain fully functional.
- There is still no automated test suite; `pytest` collects nothing. Verification for
  this release was import smoke tests, live runs against Ollama, black/flake8/mypy, and
  the hand-run scripts in `tests/`.

## [0.3.0] - 2025-11-24

### MVP Release - Production Ready! 🎉

smugVision is now a fully functional, production-ready tool for automated photo metadata generation!

### Added

**Core Processing:**
- `ImageProcessor` class for orchestrating the complete processing pipeline
- `MetadataFormatter` for combining AI-generated, EXIF, and face recognition metadata
- End-to-end processing with detailed statistics tracking
- Batch processing with progress indicators
- Processing result tracking with success/skip/error counts

**CLI Interface:**
- Production-ready `__main__.py` CLI entry point
- Rich formatted output with banners and summaries
- Support for `--gallery` (album key) and `--url` (SmugMug URL) inputs
- `--dry-run` flag for previewing changes without updating SmugMug
- `--force-reprocess` flag to reprocess already-tagged images
- `--include-videos` flag to process video files (skipped by default)
- `--verbose` and `--quiet` modes for logging control
- `--config` option for custom configuration files
- Exit codes for proper shell integration
- Comprehensive error messages with troubleshooting hints

**Face Recognition Enhancements:**
- Person name formatting (converts underscores to spaces)
- Configurable confidence thresholds via config.yaml
- Integration of identified people into captions and tags

**Testing & Utilities:**
- `test_processor.py` for testing the full processing pipeline
- Support for URL-based album resolution in test scripts
- Dry-run mode with detailed preview output

**Documentation:**
- Comprehensive README.md with installation, usage, and troubleshooting
- QUICKSTART.md for fast setup and first use
- Updated DESIGN.md with current status and architecture
- CHANGELOG.md for tracking releases

### Changed

- Album processing now returns `BatchProcessingStats` with detailed metrics
- Improved logging throughout the processing pipeline
- Enhanced error handling with specific exception types
- Metadata formatting now preserves existing captions/tags (configurable)

### Fixed

- Person names now display with spaces instead of underscores
- Reference faces directory path now properly expands tilde (~)
- Video downloads now use `LargestVideo` endpoint for actual video files
- Image downloads use correct size-specific URLs from `ImageSizes` expansion

---

## [0.2.0] - 2025-11-23

### Enhanced Media Handling & EXIF Integration

### Added

**SmugMug Integration:**
- Video file detection via `is_video` property
- `LargestVideo` endpoint support for proper video downloads
- Configurable video inclusion/exclusion (skip by default)
- Album resolution from URLs, node IDs, and names
- Recursive album search within folder structures
- URL path resolution for folder navigation
- Pagination support for large datasets
- Multiple image size options (Thumb through X3Large, Original)

**EXIF & Location:**
- EXIF data extraction with GPS coordinates
- Reverse geocoding for human-readable location names
- Location context integration into captions and tags
- HEIC/HEIF image format support
- Automatic orientation correction

**Testing:**
- `test_smugmug.py` with caching, URL parsing, and album listing
- `--cache` flag for local image downloads
- `--size` option for configurable download sizes
- `--force` flag for re-downloading existing files
- `--include-videos` flag in test script

### Changed

- Improved SmugMug API error messages
- Enhanced album key resolution logic
- Better handling of SmugMug folder hierarchies

### Fixed

- Video files now download correctly (not as thumbnails)
- Image downloads use proper size-specific URLs
- Content-type validation prevents HTML/JSON downloads
- Pagination now correctly fetches all results

---

## [0.1.0] - 2025-11-22

### Initial MVP - Core Infrastructure

### Added

**Configuration System:**
- YAML-based configuration (`~/.smugvision/config.yaml`)
- Interactive setup wizard
- Default values for all settings
- Configuration validation
- Support for required and optional fields

**SmugMug API Client:**
- OAuth 1.0a authentication
- Album and image retrieval
- Metadata updates (PATCH endpoint)
- Marker tag system for tracking processed images
- Error handling with custom exceptions
- Rate limiting awareness

**Vision Model Integration:**
- Factory pattern for vision models
- Abstract base class for extensibility
- Llama 3.2 Vision integration via Ollama _(superseded in [Unreleased]: any Ollama
  vision model is now accepted)_
- Caption generation with customizable prompts
- Tag generation with keyword extraction
- Temperature and max_tokens configuration

**Face Recognition:**
- Face detection and recognition using `face_recognition` library
- Reference faces management (folder-based organization)
- Multiple reference images per person
- Confidence-based matching
- Relationship context integration
- Face encoding optimization script

**Cache Management:**
- Local image caching with folder structure mirroring
- Skip existing files to avoid re-downloads
- Configurable cache directory
- Automatic directory creation

**Testing & Utilities:**
- `test_vision.py` for vision model testing
- `debug_face_recognition.py` for face detection debugging
- `get_smugmug_tokens.py` for OAuth token acquisition
- `find_album_key.py` for album discovery
- `optimize_reference_faces.py` for face encoding optimization

**Documentation:**
- DESIGN.md with architecture and roadmap
- README_FACE_RECOGNITION.md with face recognition guide
- config.yaml.example with all configuration options
- Code documentation and inline comments

### Core Components

- `smugvision.config` - Configuration management
- `smugvision.smugmug` - SmugMug API client and models
- `smugvision.vision` - Vision model abstraction and implementations
- `smugvision.face` - Face recognition system
- `smugvision.cache` - Cache management
- `smugvision.utils` - EXIF extraction, geocoding, and utilities

---

## Future Plans

See [DESIGN.md](DESIGN.md) for detailed roadmap. Key planned features:

### Version 1.0.0 (Future)
- Folder batch processing
- Cache cleanup utilities
- Unit and integration tests
- Performance optimizations
- Non-Ollama vision backends (cloud models)
- Docker deployment option

_Delivered since this list was written: the web UI (`smugvision-web`) and support for any
Ollama vision model._

---

**Legend:**
- 🎉 Major milestone
- ✨ New feature
- 🐛 Bug fix
- 📝 Documentation
- ♻️ Refactoring
- ⚡ Performance improvement

