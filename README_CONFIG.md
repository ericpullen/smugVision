# smugVision Configuration Guide

This guide explains how to configure smugVision for your SmugMug gallery processing needs.

## Quick Start

1. **Copy the example configuration file to your user directory:**
   ```bash
   mkdir -p ~/.smugvision
   cp config.yaml.example ~/.smugvision/config.yaml
   ```

2. **Edit `~/.smugvision/config.yaml` with your SmugMug credentials:**
   - Get API credentials from https://api.smugmug.com/api/developer/apply
   - Fill in your `api_key`, `api_secret`, `user_token`, and `user_secret`

3. **Test your configuration:**
   ```bash
   python test_config.py
   ```

## Configuration File Location

smugVision searches for configuration files in the following order:

1. `~/.smugvision/config.yaml` (user home directory - **primary location**)
2. `./config.yaml` (current directory - for development/testing)

You can also specify a custom path when loading the configuration programmatically.

## Interactive Configuration Setup

If no configuration file is found, smugVision will prompt you for required values:

```bash
$ python test_config.py

======================================================================
smugVision Configuration Setup
======================================================================

Some required configuration values are missing.
Please provide the following information:

SmugMug API Key (from https://api.smugmug.com/api/developer/apply)
smugmug.api_key: YOUR_API_KEY

SmugMug API Secret
smugmug.api_secret: YOUR_API_SECRET

SmugMug User OAuth Token
smugmug.user_token: YOUR_USER_TOKEN

SmugMug User OAuth Secret
smugmug.user_secret: YOUR_USER_SECRET

======================================================================
Configuration setup complete!
======================================================================
```

The configuration is saved to `~/.smugvision/config.yaml`.

## Configuration Sections

### SmugMug API Configuration

Required fields for SmugMug API access:

```yaml
smugmug:
  api_key: "YOUR_API_KEY_HERE"
  api_secret: "YOUR_API_SECRET_HERE"
  user_token: "YOUR_USER_TOKEN_HERE"
  user_secret: "YOUR_USER_SECRET_HERE"
```

### Vision Model Configuration

Settings for the local vision model served by Ollama. **Any vision-capable model works** —
there is no allow-list in the code, so switching models is a config change, not a code
change. Run `ollama list` to see what you have installed.

```yaml
vision:
  model: "qwen3-vl:8b"               # Any model from `ollama list`
  endpoint: null                     # null = use $OLLAMA_HOST, else localhost:11434
  temperature: 0.7                   # Creativity (0.0 = deterministic, 1.0 = creative)
  max_tokens: 500                    # Budget for ONE reply holding caption + tags
  timeout: 120                       # API timeout in seconds
  think: false                       # false | "low" | "medium" | "high" | null
  keep_alive: "30m"                  # Keep the model resident between images
  single_call: true                  # One request per image for caption + tags
  structured_output: true            # Constrain the reply with a JSON schema
  max_image_dimension: 1568          # Downscale long edge before upload; 0/null disables
  jpeg_quality: 85                   # JPEG quality when re-encoding for transport
  validate_model: true               # Warn (never fail) if the model is not installed
```

The shipped defaults live in `smugvision/config/defaults.py`. Every key you leave out of
your own `config.yaml` falls back to the default there, so a short config file is fine.

**Tips:**

- **`temperature`** — lower (0.3-0.5) for consistent, factual captions; higher (0.7-0.9)
  for more varied descriptions.
- **`max_tokens`** — this is one budget covering the caption *and* the tags, because they
  now arrive in one reply. If output is being cut off, raise it; but first check `think`.
- **`think`** — the usual cause of an empty or truncated reply on a reasoning model. With
  reasoning enabled the model can consume the entire `max_tokens` budget before producing
  any content. `false` (the default) disables it. `null` omits the parameter and lets the
  model choose, which is slower.
- **`keep_alive`** — the usual cause of "the first image of every album is slow". Without
  it Ollama can unload the model between requests, so each image pays load time.
- **`single_call`** — `true` sends one request per image. Set it to `false` for the legacy
  path of a separate caption request and tags request. Worth trying only if a model
  produces noticeably worse output when asked for both at once.
- **`structured_output`** — `true` constrains the reply with a JSON schema so it parses
  deterministically. Set it to `false` for older or smaller models that ignore or mishandle
  schemas; the reply is then parsed with the legacy free-text heuristics. Treat this as a
  compatibility fallback, not a neutral choice — unconstrained replies tend to ramble
  toward `max_tokens` and are slower.
- **`max_image_dimension`** — vision models tile input to roughly 1024-1568px, so sending
  a 3840px original is wasted bandwidth. `0` or `null` disables downscaling; images are
  never upscaled.
- **`validate_model`** — warns at startup when `model` is missing from Ollama's tag list.
  It never aborts the run.
- **`timeout`** — raise it on slower systems. It is applied to the underlying HTTP client.

### Face Recognition Configuration

Settings for face detection and recognition:

```yaml
face_recognition:
  enabled: true                      # Enable/disable face recognition
  reference_faces_dir: "~/.smugvision/reference_faces"
  backend: "dlib"                    # "dlib" (default) | "insightface" (optional)

  # dlib-only settings
  tolerance: 0.6                     # Euclidean distance ceiling (lower = stricter)
  model: "cnn"                       # dlib DETECTOR: 'hog' or 'cnn' - NOT the backend
  detection_scale: 0.5               # Image scale for detection (0.1-1.0)

  # backend-agnostic
  min_confidence: 0.25               # Normalized confidence threshold (0.0-1.0)
  use_cache: true                    # Cache reference encodings between runs
  cache_dir: "~/.smugvision/cache/face_encodings"

  # insightface-only settings
  insightface:
    model_name: "buffalo_l"
    det_size: [640, 640]
    similarity_threshold: 0.4        # COSINE similarity - HIGHER is stricter
```

**Choosing a backend:**

- `backend: "dlib"` is the default and needs nothing beyond the core install.
- `backend: "insightface"` (ArcFace via ONNX Runtime) is **optional and experimental**.
  Install it with `pip install -e ".[insightface]"`. If its dependencies are missing,
  smugVision logs an error and falls back to dlib rather than failing the run. Its
  recognition accuracy has not been benchmarked against dlib in this project — only
  mechanical correctness was verified.

Each backend keeps its own encoding cache file, so switching back and forth does not force
a re-encode and the two vector formats can never be mixed.

**Which knob belongs to which backend:**

| Key | Applies to | Meaning |
|---|---|---|
| `tolerance` | dlib only | Euclidean distance ceiling. **Lower is stricter.** |
| `model` | dlib only | The *detector* (`hog` = faster, `cnn` = more accurate) |
| `detection_scale` | dlib only | Pre-detection downscale; InsightFace uses `det_size` instead |
| `insightface.similarity_threshold` | insightface only | Cosine similarity. **Higher is stricter.** |
| `min_confidence` | both | Normalized 0.0-1.0 score after the backend's own threshold |

`tolerance` and `similarity_threshold` are different metrics running in opposite
directions — do not copy a value from one to the other.

> **Known gap:** `ImageProcessor` does not currently forward the top-level `tolerance`,
> `model` or `detection_scale` values to the recognizer, so the built-in dlib defaults
> (0.6 / `cnn` / 0.5) apply regardless of what you set there. Until that call site is
> wired up, put them in a `face_recognition.dlib` sub-block instead, which *is* passed
> through as backend options:
>
> ```yaml
> face_recognition:
>   backend: "dlib"
>   dlib:
>     tolerance: 0.5
>     model: "hog"
>     detection_scale: 0.75
> ```
>
> `min_confidence` is forwarded and does take effect.

**Face Recognition Setup:**

1. Create reference faces directory structure:
   ```
   ~/.smugvision/reference_faces/
   ├── John_Doe/
   │   ├── photo1.jpg
   │   ├── photo2.jpg
   │   └── vacation.png
   ├── Jane_Smith/
   │   ├── profile.jpg
   │   └── headshot.heic
   └── ...
   ```

2. Each person should have their own subdirectory
3. Use the person's name as the directory name (underscores will be replaced with spaces)
4. Add multiple clear photos of each person for better accuracy

**Parameter Tuning (dlib backend):**
- **tolerance**: Lower values (0.4-0.5) = stricter matching, fewer false positives
- **tolerance**: Higher values (0.6-0.7) = more lenient, may catch more faces but with more false positives
- **model**: `cnn` is more accurate but slower, `hog` is faster but less accurate. This is
  the face *detector*, not the backend — the backend is `backend`.
- **detection_scale**: Lower values = faster processing but may miss distant faces
- **min_confidence**: Higher values = only include high-confidence matches in results.
  This one is backend-agnostic: every backend reports a normalized 0.0-1.0 score where
  0.0 sits exactly at that backend's own match threshold and 1.0 is a perfect match.

**Parameter Tuning (insightface backend):**
- **insightface.similarity_threshold**: cosine similarity, higher is stricter. 0.4 is the
  default; raise it toward 0.5 for fewer false positives.
- **insightface.det_size**: larger finds smaller/more distant faces at more cost.
- `tolerance`, `model` and `detection_scale` are ignored by this backend.

### Processing Configuration

Settings for image processing behavior:

```yaml
processing:
  marker_tag: "smugvision"           # Tag to mark processed images
  generate_captions: true            # NOT WIRED - captions are always produced
  generate_tags: true                # NOT WIRED - tags are always produced
  generate_titles: false             # Also propose a short SmugMug Title (3-6 words)
  preserve_existing: true            # Keep existing metadata
  image_size: "Medium"               # Download size from SmugMug (case-insensitive)
  use_exif_location: true            # Extract GPS location from EXIF
```

`marker_tag` is the idempotency mechanism: an image carrying it is left out of a run
entirely unless `--force-reprocess` is passed (or the matching box ticked in the web UI).

`generate_titles` is off by default. The title comes back from the same single request, so it
costs nothing extra, but it writes a SmugMug field nothing else touches. Both the CLI and the
web UI can override `preserve_existing` and `generate_titles` per run.

### Hint Configuration

Hints are facts you assert about a photo, injected into the prompt as ground truth and
outranking whatever the model thinks it sees.

```yaml
hints:
  enabled: true                      # false ignores hints.yaml entirely
  file: "~/.smugvision/hints.yaml"   # created on first write; safe to hand-edit
```

The file itself holds notes at three scopes (`global`, `albums`, `images`, which
accumulate) plus `locations:`, `people:` and `pets:` sections, where the most specific
scope wins outright. Pet *definitions* live separately in `~/.smugvision/pets.yaml`; see
the main [README](README.md#telling-smugvision-what-it-cannot-see) for the whole picture.

Both front ends read the same file, so a hint added in the web UI takes effect on the next
CLI run and vice versa.

### Prompt Configuration

Customize how the AI describes your images. These are *instructions*, not the whole
prompt: the album name, resolved location, recognized people, and any relationships from
`~/.smugvision/relationships.yaml` are appended as context by the vision layer before the
request goes out. With `single_call: true` both instructions are combined into one prompt.

```yaml
prompts:
  caption: |
    You are a photo captioning assistant. Write exactly ONE caption (1-2 sentences)
    for this image. Describe the main subject, setting, and activity.
    IMPORTANT: Output ONLY the caption text. No options, no explanations, no
    introductions like 'Here is...' - just the caption itself.

  # Only used when processing.generate_titles is on.
  title: |
    Also give a very short title for this image: 3-6 words, like a label in a photo
    album. Not a sentence. No trailing punctuation.

  tags: |
    Output a comma-separated list of 5-10 keyword tags for this image.
    IMPORTANT: Output ONLY the tags separated by commas. No explanations,
    no numbering, no extra text. Example output: dog, playing, park, sunny, happy
```

The values above are the shipped defaults (see `smugvision/config/defaults.py`).

**Tips for custom prompts:**
- Say what to focus on (subjects, activities, mood, etc.)
- Keep prompts concise and clear
- Output-format instructions ("comma-separated list", "no introductions") are redundant
  under `structured_output: true`, where a JSON schema already forces the shape. They
  still matter with `structured_output: false`, which is why the defaults keep them.

### Cache Configuration

Settings for local image caching:

```yaml
cache:
  directory: "~/.smugvision/cache"   # Local cache directory
  clear_on_exit: false               # Auto-clear cache after processing
  preserve_structure: true           # Mirror SmugMug folder structure
```

### Logging Configuration

Settings for logging output:

```yaml
logging:
  level: "INFO"                      # DEBUG, INFO, WARNING, ERROR, CRITICAL
  file: "~/.smugvision/smugvision.log"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

**Log Levels:**
- **DEBUG**: Very detailed information, useful for troubleshooting
- **INFO**: General information about processing progress
- **WARNING**: Important notices (e.g., face not recognized)
- **ERROR**: Errors that prevent specific images from processing
- **CRITICAL**: Fatal errors that stop the entire process

## Using ConfigManager in Python

You can use the `ConfigManager` class in your own scripts:

```python
from smugvision.config import ConfigManager

# Load configuration (will prompt for missing values if interactive)
config = ConfigManager.load()

# Get configuration values using dot notation
model_name = config.get("vision.model")
temperature = config.get("vision.temperature")
api_key = config.get("smugmug.api_key")

# Get with default value if not found
timeout = config.get("vision.timeout", 60)

# Set configuration values
config.set("vision.temperature", 0.8)
config.set("processing.marker_tag", "ai-processed")

# Save changes
config.save()

# Get entire config as dictionary
config_dict = config.to_dict()
```

### Loading Specific Config File

```python
# Load from specific path
config = ConfigManager.load(config_path="/path/to/config.yaml")

# Non-interactive mode (will raise error if required fields missing)
config = ConfigManager.load(interactive=False)

# Don't create config if missing (will raise error)
config = ConfigManager.load(create_if_missing=False)
```

### Error Handling

```python
from smugvision.config import ConfigManager, ConfigError

try:
    config = ConfigManager.load()
except ConfigError as e:
    print(f"Configuration error: {e}")
    sys.exit(1)
```

## Configuration Best Practices

1. **Keep credentials secure:**
   - Never commit `config.yaml` to version control
   - Add `config.yaml` to `.gitignore`
   - Use environment variables for CI/CD

2. **Start with defaults:**
   - Copy `config.yaml.example` and modify as needed
   - Only change values you need to customize

3. **Test your configuration:**
   - Run `python test_config.py` to validate settings
   - Use `test_vision.py` to test with actual images

4. **Face recognition setup:**
   - Use clear, front-facing photos for reference faces
   - Include 2-3 photos per person for better accuracy
   - Test with known images before processing entire galleries

5. **Prompt tuning:**
   - Start with default prompts
   - Adjust based on results from test images
   - Be specific about desired output format

## Troubleshooting

### "Missing required configuration fields" error

Make sure all SmugMug API credentials are filled in:
- `smugmug.api_key`
- `smugmug.api_secret`
- `smugmug.user_token`
- `smugmug.user_secret`

### Face recognition not working

1. Check that reference faces directory exists and has correct structure
2. Ensure `face_recognition` library is installed:
   ```bash
   pip install face_recognition
   pip install git+https://github.com/ageitgey/face_recognition_models
   ```
3. Try adjusting `tolerance` and `min_confidence` values
4. Use `cnn` model for better accuracy (but slower)

### Configuration file not found

smugVision searches for configuration in:
1. `~/.smugvision/config.yaml` (primary location)
2. `./config.yaml` (current directory)

The recommended location is `~/.smugvision/config.yaml` for consistency with other smugVision configuration files (relationships.yaml, geocoding_config.yaml).

Either create a config file in one of these locations, or specify the path explicitly.

### YAML parsing errors

- Check for proper indentation (use spaces, not tabs)
- Ensure strings with special characters are quoted
- Validate YAML syntax online: https://www.yamllint.com/

## Support

For more information, see:
- Main README: `README.md`
- Face Recognition Guide: `README_FACE_RECOGNITION.md`
- Design Document: `DESIGN.md`

