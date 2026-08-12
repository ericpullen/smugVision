# smugVision Image Processor

The `ImageProcessor` is the core orchestration module that ties together all smugVision components to automatically generate and update metadata for SmugMug photos.

## Overview

The processor handles the complete pipeline:

1. **Download** - Fetches images from SmugMug to local cache
2. **Extract** - Pulls EXIF data (GPS, date, camera info)
3. **Detect** - Identifies faces using face recognition
4. **Analyze** - Generates captions and tags using AI vision model
5. **Format** - Combines AI output with EXIF data and person names
6. **Update** - Pushes metadata back to SmugMug
7. **Mark** - Adds marker tag to track processed images

## Quick Start

### Test the Processor

```bash
# Process an album (dry run first to preview)
python test_processor.py b38H7c --dry-run

# Actually process the album
python test_processor.py b38H7c

# Force reprocess already-tagged images
python test_processor.py b38H7c --force-reprocess

# Verbose output
python test_processor.py b38H7c --verbose
```

### Use in Code

```python
from smugvision.config import ConfigManager
from smugvision.processing import ImageProcessor

# Load config
config = ConfigManager.load()

# Create processor
processor = ImageProcessor(
    config=config,
    dry_run=False  # Set True to preview without updating
)

# Process entire album
stats = processor.process_album(
    album_key="b38H7c",
    force_reprocess=False,
    skip_videos=True
)

# Check results
print(f"Processed: {stats.processed}")
print(f"Skipped: {stats.skipped}")
print(f"Errors: {stats.errors}")
```

## Architecture

### ImageProcessor Class

The main orchestrator that coordinates all components:

```python
ImageProcessor(
    config: ConfigManager,           # Configuration
    smugmug_client: SmugMugClient,  # Optional: SmugMug API
    vision_model: VisionModel,       # Optional: AI vision model
    cache_manager: CacheManager,     # Optional: Image cache
    face_recognizer: FaceRecognizer, # Optional: Face detection
    dry_run: bool = False           # Preview mode
)
```

**Key Methods:**

- `process_album(album_key, force_reprocess, skip_videos)` - Process all images in album
- `process_image(image, album, force_reprocess)` - Process single image

### MetadataFormatter Class

Handles metadata formatting and combination:

```python
MetadataFormatter(
    preserve_existing: bool = True,  # Keep existing captions/tags
    marker_tag: str = "smugvision"   # Processing marker
)
```

**Key Methods:**

- `format_caption(ai_caption, existing_caption, location, person_names)` - Combine caption sources
- `format_tags(ai_tags, existing_tags, person_names, location_tags)` - Merge tags
- `create_update_payload(caption, tags, title)` - Create SmugMug API payload

## Processing Pipeline

### Step-by-Step Flow

```
For each image in album:
  │
  ├─ Check marker tag
  │  └─ Skip if already processed (unless force_reprocess)
  │
  ├─ Download to cache
  │  └─ Skip if already cached
  │
  ├─ Extract EXIF data
  │  ├─ GPS coordinates → reverse geocode to location
  │  ├─ Camera info
  │  └─ Date/time
  │
  ├─ Detect faces
  │  ├─ Find faces in image
  │  ├─ Match against reference faces
  │  └─ Get person names with confidence
  │
  ├─ Generate AI metadata  (ONE request: generate_metadata)
  │  ├─ Combine the caption + tags instructions into one prompt
  │  ├─ Append context: album name, location, people, relationships
  │  ├─ Encode the image once (downscaled to max_image_dimension)
  │  └─ One JSON-schema-constrained chat call → {caption, tags}
  │     (vision.single_call: false restores the legacy two-request path)
  │
  ├─ Format metadata
  │  ├─ Merge AI caption with location + person names
  │  ├─ Combine tags from AI, faces, and location
  │  └─ Add marker tag
  │
  └─ Update SmugMug
     ├─ PATCH image metadata
     └─ Log results
```

### Example Output

```
Processing album: 2025:03:26 Grand Finale Cleaning House (16 items)
Skipping 1 video file(s)

[1/15] Processing: IMG_9887.JPG
  Identified 2 of 3 face(s): Alice, Bob
  Result: ✓ Success (12.3s)

[2/15] Processing: IMG_9888.JPG
  Result: ○ Skipped (already processed)

...

Processing Complete:
  Total images:    15
  Processed:       12
  Skipped:         2
  Errors:          1
  Total time:      145.6s
  Avg time/image:  9.7s
```

## Configuration

### Required Settings

```yaml
# SmugMug credentials
smugmug:
  api_key: "YOUR_KEY"
  api_secret: "YOUR_SECRET"
  user_token: "YOUR_TOKEN"
  user_secret: "YOUR_SECRET"

# Vision model (any vision-capable model from `ollama list`)
vision:
  model: "qwen3-vl:8b"
  endpoint: "http://localhost:11434"
  think: false                    # Disable reasoning; see README_CONFIG.md
  keep_alive: "30m"               # Keep the model loaded between images
  single_call: true               # One request per image for caption + tags
  structured_output: true         # JSON-schema-constrained reply
  max_image_dimension: 1568       # Downscale long edge before upload

# Processing options
processing:
  marker_tag: "smugvision"        # Tag for processed images
  generate_captions: true         # Enable captions
  generate_tags: true             # Enable tags
  preserve_existing: true         # Keep existing metadata
  image_size: "Medium"            # Download size

# Prompts (instructions only - album/location/people context is appended
# by the vision layer). See smugvision/config/defaults.py for the shipped text.
prompts:
  caption: "You are a photo captioning assistant. Write exactly ONE caption..."
  tags: "Output a comma-separated list of 5-10 keyword tags for this image..."
```

### Optional Settings

```yaml
# Face recognition
face_recognition:
  enabled: true
  reference_faces_dir: "~/.smugvision/reference_faces"
  backend: "dlib"                 # "dlib" (default) | "insightface" (optional)
  min_confidence: 0.25

# Cache
cache:
  directory: "~/.smugvision/cache"
  preserve_structure: true

# EXIF/Geocoding
exif:
  enable_geocoding: true
  reverse_geocoding:
    provider: "nominatim"
```

## Data Models

### ProcessingResult

Result of processing a single image:

```python
@dataclass
class ProcessingResult:
    image_key: str              # SmugMug image key
    filename: str               # Image filename
    success: bool               # Processing succeeded
    skipped: bool              # Already processed
    caption_generated: bool    # Caption was created
    tags_generated: int        # Number of tags created
    faces_detected: int        # Number of faces found
    processing_time: float     # Seconds taken
    error: Optional[str]       # Error message if failed
```

### BatchProcessingStats

Statistics for album processing:

```python
@dataclass
class BatchProcessingStats:
    total_images: int          # Total in album
    processed: int             # Successfully processed
    skipped: int               # Skipped (already done)
    errors: int                # Failed
    total_time: float          # Total seconds
    results: List[ProcessingResult]  # Individual results
```

## Error Handling

### Strategy

The processor uses **continue-on-error** by default:

- Individual image failures don't stop album processing
- All errors are logged with full context
- Statistics track success/skip/error counts
- Failed images can be retried with `--force-reprocess`

### Common Errors

**SmugMugError**: API access issues
```python
try:
    stats = processor.process_album(album_key)
except SmugMugError as e:
    logger.error(f"SmugMug API error: {e}")
```

**Vision Model Timeout**: AI model not responding
```bash
# Check Ollama is running and that vision.model is one of the models it serves:
ollama serve
ollama list
ollama pull <the model named in vision.model>
```
`vision.timeout` is applied to the underlying HTTP client; raise it on slow hardware.

**Face Recognition Error**: Reference faces not found
```python
# Check reference faces directory:
ls ~/.smugvision/reference_faces/
```

## Performance

### Typical Processing Times

- **Image download**: 0.5-2s (depends on size and network)
- **EXIF extraction**: 0.1-0.3s
- **Face detection**: 1-3s (if enabled)
- **AI caption + tags**: model- and hardware-dependent. On `gemma4:latest` with the
  defaults (`single_call: true`, `structured_output: true`, `think: false`) the single
  request measured **1.76-2.77s warm**, with a **9.57s cold first request**. See
  [`TIMING_ANALYSIS.md`](TIMING_ANALYSIS.md) for the full measurement table and its
  caveats — those numbers are for one model on one machine.
- **SmugMug update**: 0.2-0.5s

The per-image total has not been re-measured end to end since the vision rewrite; do not
quote one.

### Optimization Tips

1. **Use cached images**: Skip re-downloading with `cache.preserve_structure: true`
2. **Medium size images**: Balance quality vs speed (`image_size: "Medium"`)
3. **Disable face recognition**: If not needed, saves 1-3s per image
4. **Batch processing**: Process albums in off-peak hours
5. **Keep the model resident**: `vision.keep_alive` stops Ollama unloading the model
   between images, so only the first image of a run pays load time
6. **Downscale before upload**: `vision.max_image_dimension` (default 1568) shrinks the
   base64 payload — a 3840x2880 JPEG measured 5.59 MB encoded at full size versus
   1.07 MB at 1568px, for input the model tiles down anyway
7. **Leave `structured_output: true`**: unconstrained free-text replies ramble toward
   `max_tokens` and measured slower in every configuration tested

## Advanced Usage

### Custom Vision Model

Any vision-capable model that Ollama serves works — the factory has no allow-list, so
switching models is normally just a `vision.model` config change. To construct one
explicitly and inject it:

```python
from smugvision.vision import VisionModelFactory

# model_name is whatever `ollama list` reports.
vision = VisionModelFactory.create(
    "gemma4:latest",
    endpoint="http://localhost:11434",
    timeout=120,
    think=False,
    keep_alive="30m",
    single_call=True,
    structured_output=True,
)

processor = ImageProcessor(
    config=config,
    vision_model=vision
)
```

`create(model_name, endpoint=None, **kwargs)` forwards `**kwargs` to the model
constructor. To see what a running Ollama can serve:

```python
VisionModelFactory.list_models("http://localhost:11434")  # vision-capable models
```

For a genuinely different implementation — a non-Ollama backend, say — subclass
`VisionModel` (implementing `generate_metadata`, `generate_caption` and `generate_tags`)
and register it; a registration takes precedence over the default adapter:

```python
VisionModelFactory.register_model("my-backend", MyVisionModel)
```

### Process Specific Images

```python
# Get album images
images = smugmug_client.get_album_images(album_key)

# Filter specific images
selected = [img for img in images if "2025" in img.file_name]

# Process individually
for image in selected:
    result = processor.process_image(
        image=image,
        album=album,
        force_reprocess=True
    )
```

### Custom Metadata Formatting

```python
from smugvision.processing import MetadataFormatter

# Create custom formatter
formatter = MetadataFormatter(
    preserve_existing=False,  # Replace existing metadata
    marker_tag="my-custom-tag"
)

# Format manually
caption = formatter.format_caption(
    ai_caption="A beautiful sunset",
    location="Golden Gate Bridge, San Francisco",
    person_names=["Alice", "Bob"]
)
# Result: "A beautiful sunset. Featuring Alice and Bob at Golden Gate Bridge, San Francisco."
```

## Testing

### Unit Tests (TODO)

```bash
pytest tests/test_processor.py
pytest tests/test_metadata.py
```

### Integration Test

```bash
# Test with small album first
python test_processor.py TEST_ALBUM --dry-run --verbose

# Then process for real
python test_processor.py TEST_ALBUM
```

## Troubleshooting

### "No images to process"

- Check album key is correct
- Verify album has images (not just videos)
- Check `skip_videos` setting

### "Authentication failed"

- Verify SmugMug credentials in config.yaml
- Check token hasn't expired
- Run `smugvision-get-tokens` to refresh

### "Vision model timeout"

- Ensure Ollama is running: `ollama serve`
- Check model is downloaded: `ollama list`
- Try smaller/faster model if M4 Pro struggles

### "All images skipped"

- Images already have marker tag
- Use `--force-reprocess` to process again
- Or change `marker_tag` in config

## Next Steps

- [ ] Add parallel processing for faster throughput
- [ ] Implement resume capability for interrupted processing
- [ ] Add progress bar with ETA
- [ ] Support folder-level recursive processing
- [ ] Add metadata backup before updates
- [ ] Create rollback functionality

## See Also

- [DESIGN.md](DESIGN.md) - Overall architecture
- [README_CONFIG.md](README_CONFIG.md) - Configuration guide
- [README_SMUGMUG_TESTING.md](README_SMUGMUG_TESTING.md) - SmugMug API testing

