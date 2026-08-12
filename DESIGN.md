# smugVision Design Document

**Version:** 1.0  
**Last Updated:** November 23, 2024  
**Platform:** macOS  
**Language:** Python 3.10+

---

## Project Overview

smugVision is a command-line tool that uses local AI vision models to automatically generate descriptive captions and relevant keyword tags for photos stored in SmugMug galleries. The tool processes images locally, generates metadata using AI, and updates SmugMug galleries directly via their API without requiring image re-uploads.

### Core Goals

1. Improve SmugMug photo searchability through AI-generated metadata
2. Process images locally for privacy and cost efficiency
3. Support batch processing of entire galleries or folders
4. Provide extensibility for future enhancements (face detection, custom prompts, etc.)
5. Maintain a clean, modular architecture

---

## Technical Stack

### Primary Components

- **Language:** Python 3.10+
- **AI Model:** any vision-capable model served by Ollama; the name lives in `vision.model`
  and switching models requires no code change
- **SmugMug Integration:** Python library or custom API wrapper
- **Configuration:** YAML or JSON config file
- **Logging:** Python standard logging library with timestamps and module identification

### Model Selection Rationale

The project originally hard-selected Llama 3.2 Vision 11B. That is no longer the case:
`VisionModelFactory` has no allow-list and maps every model name to the one Ollama
adapter, so the "default" is now purely the `vision.model` value in
`smugvision/config/defaults.py`, changeable in config alone.

What still guides the choice:
- Runs locally on Apple Silicon without API costs
- Strong image understanding and captioning
- Available through Ollama, so installation and lifecycle management are `ollama pull`

`vision.validate_model` (default true) checks at startup whether the configured model is
in Ollama's tag list and **warns** if it is not; it never aborts the run.

**Image size:** the client now downscales the image's long edge to
`vision.max_image_dimension` (default 1568px; `0`/`null` disables, never upscales) before
base64 encoding. Modern vision models tile input to roughly 1024-1568px, so the SmugMug
download size no longer has to be the resolution guard — `processing.image_size` is now a
bandwidth and cache-size choice rather than a correctness one. Measured on one real
3840x2880 JPEG: 5.59 MB base64 with downscaling off, 1.07 MB at 1568px.

### Alternative Models

Any vision-capable model Ollama serves already works with no code change — that part of
this section is done. What remains future work is **non-Ollama** backends:
- OpenAI GPT-4o/GPT-4o-mini (cloud-based)
- Google Gemini Vision (cloud-based)
- Microsoft Florence-2 (lightweight, can run locally outside Ollama)
- RAM/RAM++ (specialized for tagging, open-source)

Each of those would be a new `VisionModel` subclass wired in via
`VisionModelFactory.register_model(name, cls)`, which takes precedence over the default
Ollama adapter.

---

## Architecture

### High-Level Design

```
┌─────────────────┐
│   CLI Interface │
│  (arg parsing)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Config Manager  │
│  (config.yaml)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐       ┌──────────────────┐
│ SmugMug Manager │◄─────►│   Image Cache    │
│  (API wrapper)  │       │   (local temp)   │
└────────┬────────┘       └──────────────────┘
         │
         ▼
┌─────────────────┐
│  Vision Model   │
│   (Ollama/LLM)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Metadata Gen   │
│ (tags/captions) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  SmugMug API    │
│   (push data)   │
└─────────────────┘
```

### Module Structure

```
smugvision/
├── __init__.py
├── __main__.py              # CLI entry point
├── config/                  # ✅ IMPLEMENTED
│   ├── __init__.py
│   ├── manager.py           # Configuration loading and validation
│   └── defaults.py          # Default configuration values
├── face/                    # ✅ IMPLEMENTED
│   ├── __init__.py
│   ├── recognizer.py        # Backend-agnostic coordinator: reference map,
│   │                        #   encoding cache, confidence filtering
│   └── backends/            # Pluggable embedding backends
│       ├── base.py          # FaceBackend ABC (normalized score contract)
│       ├── dlib_backend.py  # Default: face_recognition / dlib, 128-d euclidean
│       └── insightface_backend.py  # Optional: ArcFace ONNX, 512-d cosine
├── smugmug/                 # ✅ IMPLEMENTED
│   ├── __init__.py
│   ├── client.py            # SmugMug API client wrapper
│   ├── models.py            # Data models for Album, AlbumImage
│   └── exceptions.py        # Custom exceptions
├── vision/                  # ✅ IMPLEMENTED
│   ├── __init__.py
│   ├── base.py              # Abstract base class for vision models
│   ├── llama.py             # LlamaVisionModel: the generic Ollama adapter
│   │                        #   (historical name; drives every Ollama vision model)
│   ├── factory.py           # Factory pattern for model selection
│   └── exceptions.py        # Custom exceptions for vision models
├── cache/                  # ✅ IMPLEMENTED
│   ├── __init__.py
│   └── manager.py           # Image cache management
├── processing/
│   ├── __init__.py
│   ├── processor.py         # Main processing orchestration
│   └── metadata.py          # Metadata generation utilities
└── utils/                   # ✅ PARTIALLY IMPLEMENTED
    ├── __init__.py
    ├── exif.py              # EXIF data extraction and geocoding
    ├── locations.py         # Custom location resolution (✅ IMPLEMENTED)
    ├── relationships.py     # Person relationship management
    └── helpers.py           # Helper functions

config.yaml                  # User configuration file
config.yaml.example          # Example configuration (✅ CREATED)
locations.yaml.example       # Example custom locations file (✅ CREATED)
requirements.txt             # Python dependencies
setup.py                     # Package installation
README.md                    # User documentation
DESIGN.md                    # This document
```

---

## Configuration File Structure

**config.yaml** example:

```yaml
# SmugMug API Configuration
smugmug:
  api_key: "YOUR_API_KEY"
  api_secret: "YOUR_API_SECRET"
  user_token: "YOUR_USER_TOKEN"
  user_secret: "YOUR_USER_SECRET"

# Vision Model Configuration
vision:
  model: "qwen3-vl:8b"          # Any vision-capable model from `ollama list`
  endpoint: null                # Ollama endpoint (null = $OLLAMA_HOST, else localhost)
  temperature: 0.7
  max_tokens: 500               # One budget for caption + tags in a single reply
  timeout: 120
  think: false                  # false | "low" | "medium" | "high" | null
  keep_alive: "30m"             # Keep the model resident between images
  single_call: true             # One request per image; false = legacy two-call path
  structured_output: true       # JSON-schema-constrained reply; false = free text
  max_image_dimension: 1568     # Downscale long edge before encoding; 0/null disables
  jpeg_quality: 85
  validate_model: true          # Warn (never fail) if the model is not installed

# Face Recognition Configuration
face_recognition:
  enabled: true
  reference_faces_dir: "~/.smugvision/reference_faces"
  backend: "dlib"               # "dlib" (default) | "insightface" (optional extra)
  tolerance: 0.6                # dlib only - euclidean distance, lower is stricter
  model: "cnn"                  # dlib DETECTOR ('hog'/'cnn'), not the backend selector
  detection_scale: 0.5          # dlib only
  min_confidence: 0.25          # backend-agnostic normalized 0.0-1.0 threshold
  insightface:                  # only read when backend: "insightface"
    model_name: "buffalo_l"
    det_size: [640, 640]
    similarity_threshold: 0.4   # cosine similarity, HIGHER is stricter

# Processing Configuration
processing:
  marker_tag: "smugvision"      # Tag to mark processed images
  generate_captions: true       # Enable caption generation
  generate_tags: true           # Enable tag generation
  preserve_existing: true       # Keep existing captions/tags
  image_size: "Medium"          # Download size from SmugMug (case-insensitive)

# Location Resolution Configuration
location:
  custom_locations_file: "~/.smugvision/locations.yaml"
  check_custom_first: true      # Check custom locations before geocoding
  use_aliases_as_tags: true     # Add location aliases as keyword tags

# Prompt Configuration
prompts:
  caption: |
    Analyze this image and provide a concise, descriptive caption (1-2 sentences) 
    that describes the main subject, setting, and any notable activities or features.
    If EXIF location data is available, incorporate the location naturally.
  
  tags: |
    Generate 5-10 relevant keyword tags for this image. Focus on:
    - Main subjects and objects
    - Activities or actions
    - Setting and location
    - Colors and mood
    - Time of day or season (if apparent)
    Provide tags as a comma-separated list.

# Cache Configuration
cache:
  directory: "~/.smugvision/cache"
  clear_on_exit: false          # Will be implemented later
  preserve_structure: true      # Mirror gallery/folder structure

# Logging Configuration
logging:
  level: "INFO"                 # DEBUG, INFO, WARNING, ERROR
  file: "~/.smugvision/smugvision.log"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

---

## Key Features & Implementation Details

### 1. Marker Tag System

- Each processed image receives a configurable marker tag (default: "smugvision")
- Before processing, check if marker tag exists to avoid duplicate processing
- Command-line flag `--force-reprocess` bypasses this check
- Marker tag is added to SmugMug keywords array

### 2. EXIF-Aware Captions

- Extract EXIF data from images (GPS coordinates, camera model, date/time)
- If GPS coordinates exist, attempt reverse geocoding to location name
- Include location information naturally in prompts sent to vision model
- Example: "A sunset over the Golden Gate Bridge in San Francisco, California"

### 2.1 Custom Location Reference (✅ IMPLEMENTED)

Custom locations allow users to define friendly names for places like their home, relatives' houses, or frequently visited locations. These override reverse geocoding results.

**Use Cases:**
- Get "Eric's House" instead of a street address
- Ensure consistent naming across all photos at the same location
- Faster processing (no API calls needed for custom locations)
- Add searchable aliases as tags

**locations.yaml** example:

```yaml
locations:
  - name: "My House"
    latitude: 38.123456
    longitude: -85.654321
    radius: 50                    # Match radius in meters
    address: "123 Example Street, Yourtown, ST"
    aliases:
      - "Home"
      - "Example Residence"
  
  - name: "Louisville Slugger Field"
    latitude: 38.256510
    longitude: -85.747476
    radius: 200                   # Larger radius for a stadium
    aliases:
      - "Bats Game"
      - "Baseball Stadium"
```

**Resolution Priority:**
1. Check custom locations file first (closest match within radius)
2. If no match, fall back to Overpass API / Nominatim reverse geocoding
3. If geocoding fails, return coordinates as string

**Key Classes:**
- `LocationResolver`: Loads and manages custom locations from YAML
- `CustomLocation`: Data class for a single location definition
- `LocationMatch`: Result of a coordinate match including distance

### 3. Metadata Preservation

- When generating new metadata, preserve existing captions and keywords
- Append new captions after existing ones (with separator if needed)
- Merge new keywords with existing ones (avoiding duplicates)
- Configuration option to control this behavior

### 4. Image Cache Management

```
~/.smugvision/cache/
└── [user_nickname]/
    └── [folder_name]/
        └── [gallery_name]/
            ├── image1.jpg
            ├── image2.jpg
            └── ...
```

- Download images to local cache with structure mirroring SmugMug
- Check cache before downloading (skip if already exists)
- Command-line flag `--clear-cache` to remove cached images
- Future: Auto-cleanup after successful processing (configurable)

### 5. Error Handling

**Strategy:** Fail-fast for initial implementation

- Stop processing on first API error
- Log detailed error information including:
  - Image filename/URL
  - Error type and message
  - Stack trace
  - Gallery/folder context
- Save processing state for potential resume (future enhancement)

**Error Types:**
- SmugMug API errors (auth, rate limit, network)
- Vision model errors (timeout, invalid response)
- File I/O errors (cache write failures)
- Configuration errors (missing/invalid settings)

### 6. Processing Scope

**Initial Implementation:**
- Single gallery processing via `--gallery <gallery_id>`
- Gallery ID obtained from SmugMug URL or API

**Future Enhancement:**
- Folder processing via `--folder <folder_id>`
- Recursive processing of all galleries in folder
- Progress tracking across multiple galleries

---

## Command-Line Interface

### Basic Usage

```bash
# Process a single gallery
smugvision --gallery "abc123"

# Force reprocessing of already-processed images
smugvision --gallery "abc123" --force-reprocess

# Clear cache before processing
smugvision --gallery "abc123" --clear-cache

# Dry run (don't update SmugMug, just show what would happen)
smugvision --gallery "abc123" --dry-run

# Use alternate config file
smugvision --gallery "abc123" --config /path/to/config.yaml

# Verbose logging
smugvision --gallery "abc123" --verbose
```

### CLI Arguments

```
Required:
  --gallery GALLERY_ID       SmugMug gallery ID to process

Optional:
  --folder FOLDER_ID         Process all galleries in folder (future)
  --config PATH              Path to config file (default: ./config.yaml)
  --force-reprocess          Reprocess images even if already tagged
  --clear-cache              Clear image cache before processing
  --dry-run                  Preview actions without updating SmugMug
  --verbose, -v              Enable verbose DEBUG logging
  --help, -h                 Show help message
  --version                  Show version information
```

---

## Coding Practices

### Code Style

- **PEP 8** compliance for all Python code
- **Type hints** for all function signatures
- **Docstrings** for all classes and public methods (Google style)
- **Line length:** 100 characters maximum
- **Formatter:** Black (with line-length=100)
- **Linter:** Pylint with custom configuration

### Example Code Style

```python
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class SmugMugClient:
    """Client for interacting with SmugMug API.
    
    This class handles authentication, API requests, and data retrieval
    from SmugMug galleries and images.
    
    Attributes:
        api_key: SmugMug API key
        api_secret: SmugMug API secret
        access_token: OAuth access token
    """
    
    def __init__(self, api_key: str, api_secret: str) -> None:
        """Initialize SmugMug client with credentials.
        
        Args:
            api_key: SmugMug API key
            api_secret: SmugMug API secret
            
        Raises:
            ValueError: If credentials are empty or invalid
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self._authenticated = False
        logger.info("SmugMugClient initialized")
    
    def get_gallery_images(
        self, 
        gallery_id: str, 
        include_processed: bool = False
    ) -> List[dict]:
        """Retrieve all images from specified gallery.
        
        Args:
            gallery_id: Unique identifier for the gallery
            include_processed: If False, skip images with marker tag
            
        Returns:
            List of image dictionaries containing metadata
            
        Raises:
            SmugMugAPIError: If API request fails
            AuthenticationError: If not authenticated
        """
        logger.info(f"Fetching images from gallery {gallery_id}")
        # Implementation here
        pass
```

### Error Handling Pattern

```python
from smugvision.smugmug.exceptions import SmugMugAPIError

try:
    images = smugmug_client.get_gallery_images(gallery_id)
except SmugMugAPIError as e:
    logger.error(f"Failed to fetch gallery images: {e}", exc_info=True)
    print(f"Error: Unable to access gallery {gallery_id}")
    print(f"Details: {str(e)}")
    sys.exit(1)
```

### Testing Strategy

- **Unit tests** for individual modules
- **Integration tests** for API interactions (with mocking)
- **Fixture data** for consistent test inputs
- **Test coverage target:** 80% minimum
- **Framework:** pytest

### Dependency Management

```
# requirements.txt
ollama>=0.1.0
pyyaml>=6.0
requests>=2.31.0
pillow>=10.0.0
python-dateutil>=2.8.2
pytest>=7.4.0
black>=23.0.0
pylint>=3.0.0
```

### Git Practices

- **Branch naming:** `feature/description`, `bugfix/description`
- **Commit messages:** Conventional Commits format
  - `feat: add support for folder processing`
  - `fix: correct EXIF coordinate parsing`
  - `docs: update configuration examples`
- **Pull requests:** Required for all changes
- **Version tags:** Semantic versioning (v1.0.0, v1.1.0, etc.)

---

## Logging Standards

All log messages must include:
1. **Timestamp** (ISO 8601 format)
2. **Module name** (automatically via `__name__`)
3. **Log level** (DEBUG, INFO, WARNING, ERROR, CRITICAL)
4. **Message** with relevant context

### Logging Example

```python
import logging

logger = logging.getLogger(__name__)

# INFO: General progress
logger.info(f"Processing image: {image_filename}")

# DEBUG: Detailed information
logger.debug(f"Vision model response: {response[:100]}...")

# WARNING: Recoverable issues
logger.warning(f"Image {image_id} already has marker tag, skipping")

# ERROR: Failures requiring attention
logger.error(f"Failed to update image {image_id}: {error}", exc_info=True)
```

### Log Output Format

```
2024-11-23 14:32:15,123 - smugvision.smugmug.client - INFO - Authenticating with SmugMug API
2024-11-23 14:32:16,456 - smugvision.processing.processor - INFO - Processing gallery abc123
2024-11-23 14:32:17,789 - smugvision.vision.llama - DEBUG - Sending prompt to the vision model
2024-11-23 14:32:19,012 - smugvision.smugmug.client - ERROR - API request failed: 401 Unauthorized
```

---

## Future Enhancements

### Phase 2 (Near-term)
- Face detection and recognition using provided reference faces
- Folder-level processing (recursive gallery processing)
- Resume capability for interrupted processing
- Parallel processing for faster throughput
- Progress bar with ETA

### Phase 3 (Medium-term)
- Web UI for easier configuration and monitoring
- Support for additional vision models (GPT-4o, Gemini, etc.)
- Batch prompt customization per gallery
- Advanced filtering (by date range, existing tags, etc.)
- Export metadata to local database for analytics

### Phase 4 (Long-term)
- SmugMug to SmugMug gallery migration with metadata
- Integration with other photo services (Google Photos, iCloud)
- Custom model fine-tuning for specific photo collections
- Automated tagging based on learned patterns

---

## Development Roadmap

### Version 0.1.0 (MVP) - Mostly Complete
- [x] Production CLI interface (`__main__.py`)
- [x] Config file loading and validation
- [x] Configuration manager with interactive setup
- [x] YAML configuration support
- [x] SmugMug API authentication (OAuth 1.0a)
- [x] SmugMug API client wrapper
- [x] Single gallery image retrieval
- [x] Album and image data models
- [x] Image cache management (download, organize, skip existing)
- [x] Ollama integration (any vision-capable model; no allow-list)
- [x] Caption and tag generation
- [x] Metadata update to SmugMug (PATCH endpoint)
- [x] Marker tag system (check and add tags)
- [x] Image download from SmugMug (multiple sizes)
- [x] Video download support with LargestVideo endpoint
- [x] Video filtering (skip by default, optional include)
- [x] Basic error handling and logging
- [x] Vision model factory pattern
- [x] Abstract base class for vision models
- [x] Custom exceptions for vision models
- [x] Custom exceptions for SmugMug API
- [x] Album resolution from URLs, node IDs, and names
- [x] Recursive album search within folder structures
- [x] URL path resolution for folder navigation
- [x] Pagination support for large result sets
- [x] ImageProcessor orchestration class
- [x] MetadataFormatter for combining AI and EXIF metadata
- [x] End-to-end processing pipeline with statistics
- [x] Test utilities (test_smugmug.py, test_processor.py, test_vision.py, debug_face_recognition.py)

### Version 0.2.0 - Complete ✓
- [x] Force reprocessing flag
- [x] Preserve existing metadata
- [x] EXIF data extraction and integration
- [x] EXIF orientation handling
- [x] HEIC/HEIF image format support
- [x] Reverse geocoding for location names
- [x] Improved error messages
- [ ] Unit tests for core modules

### Version 0.3.0 - Complete ✓
- [x] Face detection and recognition system
- [x] Reference faces management
- [x] Person name identification (with proper formatting)
- [x] Relationship context integration
- [x] Dry-run mode
- [x] Progress indicators and statistics
- [ ] Folder processing support (planned for 1.0)
- [ ] Cache cleanup functionality (planned for 1.0)
- [ ] Integration tests

### Version 1.0.0
- [ ] Complete documentation
- [x] Installation via pip
- [ ] Comprehensive test coverage
- [ ] Production-ready error handling
- [ ] Performance optimizations

### Completed Features (Beyond Original Roadmap)
- [x] Advanced EXIF location extraction with venue search
- [x] Overpass API integration for POI discovery
- [x] Configurable geocoding with exclusion filters
- [x] Interactive venue selection
- [x] Face encoding with multiple reference images per person
- [x] Confidence-based face matching
- [x] Person relationship management system
- [x] Context-aware caption generation with person names
- [x] Processing time tracking and metrics
- [x] Multi-format image support (JPEG, PNG, HEIC)
- [x] Image scaling for performance optimization
- [x] Comprehensive logging with module identification
- [x] SmugMug URL parsing and album resolution
- [x] Recursive folder navigation and album discovery
- [x] Cache folder structure mirroring SmugMug hierarchy
- [x] Multiple image size support (Thumb through X3Large, Original)
- [x] Video file detection and separate handling
- [x] Video download via LargestVideo API endpoint
- [x] Configurable video inclusion/exclusion
- [x] SmugMug API pagination for large datasets
- [x] Node-based folder hierarchy navigation
- [x] Test utilities for SmugMug integration (test_smugmug.py)
- [x] Test utilities for full processing pipeline (test_processor.py)
- [x] OAuth token acquisition helper (get_smugmug_tokens.py)
- [x] Album discovery tool (find_album_key.py)
- [x] Production-ready CLI with rich output formatting
- [x] Batch processing statistics and reporting
- [x] Dry-run mode with detailed preview output
- [x] Person name formatting (converting underscores to spaces)
- [x] Pip-installable package with pyproject.toml
- [x] Console scripts: smugvision, smugvision-config, smugvision-get-tokens, smugvision-optimize-faces
- [x] Organized project structure (tests/, scripts/ directories)

---

## Installation & Setup

### Prerequisites

1. **Python 3.10+**
2. **Ollama installed** on macOS
3. **SmugMug API credentials** (API key, secret, OAuth tokens)
4. **A vision-capable model** downloaded via Ollama (`ollama list` to check)

### Installation Steps

```bash
# 1. Install Ollama
brew install ollama

# 2. Start Ollama service
ollama serve

# 3. Download a vision model - any vision-capable Ollama model works.
#    Set its name as vision.model in ~/.smugvision/config.yaml.
ollama list
ollama pull <model>

# 4. Clone repository
git clone https://github.com/yourusername/smugvision.git
cd smugvision

# 5. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 6. Install dependencies
pip install -r requirements.txt

# 7. Copy and configure config file
cp config.yaml.example config.yaml
# Edit config.yaml with your SmugMug credentials

# 8. Run smugVision
python -m smugvision --gallery "abc123"
```

---

## SmugMug API Integration

### Authentication Flow

SmugMug uses OAuth 1.0a for authentication. The process:

1. **Application credentials** (API key + secret) - obtained from SmugMug
2. **Request token** - temporary token for authorization
3. **User authorization** - user approves access via web browser
4. **Access token** - permanent token for API requests

These tokens should be stored in `config.yaml` after initial setup.

### Key API Endpoints

```python
# Get gallery details
GET /api/v2/album/{gallery_id}

# Get gallery images
GET /api/v2/album/{gallery_id}!images

# Get image details
GET /api/v2/image/{image_key}

# Update image metadata
PATCH /api/v2/image/{image_key}
{
    "Caption": "New caption text",
    "Keywords": ["tag1", "tag2", "tag3"]
}

# Download image
GET {image_url}?size={size}  # size: Medium, Large, X2Large, etc.
```

### Rate Limiting

- SmugMug API rate limits should be respected
- Implement exponential backoff for rate limit errors
- Consider batch operations where possible
- Log rate limit warnings

---

## Data Models

### Gallery Model

```python
@dataclass
class Gallery:
    """Represents a SmugMug gallery."""
    gallery_id: str
    name: str
    url: str
    image_count: int
    uri: str
```

### Image Model

```python
@dataclass
class Image:
    """Represents a SmugMug image with metadata."""
    image_key: str
    filename: str
    uri: str
    caption: Optional[str]
    keywords: List[str]
    download_url: str
    date_uploaded: str
    exif: Optional[dict]
    has_marker: bool  # Whether smugvision tag exists
```

### Metadata Result Model

```python
@dataclass
class MetadataResult:
    """Generated metadata for an image."""
    caption: str
    tags: List[str]
    confidence: float  # 0.0 to 1.0
    model_used: str
    processing_time: float  # seconds
```

### Vision Model Interface

The primary entry point is `generate_metadata()` — one prompt, one image encode, one
JSON-schema-constrained chat call returning both halves of the metadata:

```python
def generate_metadata(
    self,
    image_path: str,
    caption_instruction: str,
    tags_instruction: str,
    *,
    temperature: float = 0.4,
    max_tokens: int = 400,
    location_context: Optional[str] = None,
    person_names: Optional[List[str]] = None,   # raw names, underscores intact
    total_faces: Optional[int] = None,          # detected, not recognized
    album_name: Optional[str] = None,
) -> MetadataResult:
    ...
```

Passing `""` for either instruction skips that half of the reply. The keyword defaults
above are rarely what runs: `ImageProcessor` passes `vision.temperature` and
`vision.max_tokens` from config explicitly, so the config values win.

`generate_caption()` and `generate_tags()` retain their original signatures and remain
part of the ABC; they are thin wrappers over `generate_metadata`. `vision.single_call:
false` restores two separate requests, and `vision.structured_output: false` restores
free-text output with heuristic parsing — both paths are live, not stubs.

---

## Current Status & Next Steps

### What's Complete (Ready for Integration)

**Infrastructure Layer:**
- ✅ Configuration system with defaults, validation, and interactive setup
- ✅ SmugMug API client with OAuth 1.0a authentication
- ✅ Album/image retrieval with pagination and filtering
- ✅ Image and video download with multiple size options
- ✅ Local cache management with folder structure preservation
- ✅ URL/path-based album resolution and folder navigation

**AI/ML Layer:**
- ✅ Vision model abstraction (factory pattern with base class)
- ✅ Ollama integration supporting any vision-capable model (no allow-list)
- ✅ Caption and tag generation from a single structured call per image
- ✅ EXIF data extraction with GPS and reverse geocoding
- ✅ Face detection and recognition with configurable confidence
- ✅ Person relationship management for context-aware captions

**Data Layer:**
- ✅ Album and AlbumImage data models
- ✅ Metadata update to SmugMug (PATCH endpoint)
- ✅ Marker tag system for tracking processed images
- ✅ Custom exceptions for error handling

### ✅ Processing Module - COMPLETE

All core processing components are now implemented and tested:

**Completed Components:**
1. ✅ **`processing/processor.py`** - Main `ImageProcessor` orchestrator:
   - Accepts album key/URL and processes all unprocessed images
   - Downloads images to cache (using CacheManager)
   - Extracts EXIF data and identifies faces
   - Generates captions and tags (using VisionModel)
   - Updates SmugMug with new metadata
   - Adds marker tag to processed images
   - Tracks progress and reports detailed statistics

2. ✅ **`processing/metadata.py`** - `MetadataFormatter` utilities:
   - Combines vision-generated captions with EXIF location data
   - Merges person names from face recognition
   - Handles metadata preservation (append vs replace)
   - Formats tags and captions for SmugMug API

3. ✅ **`__main__.py`** - Production CLI entry point:
   - Parses command-line arguments
   - Initializes configuration
   - Creates processor instance
   - Runs processing and displays rich formatted results
   - Supports dry-run, force-reprocess, and video filtering

### Recommended Next Steps

**Phase 1: Testing & Documentation** ✅ **COMPLETE**
1. ✅ Test with real albums (validated)
2. ✅ Document CLI usage in README.md
3. ✅ Create comprehensive documentation
4. ✅ Add usage examples

**Phase 2: Future Enhancements (Version 1.0+)**
1. Folder batch processing (process entire folder trees)
2. Cache cleanup utilities
3. Unit and integration tests
4. Performance optimizations (parallel downloads, batch API calls)

**Phase 2: CLI Interface**
1. Create `__main__.py` with argument parsing
2. Add commands: `process`, `list`, `status`
3. Support for `--gallery`, `--url`, `--node`, `--force-reprocess`
4. Add `--dry-run` mode for preview
5. Implement verbose logging flag

**Phase 3: Testing & Refinement**
1. Test with real SmugMug galleries
2. Handle edge cases (no faces, no EXIF, processing errors)
3. Optimize for performance (parallel downloads, batch updates)
4. Add progress bars and ETA
5. Write unit tests for processor

**Phase 4: Documentation & Packaging**
1. Complete README with usage examples
2. Add troubleshooting guide
3. Create setup.py for pip installation
4. Add example configurations

### Design Considerations for Processor

**Processing Flow:**
```
For each image in album:
  1. Check if already processed (marker tag) → skip if yes
  2. Download to cache (skip if cached)
  3. Extract EXIF data (GPS, camera info, date)
  4. Detect and identify faces (if enabled)
  5. Generate caption AND tags in ONE structured vision call
     (generate_metadata; vision.single_call=false restores the old 5/6 split)
  6. Format metadata (merge person names, location)
  7. Update SmugMug via PATCH API
  8. Add marker tag
  9. Log results and metrics
```

**Error Handling Strategy:**
- Continue processing on single image failure
- Log errors with full context
- Collect statistics (success/skip/error counts)
- Display summary at end
- Option for `--stop-on-error` for strict mode

**Performance Optimizations:**
- Cache downloaded images (already implemented)
- Reuse face encodings across images
- Batch SmugMug updates where possible
- Show progress with ETA

---

## Questions for Future Consideration

1. **Metadata backup**: Should we maintain local backup of original metadata before modification?
2. **Prompt templates**: Should we support per-gallery custom prompts?
3. **Batch size**: What's the optimal number of images to process before syncing to SmugMug?
4. **Model switching**: switching models is already just a `vision.model` config change.
   The open part is running two models *simultaneously* on the same image for comparison.
5. **Undo functionality**: How to implement rollback of metadata changes?

---

---

## Web UI Design (Phase 3 Feature)

### Overview

A local web-based interface for smugVision that provides a visual preview of AI-generated metadata before committing changes to SmugMug. The UI defaults to dry-run mode, showing proposed changes alongside thumbnails, and requires explicit user confirmation to commit.

### Goals

1. Provide visual feedback for processing decisions
2. Show side-by-side comparison of current vs. proposed metadata
3. Default to safe dry-run mode (no changes without explicit commit)
4. Display detected faces and location information
5. Surface reference face and relationship data for transparency

### Technical Stack

- **Backend**: Flask (Python) - Simple, integrates directly with existing smugVision modules
- **Frontend**: Vanilla HTML/CSS/JavaScript with a simple CSS framework (e.g., Pico CSS or similar minimal framework)
- **Communication**: REST API + Server-Sent Events (SSE) for progress updates
- **Deployment**: Localhost only (e.g., `http://localhost:5050`)

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web Browser                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    smugVision Web UI                      │   │
│  │  ┌─────────────┐  ┌────────────────────────────────────┐ │   │
│  │  │ Album Input │  │         Preview Grid               │ │   │
│  │  │ (URL paste) │  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐  │ │   │
│  │  └─────────────┘  │  │thumb│ │thumb│ │thumb│ │thumb│  │ │   │
│  │  ┌─────────────┐  │  │+diff│ │+diff│ │+diff│ │+diff│  │ │   │
│  │  │   Actions   │  │  └─────┘ └─────┘ └─────┘ └─────┘  │ │   │
│  │  │ [Preview]   │  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐  │ │   │
│  │  │ [Commit]    │  │  │thumb│ │thumb│ │thumb│ │thumb│  │ │   │
│  │  └─────────────┘  │  │+diff│ │+diff│ │+diff│ │+diff│  │ │   │
│  │  ┌─────────────┐  │  └─────┘ └─────┘ └─────┘ └─────┘  │ │   │
│  │  │  Progress   │  │         (infinite scroll)          │ │   │
│  │  │  [███░░] 60%│  └────────────────────────────────────┘ │   │
│  │  └─────────────┘                                          │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP/SSE
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Flask Backend (localhost:5050)               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                      REST API Routes                      │   │
│  │  POST /api/preview     - Start preview processing         │   │
│  │  GET  /api/preview/status - SSE stream for progress       │   │
│  │  GET  /api/preview/results - Get preview results          │   │
│  │  POST /api/commit      - Commit changes to SmugMug        │   │
│  │  GET  /api/faces       - List known reference faces       │   │
│  │  GET  /api/relationships - Get relationship graph data    │   │
│  │  GET  /api/thumbnail/<key> - Proxy thumbnail from SmugMug │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Existing smugVision Modules                  │   │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────────────┐    │   │
│  │  │ SmugMug    │ │ Image      │ │ Face               │    │   │
│  │  │ Client     │ │ Processor  │ │ Recognizer         │    │   │
│  │  └────────────┘ └────────────┘ └────────────────────┘    │   │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────────────┐    │   │
│  │  │ Vision     │ │ Cache      │ │ Relationship       │    │   │
│  │  │ Model      │ │ Manager    │ │ Manager            │    │   │
│  │  └────────────┘ └────────────┘ └────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Module Structure

```
smugvision/
├── web/                        # NEW: Web UI module
│   ├── __init__.py
│   ├── app.py                  # Flask application factory
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── api.py              # REST API endpoints
│   │   └── pages.py            # HTML page routes
│   ├── services/
│   │   ├── __init__.py
│   │   └── preview.py          # Preview processing service
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css       # Custom styles
│   │   └── js/
│   │       └── app.js          # Frontend JavaScript
│   └── templates/
│       ├── base.html           # Base template
│       ├── index.html          # Main page
│       ├── preview.html        # Preview results
│       ├── faces.html          # Known faces display
│       └── relationships.html  # Relationship graph
```

### API Endpoints

#### POST /api/preview
Start a preview (dry-run) processing job for an album.

**Request:**
```json
{
  "url": "https://site.smugmug.com/.../n-XXXXX/album-name",
  "force_reprocess": false
}
```

**Response:**
```json
{
  "job_id": "abc123",
  "album_key": "XXXXX",
  "album_name": "Album Name",
  "total_images": 42,
  "status": "processing"
}
```

#### GET /api/preview/status?job_id=abc123
Server-Sent Events stream for progress updates.

**SSE Events:**
```
event: progress
data: {"current": 5, "total": 42, "filename": "IMG_1234.jpg", "percent": 12}

event: image_complete
data: {"image_key": "xxx", "filename": "IMG_1234.jpg", "success": true}

event: complete
data: {"processed": 40, "skipped": 2, "errors": 0}

event: error
data: {"message": "Failed to process IMG_5678.jpg: timeout"}
```

#### GET /api/preview/results?job_id=abc123
Get the full preview results after processing completes.

**Response:**
```json
{
  "job_id": "abc123",
  "album_key": "XXXXX",
  "album_name": "Album Name",
  "status": "complete",
  "stats": {
    "total": 42,
    "processed": 40,
    "skipped": 2,
    "errors": 0
  },
  "images": [
    {
      "image_key": "img123",
      "filename": "IMG_1234.jpg",
      "thumbnail_url": "/api/thumbnail/img123",
      "web_uri": "https://site.smugmug.com/...",
      "status": "processed",
      "current": {
        "caption": "Existing caption or null",
        "keywords": ["tag1", "tag2"]
      },
      "proposed": {
        "caption": "AI-generated caption with location and people",
        "keywords": ["tag1", "tag2", "newtag1", "newtag2", "smugvision"]
      },
      "details": {
        "faces_detected": ["John Doe", "Jane Smith"],
        "location": "Golden Gate Bridge, San Francisco, CA",
        "exif_date": "2024-06-15T14:30:00"
      }
    },
    {
      "image_key": "img456",
      "filename": "IMG_5678.jpg",
      "thumbnail_url": "/api/thumbnail/img456",
      "status": "skipped",
      "reason": "Already has smugvision marker tag"
    },
    {
      "image_key": "img789",
      "filename": "IMG_9012.jpg",
      "thumbnail_url": "/api/thumbnail/img789",
      "status": "error",
      "error": "Vision model timeout"
    }
  ]
}
```

#### POST /api/commit
Commit the previewed changes to SmugMug.

**Request:**
```json
{
  "job_id": "abc123"
}
```

**Response:**
```json
{
  "status": "success",
  "committed": 40,
  "errors": 0
}
```

#### GET /api/faces
Get list of known reference faces.

**Response:**
```json
{
  "faces": [
    {
      "name": "John Doe",
      "display_name": "John Doe",
      "reference_count": 3,
      "sample_image": "/api/face-sample/John_Doe"
    },
    {
      "name": "Jane Smith",
      "display_name": "Jane Smith", 
      "reference_count": 2,
      "sample_image": "/api/face-sample/Jane_Smith"
    }
  ],
  "total": 2
}
```

#### GET /api/relationships
Get relationship graph data for visualization.

**Response:**
```json
{
  "nodes": [
    {"id": "John_Doe", "label": "John Doe"},
    {"id": "Jane_Smith", "label": "Jane Smith"},
    {"id": "Junior_Doe", "label": "Junior Doe"}
  ],
  "edges": [
    {"from": "John_Doe", "to": "Jane_Smith", "label": "spouse"},
    {"from": "John_Doe", "to": "Junior_Doe", "label": "parent"},
    {"from": "Jane_Smith", "to": "Junior_Doe", "label": "parent"}
  ],
  "groups": [
    {
      "members": ["John_Doe", "Jane_Smith", "Junior_Doe"],
      "description": "The Doe Family"
    }
  ]
}
```

#### GET /api/thumbnail/<image_key>
Proxy thumbnail image from SmugMug (avoids CORS issues).

**Response:** Image binary (JPEG)

#### GET /api/face-sample/<person_name>
Get a sample reference face image for display.

**Response:** Image binary (JPEG)

### UI Pages

#### Main Page (index.html)
- URL input field for SmugMug album URL
- "Preview" button to start dry-run processing
- Navigation to Faces and Relationships pages
- Status indicator for Ollama/vision model availability

#### Preview Results Page (preview.html)
- Album info header (name, image count)
- Progress bar (during processing)
- Infinite-scroll grid of image cards:
  - Thumbnail image
  - Status indicator (processed/skipped/error)
  - Current vs. Proposed metadata diff view
  - Detected faces chips
  - Location badge
- "Commit All Changes" button (disabled during processing, enabled after)
- Summary statistics

#### Image Card Component
```
┌─────────────────────────────────────────────────────────┐
│  ┌─────────────┐  IMG_1234.jpg                    [✓]  │
│  │             │  ─────────────────────────────────────│
│  │  thumbnail  │  Caption:                             │
│  │             │  - "Family at the beach"              │
│  │             │  + "John and Jane enjoying sunset at  │
│  └─────────────┘    Golden Gate Bridge, San Francisco" │
│                                                         │
│  Keywords:                                              │
│  [beach] [vacation] + [Golden Gate] + [sunset]         │
│  + [John Doe] + [Jane Smith] + [smugvision]            │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 👤 John Doe, Jane Smith  📍 San Francisco, CA   │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘

Skipped Card (grayed out):
┌─────────────────────────────────────────────────────────┐
│  ┌─────────────┐  IMG_5678.jpg              [SKIPPED]  │
│  │             │  ─────────────────────────────────────│
│  │  thumbnail  │  Already processed (has smugvision    │
│  │  (grayed)   │  marker tag)                          │
│  │             │                                        │
│  └─────────────┘  Current: "Existing caption..."       │
└─────────────────────────────────────────────────────────┘
```

#### Known Faces Page (faces.html)
- Grid of known people with sample face images
- Count of reference images per person
- Simple display (no add/remove functionality for now)

```
┌──────────────────────────────────────────────────────────────┐
│  Known Faces (5 people)                                      │
│  ────────────────────────────────────────────────────────────│
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   [face]    │  │   [face]    │  │   [face]    │          │
│  │  John Doe   │  │ Jane Smith  │  │ Junior Doe  │          │
│  │  3 refs     │  │  2 refs     │  │  1 ref      │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└──────────────────────────────────────────────────────────────┘
```

#### Relationships Page (relationships.html)
- Visual graph of relationships (using a simple JS graph library like vis.js or cytoscape.js)
- List view of defined groups
- Shows relationship types (spouse, parent, sibling, etc.)

```
┌──────────────────────────────────────────────────────────────┐
│  Relationship Graph                                          │
│  ────────────────────────────────────────────────────────────│
│                                                              │
│            ┌──────────┐                                      │
│            │ John Doe │                                      │
│            └────┬─────┘                                      │
│         spouse  │                                            │
│            ┌────┴─────┐                                      │
│            │Jane Smith│                                      │
│            └────┬─────┘                                      │
│          parent │                                            │
│            ┌────┴─────┐                                      │
│            │Junior Doe│                                      │
│            └──────────┘                                      │
│                                                              │
│  Groups:                                                     │
│  • The Doe Family: John Doe, Jane Smith, Junior Doe          │
└──────────────────────────────────────────────────────────────┘
```

### Data Flow

#### Preview Workflow

```
1. User enters SmugMug URL → clicks "Preview"
   │
2. POST /api/preview
   │
3. Backend:
   ├── Parse URL → extract album key
   ├── Fetch album info from SmugMug
   ├── Return job_id immediately
   │
4. Frontend connects to GET /api/preview/status?job_id=xxx (SSE)
   │
5. Backend processes each image (dry_run=True):
   │   For each image:
   │   ├── Download thumbnail/medium image
   │   ├── Extract EXIF location
   │   ├── Detect/identify faces
   │   ├── Generate caption & tags via vision model
   │   ├── Format proposed metadata
   │   ├── Store result in memory (job results dict)
   │   └── Send SSE progress event
   │
6. Frontend receives SSE events → updates progress bar
   │
7. Processing complete → SSE "complete" event
   │
8. Frontend calls GET /api/preview/results
   │
9. Frontend renders image grid with diff views
```

#### Commit Workflow

```
1. User reviews preview → clicks "Commit All Changes"
   │
2. POST /api/commit {job_id: "xxx"}
   │
3. Backend:
   │   For each processed image in job results:
   │   ├── Call SmugMug PATCH API with proposed metadata
   │   └── Track success/failure
   │
4. Return commit results
   │
5. Frontend shows success message with statistics
```

### State Management

The backend maintains in-memory state for active preview jobs:

```python
# In-memory job storage (simple dict for localhost use)
preview_jobs: Dict[str, PreviewJob] = {}

@dataclass
class PreviewJob:
    job_id: str
    album_key: str
    album_name: str
    status: str  # "processing", "complete", "error"
    total_images: int
    current_image: int
    results: List[ImagePreviewResult]
    created_at: datetime
    
@dataclass
class ImagePreviewResult:
    image_key: str
    filename: str
    thumbnail_url: str
    web_uri: str
    status: str  # "processed", "skipped", "error"
    current_caption: Optional[str]
    current_keywords: List[str]
    proposed_caption: Optional[str]
    proposed_keywords: List[str]
    faces_detected: List[str]
    location: Optional[str]
    error: Optional[str]
```

### CLI Integration

Add a new command to start the web server:

```bash
# Start the web UI server
smugvision-web

# Or with options
smugvision-web --port 5050 --debug
```

This will be a new console script entry point in pyproject.toml.

### Implementation Plan

#### Phase 1: Core Backend & Basic UI
1. Create Flask app structure with routes
2. Implement `/api/preview` endpoint (leverages existing ImageProcessor with dry_run=True)
3. Implement SSE progress streaming
4. Implement `/api/preview/results` endpoint
5. Create basic HTML templates with URL input and progress display
6. Implement thumbnail proxying

#### Phase 2: Preview Display
1. Build image card component with diff view
2. Implement infinite scroll for results
3. Style processed/skipped/error states
4. Add faces and location display to cards

#### Phase 3: Commit Flow
1. Implement `/api/commit` endpoint
2. Add commit button with confirmation
3. Show commit results/statistics

#### Phase 4: Faces & Relationships Pages
1. Implement `/api/faces` endpoint
2. Build faces gallery page
3. Implement `/api/relationships` endpoint  
4. Build relationship graph visualization (using vis.js or similar)

### Dependencies (New)

```
# Add to requirements.txt
flask>=3.0.0
```

No heavy frontend framework needed - vanilla JS with fetch API and EventSource for SSE.

### Security Considerations

- **Localhost only**: Server binds to 127.0.0.1, not 0.0.0.0
- **No authentication**: Assumes trusted local environment
- **No persistent storage**: Job data is in-memory only, cleared on restart
- **SmugMug credentials**: Read from existing config.yaml, never exposed via API

### Future Enhancements (Not in Initial Scope)

- [ ] Multiple gallery processing queue
- [ ] Gallery browser (tree view of SmugMug folders)
- [ ] Selective commit (checkbox per image)
- [ ] Edit proposed metadata before commit
- [ ] Reference face management (add/remove)
- [ ] Cache management UI
- [ ] Processing history/logs view
- [ ] Dark mode

---

## Contributing

This is currently a personal project. Contribution guidelines will be added once the MVP is complete.

---

## License

To be determined (likely MIT or Apache 2.0)

---

## Contact & Support

**Developer:** Eric  
**Repository:** https://github.com/yourusername/smugvision (update when created)  
**Issues:** Use GitHub Issues for bug reports and feature requests

---

**Document History:**
- v1.0 (2024-11-23): Initial design document created