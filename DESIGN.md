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
- **AI Model:** Llama 3.2 Vision 11B (via Ollama) - Default, but modular for future model support
- **SmugMug Integration:** Python library or custom API wrapper
- **Configuration:** YAML or JSON config file
- **Logging:** Python standard logging library with timestamps and module identification

### Model Selection Rationale

**Llama 3.2 Vision 11B** was chosen as the default model because:
- Supports images up to 1120x1120 pixels resolution
- Runs efficiently on Apple Silicon (M4 Pro) with at least 8GB VRAM
- Provides strong image understanding and captioning capabilities
- Free and runs locally without API costs
- Works well via Ollama for easy management

**Image Size Recommendation:** Download medium-sized images (approximately 1024-1200px on longest edge) from SmugMug. This balances:
- Quality sufficient for accurate AI analysis
- Bandwidth efficiency
- Processing speed
- Staying within the model's optimal resolution range (1120x1120px)

### Alternative Models (Future Consideration)

While Llama 3.2 Vision is the initial choice, the architecture should support:
- OpenAI GPT-4o/GPT-4o-mini (cloud-based, best quality)
- Google Gemini Vision (cloud-based)
- Microsoft Florence-2 (lightweight, can run locally)
- RAM/RAM++ (specialized for tagging, open-source)

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
│   └── recognizer.py        # Face detection and recognition
├── smugmug/
│   ├── __init__.py
│   ├── client.py            # SmugMug API client wrapper
│   ├── models.py            # Data models for Gallery, Image, etc.
│   └── exceptions.py        # Custom exceptions
├── vision/                  # ✅ IMPLEMENTED
│   ├── __init__.py
│   ├── base.py              # Abstract base class for vision models
│   ├── llama.py             # Llama 3.2 Vision implementation
│   ├── factory.py           # Factory pattern for model selection
│   └── exceptions.py        # Custom exceptions for vision models
├── cache/
│   ├── __init__.py
│   └── manager.py           # Image cache management
├── processing/
│   ├── __init__.py
│   ├── processor.py         # Main processing orchestration
│   └── metadata.py          # Metadata generation utilities
└── utils/                   # ✅ PARTIALLY IMPLEMENTED
    ├── __init__.py
    ├── exif.py              # EXIF data extraction and geocoding
    ├── relationships.py     # Person relationship management
    └── helpers.py           # Helper functions

config.yaml                  # User configuration file
config.yaml.example          # Example configuration (✅ CREATED)
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
  model: "llama3.2-vision"     # Options: llama3.2-vision, gpt-4o, etc.
  endpoint: "http://localhost:11434"  # Ollama endpoint
  temperature: 0.7
  max_tokens: 150

# Processing Configuration
processing:
  marker_tag: "smugvision"      # Tag to mark processed images
  generate_captions: true       # Enable caption generation
  generate_tags: true           # Enable tag generation
  preserve_existing: true       # Keep existing captions/tags
  image_size: "medium"          # Download size from SmugMug

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
2024-11-23 14:32:17,789 - smugvision.vision.llama - DEBUG - Sending prompt to Llama 3.2 Vision
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

### Version 0.1.0 (MVP) - In Progress
- [ ] Basic CLI interface
- [x] Config file loading and validation
- [x] Configuration manager with interactive setup
- [x] YAML configuration support
- [ ] SmugMug API authentication
- [ ] Single gallery image retrieval
- [ ] Image cache management
- [x] Llama 3.2 Vision integration via Ollama
- [x] Caption and tag generation
- [ ] Metadata update to SmugMug
- [ ] Marker tag system
- [x] Basic error handling and logging
- [x] Vision model factory pattern
- [x] Abstract base class for vision models
- [x] Custom exceptions for vision models

### Version 0.2.0
- [ ] Force reprocessing flag
- [ ] Preserve existing metadata
- [x] EXIF data extraction and integration
- [x] EXIF orientation handling
- [x] HEIC/HEIF image format support
- [x] Reverse geocoding for location names
- [x] Improved error messages
- [ ] Unit tests for core modules

### Version 0.3.0 - Partially Complete
- [x] Face detection and recognition system
- [x] Reference faces management
- [x] Person name identification
- [x] Relationship context integration
- [ ] Folder processing support
- [ ] Dry-run mode
- [ ] Cache cleanup functionality
- [ ] Progress indicators
- [ ] Integration tests

### Version 1.0.0
- [ ] Complete documentation
- [ ] Installation via pip
- [ ] Comprehensive test coverage
- [ ] Production-ready error handling
- [ ] Performance optimizations

### Completed Features (Not in Original Roadmap)
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

---

## Installation & Setup

### Prerequisites

1. **Python 3.10+**
2. **Ollama installed** on macOS
3. **SmugMug API credentials** (API key, secret, OAuth tokens)
4. **Llama 3.2 Vision model** downloaded via Ollama

### Installation Steps

```bash
# 1. Install Ollama
brew install ollama

# 2. Start Ollama service
ollama serve

# 3. Download Llama 3.2 Vision model
ollama pull llama3.2-vision

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

---

## Questions for Future Consideration

1. **Metadata backup**: Should we maintain local backup of original metadata before modification?
2. **Prompt templates**: Should we support per-gallery custom prompts?
3. **Batch size**: What's the optimal number of images to process before syncing to SmugMug?
4. **Model switching**: Should we support multiple models simultaneously for comparison?
5. **Undo functionality**: How to implement rollback of metadata changes?

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