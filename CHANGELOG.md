# Changelog

All notable changes to smugVision will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- Llama 3.2 Vision integration via Ollama
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
- Web UI for monitoring
- Multiple vision model support
- Docker deployment option

---

**Legend:**
- 🎉 Major milestone
- ✨ New feature
- 🐛 Bug fix
- 📝 Documentation
- ♻️ Refactoring
- ⚡ Performance improvement

