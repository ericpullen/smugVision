# smugVision

**AI-Powered Photo Metadata Generation for SmugMug**

Automatically generate descriptive captions and relevant tags for your SmugMug photos using local AI vision models. smugVision combines computer vision, face recognition, and EXIF metadata to create rich, context-aware descriptions for your photo albums.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## Features

✨ **AI-Powered Metadata Generation**
- Generate descriptive captions using local Llama 3.2 Vision model
- Create relevant keyword tags automatically
- Context-aware prompts with location and person information

👤 **Face Recognition**
- Identify people in photos automatically
- Organize reference faces in a simple folder structure
- Configurable confidence thresholds

📍 **Location Intelligence**
- Extract GPS coordinates from EXIF data
- Reverse geocoding for human-readable locations
- Automatic location context in captions and tags

🖼️ **Smart Image Processing**
- Support for HEIC/HEIF formats
- Automatic orientation correction
- Skip already-processed images
- Video file detection and exclusion (optional)

🔄 **SmugMug Integration**
- OAuth 1.0a authentication
- Batch album processing
- Preserve existing metadata (optional)
- Dry-run mode for safe previewing

🚀 **Performance & Reliability**
- Local caching to avoid re-downloading
- Configurable image sizes
- Progress tracking and detailed logging
- Comprehensive error handling

---

## Quick Start

### Prerequisites

- Python 3.9 or higher
- [Ollama](https://ollama.ai/) with `llama3.2-vision` model
- SmugMug account with API credentials

### Installation

#### Option 1: Install via pip (Recommended)

```bash
# Install from source
pip install git+https://github.com/yourusername/smugvision.git

# Or install locally for development
git clone https://github.com/yourusername/smugvision.git
cd smugvision
pip install -e .
```

#### Option 2: Install from requirements.txt

```bash
git clone https://github.com/yourusername/smugvision.git
cd smugvision
pip install -r requirements.txt
```

#### Install the Vision Model

```bash
ollama pull llama3.2-vision
```

### Initial Configuration

Run the interactive configuration setup:
```bash
# If installed via pip:
smugvision-config

# Or using Python module:
python -m smugvision.config.manager --setup
```

This will create `~/.smugvision/config.yaml` and prompt you for:
- SmugMug API key and secret
- SmugMug user token and secret
- Default processing options

### Getting SmugMug Credentials

1. **Get API Key & Secret:**
   - Visit https://api.smugmug.com/api/developer/apply
   - Create a new application
   - Note your API Key and Secret

2. **Get User Token & Secret:**
   - Run the OAuth helper:
     ```bash
     # If installed via pip:
     smugvision-get-tokens

     # Or using the script:
     python scripts/get_smugmug_tokens.py
     ```
   - Follow the OAuth flow in your browser
   - Copy the user token and secret to your config

---

## Usage

### Basic Processing

Process an album by SmugMug album key:
```bash
# If installed via pip:
smugvision --gallery abc123

# Or using Python module:
python -m smugvision --gallery abc123
```

Process an album by URL:
```bash
smugvision --url "https://site.smugmug.com/path/to/n-XXXXX/album-name"
```

### Dry Run (Preview Without Updating)

Preview what changes would be made without updating SmugMug:
```bash
smugvision --gallery abc123 --dry-run
```

### Force Reprocessing

Reprocess images even if they already have the `smugvision` marker tag:
```bash
smugvision --gallery abc123 --force-reprocess
```

### Include Videos

By default, video files are skipped. To include them:
```bash
smugvision --gallery abc123 --include-videos
```

### Verbose Logging

Enable detailed debug logging:
```bash
smugvision --gallery abc123 --verbose
```

### Custom Config File

Use a custom configuration file:
```bash
smugvision --gallery abc123 --config /path/to/config.yaml
```

---

## Configuration

smugVision stores its configuration in `~/.smugvision/config.yaml`. Here's an overview of the key settings:

### SmugMug Settings
```yaml
smugmug:
  api_key: "your_api_key"
  api_secret: "your_api_secret"
  user_token: "your_user_token"
  user_secret: "your_user_secret"
```

### Vision Model Settings
```yaml
vision:
  model: "llama3.2-vision"
  endpoint: "http://localhost:11434"
  temperature: 0.7
  max_tokens: 150
```

### Face Recognition
```yaml
face_recognition:
  enabled: true
  reference_faces_dir: "~/.smugvision/reference_faces"
  tolerance: 0.6
  model: "hog"
  detection_scale: 0.5
  min_confidence: 0.25
```

### Processing Options
```yaml
processing:
  generate_captions: true
  generate_tags: true
  preserve_existing: true
  marker_tag: "smugvision"
  image_size: "Medium"
  skip_videos: true
  use_exif_location: true
```

### Caching
```yaml
cache:
  directory: "~/.smugvision/cache"
  preserve_structure: true
```

For a complete configuration example, see [`config.yaml.example`](config.yaml.example).

---

## Face Recognition Setup

1. **Create reference faces directory:**
   ```bash
   mkdir -p ~/.smugvision/reference_faces
   ```

2. **Organize reference faces:**
   ```
   ~/.smugvision/reference_faces/
   ├── John_Doe/
   │   ├── photo1.jpg
   │   ├── photo2.jpg
   │   └── photo3.jpg
   └── Jane_Smith/
       ├── photo1.jpg
       └── photo2.jpg
   ```

3. **Optimize reference faces (optional but recommended):**
   ```bash
   # If installed via pip:
   smugvision-optimize-faces

   # Or using the script:
   python scripts/optimize_reference_faces.py ~/.smugvision/reference_faces
   ```

   This resizes images for faster loading and processing.

### Tips for Reference Faces:
- Use 3-5 clear, well-lit photos per person
- Include photos from different angles
- Avoid sunglasses or heavy shadows
- Larger faces work better (crop to face if needed)
- Name folders like `First_Last` (underscores will be converted to spaces)

---

## How It Works

1. **Album Retrieval**: Connects to SmugMug API and fetches album metadata and image list
2. **Image Download**: Downloads images to local cache (configurable size: Thumb, Small, Medium, Large, XLarge)
3. **EXIF Extraction**: Reads EXIF data for GPS coordinates, date/time, orientation
4. **Location Lookup**: Reverse geocodes coordinates to human-readable location names
5. **Face Recognition**: Detects and identifies known people in photos
6. **Context Building**: Combines location, people, and EXIF data into context
7. **AI Generation**: Sends images with context to Llama 3.2 Vision for captions and tags
8. **Metadata Formatting**: Combines AI-generated metadata with extracted context
9. **SmugMug Update**: Patches image metadata via SmugMug API
10. **Progress Tracking**: Reports statistics and any errors

---

## Advanced Features

### Location Services

smugVision can extract GPS coordinates from EXIF data and convert them to readable locations:

- **Geocoding Provider**: Uses Nominatim (OpenStreetMap) by default
- **Custom User Agent**: Configure in `~/.smugvision/geocoding_config.yaml`
- **Rate Limiting**: Respects Nominatim's usage policy (1 request/second)
- **Caching**: Location lookups are cached to minimize API calls

### Relationship Context

Add context about relationships between people in photos:

Create `~/.smugvision/relationships.yaml`:
```yaml
relationships:
  John_Doe:
    Jane_Smith: "wife"
    Billy_Doe: "son"
  Jane_Smith:
    John_Doe: "husband"
```

This helps the AI generate more contextual captions like "John with his wife Jane at the beach."

### Custom Prompts

Customize the AI prompts in your `config.yaml`:

```yaml
prompts:
  caption: |
    Analyze this image and provide a detailed, engaging caption 
    that describes the scene, subjects, and atmosphere.
  
  tags: |
    Generate descriptive keyword tags for this image.
    Focus on subjects, activities, location, mood, and composition.
```

---

## Testing & Development

### Test SmugMug Connection
```bash
python tests/test_smugmug.py --gallery abc123
```

### Test Vision Model
```bash
python tests/test_vision.py path/to/image.jpg
```

### Test Face Recognition
```bash
python tests/debug_face_recognition.py path/to/image.jpg
```

### Test Full Processor
```bash
python tests/test_processor.py --gallery abc123 --dry-run
```

### Install Development Dependencies
```bash
pip install -e ".[dev]"
```

This installs additional tools for testing and development:
- `pytest` for running tests
- `black` for code formatting
- `flake8` for linting
- `mypy` for type checking

---

## Troubleshooting

### "Ollama not responding"
- Ensure Ollama is running: `ollama serve`
- Verify model is installed: `ollama list`
- Check endpoint in config: `vision.endpoint`

### "SmugMug authentication failed"
- Verify API credentials in `~/.smugvision/config.yaml`
- Regenerate user tokens with `smugvision-get-tokens`
- Check that your SmugMug account has API access enabled

### "No faces detected"
- Ensure reference faces directory exists and contains images
- Try lowering `face_recognition.tolerance` (more permissive)
- Verify reference faces are clear and well-lit
- Run `smugvision-optimize-faces` to improve performance

### "Images not downloading"
- Check SmugMug album permissions (must be accessible via API)
- Verify album key or URL is correct
- Try a different `image_size` in config
- Check network connectivity

### "Out of memory"
- Reduce `face_recognition.detection_scale` (e.g., 0.25)
- Use smaller `image_size` for processing
- Process albums in smaller batches

---

## Performance Tips

1. **Use Medium-sized images**: Balances quality and speed
2. **Optimize reference faces**: Run `smugvision-optimize-faces` once
3. **Enable caching**: Avoid re-downloading images (default: enabled)
4. **Skip videos**: Video processing is slower (default: skipped)
5. **Adjust detection scale**: Lower values = faster face detection
6. **Use marker tags**: Automatically skip already-processed images

---

## Architecture

smugVision is organized into modular components:

- **`smugmug/`**: SmugMug API client and data models
- **`vision/`**: Vision model abstraction and Llama integration
- **`face/`**: Face detection and recognition system
- **`processing/`**: Image processing orchestration and metadata formatting
- **`cache/`**: Local image caching and management
- **`utils/`**: EXIF extraction, geocoding, and utilities
- **`config/`**: Configuration management and validation

For detailed architecture documentation, see [`DESIGN.md`](DESIGN.md).

---

## Limitations

- **Local Processing Only**: Requires local Ollama installation
- **Single Album at a Time**: No batch folder processing yet (planned)
- **SmugMug API Rate Limits**: Respects SmugMug's rate limiting
- **Face Recognition Accuracy**: Depends on quality of reference faces
- **Geocoding Rate Limits**: Nominatim allows 1 request/second

---

## Roadmap

See [`DESIGN.md`](DESIGN.md) for detailed roadmap. Planned features include:

- [ ] Batch folder processing
- [ ] Web UI for monitoring and control
- [ ] Multiple vision model support (GPT-4V, Claude Vision)
- [ ] Smart duplicate detection
- [ ] Custom metadata templates
- [ ] Integration with other photo services
- [ ] Docker deployment option

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [`LICENSE`](LICENSE) file for details.

---

## Acknowledgments

- **[Ollama](https://ollama.ai/)**: Local LLM runtime
- **[Meta's Llama](https://ai.meta.com/llama/)**: Vision model
- **[face_recognition](https://github.com/ageitgey/face_recognition)**: Face detection library
- **[SmugMug API](https://api.smugmug.com/)**: Photo hosting platform
- **[Nominatim](https://nominatim.org/)**: Geocoding service

---

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/smugvision/issues)
- **Documentation**: [`DESIGN.md`](DESIGN.md) for architecture details
- **Face Recognition Guide**: [`README_FACE_RECOGNITION.md`](README_FACE_RECOGNITION.md)


---

**Built with ❤️ for photographers who love automation**
