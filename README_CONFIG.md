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

The configuration will be automatically saved to `config.yaml` in the current directory.

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

Settings for the AI vision model (Llama 3.2 Vision via Ollama):

```yaml
vision:
  model: "llama3.2-vision"           # Model name
  endpoint: "http://localhost:11434" # Ollama API endpoint
  temperature: 0.7                   # Creativity (0.0 = deterministic, 1.0 = creative)
  max_tokens: 150                    # Maximum length of generated text
  timeout: 120                       # API timeout in seconds
```

**Tips:**
- Lower `temperature` (0.3-0.5) for more consistent, factual captions
- Higher `temperature` (0.7-0.9) for more creative, varied descriptions
- Increase `max_tokens` if captions are being cut off
- Increase `timeout` if you're getting timeout errors on slower systems

### Face Recognition Configuration

Settings for face detection and recognition:

```yaml
face_recognition:
  enabled: true                      # Enable/disable face recognition
  reference_faces_dir: "~/.smugvision/reference_faces"
  tolerance: 0.6                     # Matching strictness (0.0-1.0)
  model: "cnn"                       # Detection model: 'hog' or 'cnn'
  detection_scale: 0.5               # Image scale for detection (0.1-1.0)
  min_confidence: 0.25               # Minimum confidence threshold (0.0-1.0)
```

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

**Parameter Tuning:**
- **tolerance**: Lower values (0.4-0.5) = stricter matching, fewer false positives
- **tolerance**: Higher values (0.6-0.7) = more lenient, may catch more faces but with more false positives
- **model**: `cnn` is more accurate but slower, `hog` is faster but less accurate
- **detection_scale**: Lower values = faster processing but may miss distant faces
- **min_confidence**: Higher values = only include high-confidence matches in results

### Processing Configuration

Settings for image processing behavior:

```yaml
processing:
  marker_tag: "smugvision"           # Tag to mark processed images
  generate_captions: true            # Generate image captions
  generate_tags: true                # Generate keyword tags
  preserve_existing: true            # Keep existing metadata
  image_size: "medium"               # Download size from SmugMug
  use_exif_location: true            # Extract GPS location from EXIF
```

### Prompt Configuration

Customize how the AI describes your images:

```yaml
prompts:
  caption: |
    Analyze this image and provide a concise, descriptive caption 
    (1-2 sentences) that describes the main subject, setting, and 
    any notable activities or features.
  
  tags: |
    Generate 5-10 simple, single-word or short-phrase keyword tags.
    Focus on main subjects, activities, setting, colors, and mood.
```

**Tips for custom prompts:**
- Be specific about the format you want (e.g., "1-2 sentences", "comma-separated list")
- Mention what to focus on (subjects, activities, mood, etc.)
- Include examples of desired output format
- Keep prompts concise and clear

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

