# smugVision Quick Start Guide

Get up and running with smugVision in 5 minutes!

## Prerequisites

- Python 3.9+
- SmugMug account with API access
- [Ollama](https://ollama.ai/) installed

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/smugvision.git
cd smugvision

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install Ollama and pull the model
ollama pull llama3.2-vision
```

## Configuration

### Step 1: Get SmugMug Credentials

1. **API Key & Secret:**
   - Go to https://api.smugmug.com/api/developer/apply
   - Create a new application
   - Note your API Key and API Secret

2. **User Token & Secret:**
   ```bash
   python get_smugmug_tokens.py
   ```
   - Follow the OAuth flow in your browser
   - Copy the user token and user secret

### Step 2: Run Setup

```bash
python -m smugvision.config.manager --setup
```

This creates `~/.smugvision/config.yaml` with your settings.

### Step 3: (Optional) Setup Face Recognition

```bash
# Create reference faces directory
mkdir -p ~/.smugvision/reference_faces

# Organize faces by person
# ~/.smugvision/reference_faces/
#   ├── John_Doe/
#   │   ├── photo1.jpg
#   │   └── photo2.jpg
#   └── Jane_Smith/
#       └── photo1.jpg

# Optimize for faster processing
python optimize_reference_faces.py ~/.smugvision/reference_faces
```

## Usage

### Process an Album

**By URL:**
```bash
python -m smugvision --url "https://yoursite.smugmug.com/.../n-XXXXX/album-name"
```

**By Album Key:**
```bash
python -m smugvision --gallery abc123
```

### Preview Changes (Dry Run)

```bash
python -m smugvision --url "https://..." --dry-run
```

### Force Reprocess Already-Tagged Images

```bash
python -m smugvision --gallery abc123 --force-reprocess
```

### Include Videos

```bash
python -m smugvision --gallery abc123 --include-videos
```

## Verify Installation

### Test SmugMug Connection

```bash
python test_smugmug.py --url "https://..." --list
```

### Test Vision Model

```bash
python test_vision.py path/to/image.jpg
```

### Test Face Recognition

```bash
python debug_face_recognition.py path/to/image.jpg
```

## Common Issues

### "Ollama not responding"
```bash
# Start Ollama
ollama serve

# Verify model is installed
ollama list
```

### "SmugMug authentication failed"
- Re-run `python get_smugmug_tokens.py`
- Verify credentials in `~/.smugvision/config.yaml`

### "No faces detected"
- Ensure reference faces directory exists
- Try lowering `face_recognition.tolerance` in config
- Use clear, well-lit photos in reference faces

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Review [DESIGN.md](DESIGN.md) for architecture details
- Check [config.yaml.example](config.yaml.example) for all configuration options

## Example Workflow

```bash
# 1. Preview what would be generated
python -m smugvision --url "https://..." --dry-run

# 2. Process the album
python -m smugvision --url "https://..."

# 3. View results in SmugMug
# Captions and tags are now updated!
```

## Support

- **Issues:** [GitHub Issues](https://github.com/yourusername/smugvision/issues)
- **Documentation:** See README.md and DESIGN.md

---

**Ready to automate your photo metadata? Let's go! 🚀**

