"""Default configuration values for smugVision."""

from pathlib import Path

# Default configuration dictionary
DEFAULT_CONFIG = {
    # SmugMug API Configuration
    "smugmug": {
        "api_key": "",
        "api_secret": "",
        "user_token": "",
        "user_secret": "",
    },
    
    # Vision Model Configuration
    "vision": {
        "model": "qwen3-vl:8b",
        "endpoint": "http://localhost:11434",
        "temperature": 0.7,
        "max_tokens": 500,  # Thinking models need more tokens
        "timeout": 120,
    },
    
    # Face Recognition Configuration
    "face_recognition": {
        "enabled": True,
        "reference_faces_dir": str(Path.home() / ".smugvision" / "reference_faces"),
        "tolerance": 0.6,
        "model": "cnn",
        "detection_scale": 0.5,
        "min_confidence": 0.25,
        "use_cache": True,  # Cache face encodings for faster startup
        "cache_dir": str(Path.home() / ".smugvision" / "cache" / "face_encodings"),
    },
    
    # Processing Configuration
    "processing": {
        "marker_tag": "smugvision",
        "generate_captions": True,
        "generate_tags": True,
        "preserve_existing": True,
        "image_size": "medium",
        "use_exif_location": True,
    },
    
    # Location Resolution Configuration
    "location": {
        "custom_locations_file": str(Path.home() / ".smugvision" / "locations.yaml"),
        "check_custom_first": True,
        "use_aliases_as_tags": True,
    },
    
    # Prompt Configuration
    "prompts": {
        "caption": (
            "You are a photo captioning assistant. Write exactly ONE caption (1-2 sentences) "
            "for this image. Describe the main subject, setting, and activity. "
            "IMPORTANT: Output ONLY the caption text. No options, no explanations, no "
            "introductions like 'Here is...' - just the caption itself."
        ),
        "tags": (
            "Output a comma-separated list of 5-10 keyword tags for this image. "
            "IMPORTANT: Output ONLY the tags separated by commas. No explanations, "
            "no numbering, no extra text. Example output: dog, playing, park, sunny, happy"
        ),
    },
    
    # Cache Configuration
    "cache": {
        "directory": str(Path.home() / ".smugvision" / "cache"),
        "clear_on_exit": False,
        "preserve_structure": True,
    },
    
    # Logging Configuration
    "logging": {
        "level": "INFO",
        "file": str(Path.home() / ".smugvision" / "smugvision.log"),
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    },
}

# Required configuration fields (must be provided by user)
REQUIRED_FIELDS = [
    "smugmug.api_key",
    "smugmug.api_secret",
    "smugmug.user_token",
    "smugmug.user_secret",
]

# Configuration field descriptions for interactive setup
FIELD_DESCRIPTIONS = {
    "smugmug.api_key": "SmugMug API Key (from https://api.smugmug.com/api/developer/apply)",
    "smugmug.api_secret": "SmugMug API Secret",
    "smugmug.user_token": "SmugMug User OAuth Token",
    "smugmug.user_secret": "SmugMug User OAuth Secret",
}

