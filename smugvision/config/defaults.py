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
        # Any vision-capable model served by Ollama. Run `ollama list` to see what is
        # installed locally; the factory no longer restricts the name to an allow-list.
        # gemma4:latest (the ~9.6GB e4b variant) is the default because it runs on modest
        # hardware while still producing clean captions. On a machine with memory to spare,
        # gemma4:26b is worth setting instead: it is a mixture-of-experts model activating
        # only ~3.8B params per token, so it costs ~20% more latency than e4b for noticeably
        # better subject grounding. See scripts/benchmark_models.py to compare on your own
        # photos before switching.
        "model": "gemma4:latest",
        # Ollama endpoint. None (the default) lets the ollama client resolve the host
        # itself, which honours $OLLAMA_HOST and falls back to http://localhost:11434.
        # Setting a URL here overrides $OLLAMA_HOST.
        "endpoint": None,
        "temperature": 0.7,
        # Budget for ONE response that carries the caption AND the tags together.
        "max_tokens": 500,
        "timeout": 120,
        # Ollama `think` parameter: False | True | "low" | "medium" | "high" | None.
        # False keeps reasoning models from spending the whole token budget thinking;
        # None omits the parameter and lets the model decide.
        "think": False,
        # Ollama `keep_alive`: keeps the model resident between images in an album so
        # only the first image of a run pays load time.
        "keep_alive": "30m",
        # One request returns caption + tags. Set False for the legacy two-call path.
        "single_call": True,
        # Constrain the response with a JSON schema. Set False for free-text output
        # plus heuristic parsing (compatibility fallback for models that ignore or
        # mishandle schemas).
        "structured_output": True,
        # Downscale the long edge to this many pixels before base64 encoding.
        # 0 or None disables. Images are never upscaled.
        "max_image_dimension": 1568,
        "jpeg_quality": 85,
        # Warn (never fail) when the configured model is absent from Ollama's tag list.
        "validate_model": True,
    },
    # Face Recognition Configuration
    "face_recognition": {
        "enabled": True,
        "reference_faces_dir": str(Path.home() / ".smugvision" / "reference_faces"),
        # Embedding backend: "dlib" (default) or "insightface" (optional extra).
        "backend": "dlib",
        # dlib-only knobs. They are euclidean-distance / dlib-detector settings and
        # have no meaning for the insightface backend, which has its own block below.
        "tolerance": 0.6,
        "model": "cnn",
        "detection_scale": 0.5,
        # Backend-agnostic: minimum normalized 0.0-1.0 confidence for a named match.
        "min_confidence": 0.25,
        "use_cache": True,  # Cache face encodings for faster startup
        "cache_dir": str(Path.home() / ".smugvision" / "cache" / "face_encodings"),
        # Options for the optional InsightFace (ArcFace ONNX) backend. Ignored unless
        # backend == "insightface". similarity_threshold is a COSINE similarity, so
        # higher is stricter - it is not interchangeable with `tolerance`.
        "insightface": {
            "model_name": "buffalo_l",
            "det_size": [640, 640],
            "similarity_threshold": 0.4,
        },
    },
    # Processing Configuration
    "processing": {
        "marker_tag": "smugvision",
        "generate_captions": True,
        "generate_tags": True,
        "preserve_existing": True,
        # SmugMug size name. Matched case-insensitively by SmugMugClient.download_image,
        # so an existing lowercase "medium" in a user's config keeps working.
        "image_size": "Medium",
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
