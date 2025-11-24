#!/usr/bin/env python3
"""Test script for ConfigManager.

This script demonstrates how to use the ConfigManager to load, create,
and interact with configuration files.

Usage:
    # Load or create config interactively
    python test_config.py
    
    # Load specific config file
    python test_config.py path/to/config.yaml
    
    # Load without interactive prompts (will fail if required fields missing)
    python test_config.py --non-interactive
"""

import sys
import logging
from pathlib import Path

from smugvision.config import ConfigManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Test configuration manager."""
    # Parse arguments
    config_path = None
    interactive = True
    
    for arg in sys.argv[1:]:
        if arg == "--non-interactive":
            interactive = False
        elif not arg.startswith("--"):
            config_path = arg
    
    try:
        # Load configuration
        print("=" * 70)
        print("smugVision Configuration Manager Test")
        print("=" * 70)
        print()
        
        if config_path:
            print(f"Loading configuration from: {config_path}")
        else:
            print("Searching for configuration file in standard locations:")
            print("  1. ~/.smugvision/config.yaml (primary)")
            print("  2. ./config.yaml (development)")
        print()
        
        config = ConfigManager.load(
            config_path=config_path,
            interactive=interactive,
            create_if_missing=True
        )
        
        print("\n" + "=" * 70)
        print("Configuration loaded successfully!")
        print("=" * 70)
        print(f"\nConfiguration file: {config.config_path}")
        print()
        
        # Display some configuration values
        print("Current Configuration Values:")
        print("-" * 70)
        
        sections = [
            ("SmugMug API", [
                ("API Key", "smugmug.api_key", True),
                ("API Secret", "smugmug.api_secret", True),
                ("User Token", "smugmug.user_token", True),
                ("User Secret", "smugmug.user_secret", True),
            ]),
            ("Vision Model", [
                ("Model", "vision.model", False),
                ("Endpoint", "vision.endpoint", False),
                ("Temperature", "vision.temperature", False),
                ("Max Tokens", "vision.max_tokens", False),
            ]),
            ("Face Recognition", [
                ("Enabled", "face_recognition.enabled", False),
                ("Reference Faces Dir", "face_recognition.reference_faces_dir", False),
                ("Tolerance", "face_recognition.tolerance", False),
                ("Model", "face_recognition.model", False),
                ("Detection Scale", "face_recognition.detection_scale", False),
                ("Min Confidence", "face_recognition.min_confidence", False),
            ]),
            ("Processing", [
                ("Marker Tag", "processing.marker_tag", False),
                ("Generate Captions", "processing.generate_captions", False),
                ("Generate Tags", "processing.generate_tags", False),
                ("Use EXIF Location", "processing.use_exif_location", False),
            ]),
        ]
        
        for section_name, fields in sections:
            print(f"\n{section_name}:")
            for label, key, is_secret in fields:
                value = config.get(key)
                if is_secret and value:
                    # Mask secret values
                    display_value = value[:4] + "*" * (len(value) - 8) + value[-4:]
                else:
                    display_value = value
                print(f"  {label:.<30} {display_value}")
        
        print("\n" + "=" * 70)
        print("Test completed successfully!")
        print("=" * 70)
        
        # Demonstrate getting and setting values
        print("\nExample: Getting configuration values")
        print("-" * 70)
        print(f"Vision model: {config.get('vision.model')}")
        print(f"Temperature: {config.get('vision.temperature')}")
        print(f"Face recognition enabled: {config.get('face_recognition.enabled')}")
        print(f"Min confidence: {config.get('face_recognition.min_confidence')}")
        
        # Show how to use values that might not exist
        print(f"\nNon-existent key with default: {config.get('foo.bar', 'default_value')}")
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        logger.exception("Configuration test failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

