#!/usr/bin/env python3
"""Console script for interactive configuration setup."""

import sys
from smugvision.config import ConfigManager


def main():
    """Run interactive configuration setup."""
    print("=" * 70)
    print("smugVision Configuration Setup")
    print("=" * 70)
    print()
    
    try:
        config = ConfigManager.load(interactive=True)
        print()
        print("=" * 70)
        print("✓ Configuration saved successfully!")
        print("=" * 70)
        print()
        print(f"Configuration file: {config.config_path}")
        print()
        print("Next steps:")
        print("  1. Ensure Ollama is running: ollama serve")
        print("  2. Pull the vision model: ollama pull llama3.2-vision")
        print("  3. (Optional) Set up face recognition reference faces")
        print("  4. Process your first album: smugvision --url 'https://...'")
        print()
        return 0
    except KeyboardInterrupt:
        print()
        print("Configuration cancelled.")
        return 130
    except Exception as e:
        print()
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

