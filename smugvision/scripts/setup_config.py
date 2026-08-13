#!/usr/bin/env python3
"""Console script for interactive configuration setup."""

import sys
from typing import List, Optional

from smugvision.config import ConfigManager


def _installed_models(endpoint: str, timeout: float = 2.0) -> Optional[List[str]]:
    """Query Ollama for the vision models it currently serves.

    Delegates to :meth:`VisionModelFactory.list_models`, which is the one place that
    knows how to read Ollama's tag list across client versions. Hand-rolling the HTTP
    call here got the response shape wrong (newer servers key entries by ``model``,
    not ``name``).

    This is best effort only. Setup must succeed on a machine where Ollama is not
    installed, not running, or listening somewhere else.

    Args:
        endpoint: Base URL of the Ollama server, e.g. ``http://localhost:11434``.
        timeout: Seconds to wait before giving up on the request.

    Returns:
        A list of model names, or ``None`` if the server could not be reached.
    """
    try:
        import ollama

        from smugvision.vision import VisionModelFactory

        # Probe reachability first. list_models() deliberately never raises and falls
        # back to a static hint list, which would otherwise be presented here as
        # "installed" on a machine where Ollama is not running at all.
        ollama.Client(host=endpoint, timeout=timeout).list()

        models = VisionModelFactory.list_models(endpoint=endpoint, timeout=timeout)
        return models or None
    except Exception:
        return None


def _print_model_hint(configured_model: str, endpoint: str) -> None:
    """Tell the user how to get the configured vision model in place.

    Args:
        configured_model: Value of ``vision.model`` from the saved configuration.
        endpoint: Value of ``vision.endpoint`` from the saved configuration.
    """
    installed = _installed_models(endpoint)

    if installed is None:
        print(f"  2. Pull the vision model: ollama pull {configured_model}")
        print(f"     (could not reach Ollama at {endpoint} to check what is installed)")
        return

    if configured_model in installed:
        print(f"  2. Vision model '{configured_model}' is already installed - nothing to do")
        return

    print(f"  2. Pull the vision model: ollama pull {configured_model}")
    if installed:
        print("     Models currently installed:")
        for name in sorted(installed):
            print(f"       - {name}")
        print("     Any vision-capable model works - set vision.model in your config")
        print("     to one of the above if you would rather not pull another.")
    else:
        print(f"     (Ollama at {endpoint} reports no models installed yet)")


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

        model = config.get("vision.model", "")
        endpoint = config.get("vision.endpoint", "http://localhost:11434")

        print("Next steps:")
        print("  1. Ensure Ollama is running: ollama serve")
        _print_model_hint(model, endpoint)
        print("  3. (Optional) Set up face recognition reference faces")
        print("  4. Process your first album: smugvision --url 'https://...'")
        print()
        print("Useful settings in your config file (see config.yaml.example):")
        print("  vision.think          - false keeps reasoning models from burning tokens")
        print("  vision.keep_alive     - keeps the model loaded between images")
        print("  vision.single_call    - one request per image for caption + tags")
        print("  vision.validate_model - warns (never fails) if the model is not installed")
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
