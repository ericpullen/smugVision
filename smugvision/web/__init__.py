"""Web UI module for smugVision.

This module provides a local web interface for previewing and committing
AI-generated metadata changes to SmugMug albums.
"""

from .app import create_app

__all__ = ["create_app"]
