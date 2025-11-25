"""SmugMug API integration for smugVision."""

from smugvision.smugmug.client import SmugMugClient
from smugvision.smugmug.models import Album, AlbumImage
from smugvision.smugmug.exceptions import (
    SmugMugError,
    SmugMugAPIError,
    SmugMugAuthError,
    SmugMugNotFoundError,
    SmugMugRateLimitError,
)

__all__ = [
    "SmugMugClient",
    "Album",
    "AlbumImage",
    "SmugMugError",
    "SmugMugAPIError",
    "SmugMugAuthError",
    "SmugMugNotFoundError",
    "SmugMugRateLimitError",
]

