"""SmugMug API integration for smugVision."""

from smugvision.smugmug.client import SmugMugClient
from smugvision.smugmug.models import (
    NODE_TYPE_ALBUM,
    NODE_TYPE_FOLDER,
    NODE_TYPE_SYSTEM_ALBUM,
    ROOT_NODE_LABEL,
    Album,
    AlbumImage,
    AlbumSearchResult,
    BrowseNode,
    NodeListing,
    NodeRef,
)
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
    "BrowseNode",
    "NodeRef",
    "NodeListing",
    "AlbumSearchResult",
    "NODE_TYPE_FOLDER",
    "NODE_TYPE_ALBUM",
    "NODE_TYPE_SYSTEM_ALBUM",
    "ROOT_NODE_LABEL",
    "SmugMugError",
    "SmugMugAPIError",
    "SmugMugAuthError",
    "SmugMugNotFoundError",
    "SmugMugRateLimitError",
]

