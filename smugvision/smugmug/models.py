"""Data models for SmugMug API resources."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

#: SmugMug node ``Type`` value for a folder (a container of other nodes).
NODE_TYPE_FOLDER = "Folder"

#: SmugMug node ``Type`` value for a user-visible album.
NODE_TYPE_ALBUM = "Album"

#: SmugMug node ``Type`` value for SmugMug's own internal albums (``Profile
#: Images``, ``Cover Images``). These expose a real ``AlbumKey`` but must never
#: be offered to the user as a processable gallery.
NODE_TYPE_SYSTEM_ALBUM = "System Album"

#: Label substituted for the root node, whose SmugMug ``Name`` is the empty
#: string (its ``UrlPath`` is just ``/``).
ROOT_NODE_LABEL = "All Galleries"


@dataclass
class Album:
    """Represents a SmugMug album (gallery).
    
    Attributes:
        album_key: Unique album identifier
        url_name: URL-friendly album name
        name: Display name of the album
        description: Album description
        image_count: Number of images in the album
        uri: API URI for the album
        web_uri: Web URL for viewing the album
        sort_method: How images are sorted
        sort_direction: Sort direction (ascending/descending)
    """
    album_key: str
    url_name: str
    name: str
    uri: str
    web_uri: str
    image_count: int = 0
    description: Optional[str] = None
    sort_method: Optional[str] = None
    sort_direction: Optional[str] = None
    
    @classmethod
    def from_api_response(cls, data: dict) -> "Album":
        """Create Album instance from SmugMug API response.
        
        Args:
            data: Album data from API response
            
        Returns:
            Album instance
        """
        return cls(
            album_key=data.get("AlbumKey", ""),
            url_name=data.get("UrlName", ""),
            name=data.get("Name", ""),
            description=data.get("Description"),
            image_count=data.get("ImageCount", 0),
            uri=data.get("Uri", ""),
            web_uri=data.get("WebUri", ""),
            sort_method=data.get("SortMethod"),
            sort_direction=data.get("SortDirection"),
        )
    
    def __str__(self) -> str:
        """Return string representation of album."""
        return f"Album({self.name}, {self.image_count} images)"


@dataclass
class AlbumImage:
    """Represents an image in a SmugMug album.
    
    Attributes:
        image_key: Unique image identifier
        album_key: Parent album identifier
        uri: API URI for the image
        web_uri: Web URL for viewing the image
        file_name: Original filename
        caption: Image caption
        keywords: List of keyword tags
        title: Image title
        format: Image format (JPG, PNG, etc.)
        archived_uri: URI of archived original
        archived_size: Size of archived original
        date: Date image was taken
        uploaded: Date image was uploaded
        modified: Date image was last modified
        is_video: Whether this is a video
        hidden: Whether image is hidden
        processing: Whether image is still processing
        uris: Dictionary of available image URIs by size
        latitude: GPS latitude from EXIF (if available)
        longitude: GPS longitude from EXIF (if available)
        altitude: GPS altitude from EXIF (if available)
    """
    image_key: str
    album_key: str
    uri: str
    web_uri: str
    file_name: str
    format: str
    caption: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    title: Optional[str] = None
    archived_uri: Optional[str] = None
    archived_size: Optional[int] = None
    date: Optional[str] = None
    uploaded: Optional[str] = None
    modified: Optional[str] = None
    is_video: bool = False
    hidden: bool = False
    processing: bool = False
    uris: dict = field(default_factory=dict)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    
    @property
    def has_gps(self) -> bool:
        """Check if image has GPS coordinates."""
        return self.latitude is not None and self.longitude is not None
    
    @classmethod
    def from_api_response(cls, data: dict, album_key: str = None) -> "AlbumImage":
        """Create AlbumImage instance from SmugMug API response.
        
        Args:
            data: Image data from API response
            album_key: Parent album key (if not in data)
            
        Returns:
            AlbumImage instance
        """
        # Keywords can be a string or list
        keywords = data.get("Keywords", [])
        if isinstance(keywords, str):
            # Split comma-separated keywords and clean up
            keywords = [k.strip() for k in keywords.split(",") if k.strip()]
        elif not isinstance(keywords, list):
            keywords = []
        
        # Parse GPS coordinates if available
        latitude = data.get("Latitude")
        longitude = data.get("Longitude")
        altitude = data.get("Altitude")
        
        # Convert to float if present (API may return as string)
        if latitude is not None:
            try:
                latitude = float(latitude)
            except (ValueError, TypeError):
                latitude = None
        if longitude is not None:
            try:
                longitude = float(longitude)
            except (ValueError, TypeError):
                longitude = None
        if altitude is not None:
            try:
                altitude = float(altitude)
            except (ValueError, TypeError):
                altitude = None
        
        return cls(
            image_key=data.get("ImageKey", ""),
            album_key=album_key or data.get("AlbumKey", ""),
            uri=data.get("Uri", ""),
            web_uri=data.get("WebUri", ""),
            file_name=data.get("FileName", ""),
            caption=data.get("Caption"),
            keywords=keywords,
            title=data.get("Title"),
            format=data.get("Format", ""),
            archived_uri=data.get("ArchivedUri"),
            archived_size=data.get("ArchivedSize"),
            date=data.get("Date"),
            uploaded=data.get("Uploaded"),
            modified=data.get("Modified"),
            is_video=data.get("IsVideo", False),
            hidden=data.get("Hidden", False),
            processing=data.get("Processing", False),
            uris=data.get("Uris", {}),
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
        )
    
    def has_marker_tag(self, marker_tag: str) -> bool:
        """Check if image has the specified marker tag.
        
        Args:
            marker_tag: Tag to check for
            
        Returns:
            True if marker tag exists in keywords
        """
        return marker_tag.lower() in [k.lower() for k in self.keywords]
    
    def get_download_url(self, size: str = "Medium") -> Optional[str]:
        """Get download URL for specified image size.
        
        Args:
            size: Image size (Thumb, Small, Medium, Large, XLarge, X2Large, X3Large, Original)
            
        Returns:
            Download URL if available, None otherwise
        """
        # Try to get from Uris dictionary
        if self.uris:
            size_uri = self.uris.get(f"Image{size}")
            if size_uri and isinstance(size_uri, dict):
                return size_uri.get("Uri")
        
        # Fallback to archived URI if requesting Original
        if size == "Original" and self.archived_uri:
            return self.archived_uri
        
        return None
    
    def __str__(self) -> str:
        """Return string representation of image."""
        caption_preview = self.caption[:50] + "..." if self.caption and len(self.caption) > 50 else self.caption or "No caption"
        tags_str = f"{len(self.keywords)} tags" if self.keywords else "No tags"
        return f"AlbumImage({self.file_name}, {caption_preview}, {tags_str})"


@dataclass
class NodeRef:
    """A lightweight reference to a node in the SmugMug node tree.

    Used for breadcrumb trails and to describe the node a listing belongs to.
    It deliberately carries only what a picker needs to render a crumb and
    navigate back to it.

    Attributes:
        node_id: SmugMug NodeID (navigation identity, NOT an album key)
        name: Display name; the root node's real name is empty, so
            ``ROOT_NODE_LABEL`` is substituted for it
        node_type: Raw SmugMug ``Type`` ("Folder", "Album", "System Album")
        url_path: SmugMug site path for the node (e.g. "/FamilyPhotos/2025")
        is_root: True when this is the user's root node
    """

    node_id: str
    name: str
    node_type: str = NODE_TYPE_FOLDER
    url_path: Optional[str] = None
    is_root: bool = False

    @property
    def is_folder(self) -> bool:
        """Whether this node can be listed (has children)."""
        return self.node_type == NODE_TYPE_FOLDER

    @classmethod
    def from_api_response(cls, data: dict) -> "NodeRef":
        """Create NodeRef from a SmugMug Node object.

        Args:
            data: Node data from a ``!children``, ``!parents`` or ``/node/<id>``
                API response

        Returns:
            NodeRef instance
        """
        is_root = bool(data.get("IsRoot", False))
        name = data.get("Name") or ""
        if not name and is_root:
            name = ROOT_NODE_LABEL
        return cls(
            node_id=data.get("NodeID", ""),
            name=name,
            node_type=data.get("Type", NODE_TYPE_FOLDER),
            url_path=data.get("UrlPath"),
            is_root=is_root,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for a JSON API response.

        Returns:
            JSON-safe dictionary
        """
        return {
            "node_id": self.node_id,
            "name": self.name,
            "type": "folder" if self.is_folder else "album",
            "url_path": self.url_path,
            "is_root": self.is_root,
        }

    def __str__(self) -> str:
        """Return string representation of the node reference."""
        return f"NodeRef({self.name}, {self.node_id}, {self.node_type})"


@dataclass
class BrowseNode:
    """A single browsable entry in the node tree: a folder or an album.

    One dataclass covers both kinds because a SmugMug folder can contain a mix
    of folders and albums and a picker renders them in the same level. Use
    ``is_album`` / ``is_folder`` to discriminate. Album-only fields are None on
    folders, and ``album_key`` is only ever populated for albums.

    ``node_id`` and ``album_key`` are DIFFERENT identifiers for the same album
    (node ``Rc9FdK`` is album ``Ab3kZq``). Navigate with ``node_id``; process
    with ``album_key``.

    Attributes:
        node_id: SmugMug NodeID, used for navigation
        name: Display name
        node_type: "folder" or "album"
        album_key: Album key (albums only) - this is what the processing
            pipeline takes
        url_name: URL-friendly name / slug
        url_path: SmugMug site path
        web_uri: Full https URL for viewing on smugmug.com. For albums this
            embeds the PARENT folder's ``n-XXXXX`` node id plus the album slug,
            which is the form the URL parser already understands
        image_count: Number of images (albums only, None when not expanded)
        description: Node/album description
        date: Creation date (album ``Date`` / folder ``DateAdded``)
        last_updated: Last modification date (album ``LastUpdated`` /
            folder ``DateModified``)
        privacy: SmugMug privacy setting ("Public", "Private", "Unlisted")
        has_children: Whether a folder can be descended into
    """

    node_id: str
    name: str
    node_type: str
    album_key: Optional[str] = None
    url_name: str = ""
    url_path: Optional[str] = None
    web_uri: Optional[str] = None
    image_count: Optional[int] = None
    description: Optional[str] = None
    date: Optional[str] = None
    last_updated: Optional[str] = None
    privacy: Optional[str] = None
    has_children: bool = False

    @property
    def is_album(self) -> bool:
        """Whether this entry is a selectable album."""
        return self.node_type == "album"

    @property
    def is_folder(self) -> bool:
        """Whether this entry is a folder that can be descended into."""
        return self.node_type == "folder"

    @staticmethod
    def _album_key_from_uris(uris: dict) -> Optional[str]:
        """Extract an album key from a node's ``Uris.Album.Uri``.

        Args:
            uris: The node's ``Uris`` dictionary

        Returns:
            Album key, or None if the node exposes no album URI
        """
        album_uri = uris.get("Album") if isinstance(uris, dict) else None
        if isinstance(album_uri, dict):
            uri = album_uri.get("Uri", "")
        elif isinstance(album_uri, str):
            # _shorturis=1 collapses Uris entries to bare strings
            uri = album_uri
        else:
            return None
        if "/album/" in uri:
            return uri.split("/album/")[-1] or None
        return None

    @staticmethod
    def _expanded_album(uris: dict) -> Dict[str, Any]:
        """Return the inline-expanded Album resource, if the request asked for it.

        With ``_expand=Album&_expandmethod=inline`` SmugMug nests the whole
        Album resource at ``Uris.Album.Album``.

        Args:
            uris: The node's ``Uris`` dictionary

        Returns:
            The expanded Album dictionary, or an empty dict when absent
        """
        album_uri = uris.get("Album") if isinstance(uris, dict) else None
        if isinstance(album_uri, dict):
            expanded = album_uri.get("Album")
            if isinstance(expanded, dict):
                return expanded
        return {}

    @classmethod
    def from_api_response(cls, data: dict) -> Optional["BrowseNode"]:
        """Create a BrowseNode from a SmugMug Node object.

        Args:
            data: A single Node entry from a ``!children`` response, optionally
                carrying an inline-expanded Album at ``Uris.Album.Album``

        Returns:
            BrowseNode for folders and albums, or None for node types that are
            deliberately not offered to the user: ``System Album`` (SmugMug's own
            Profile Images / Cover Images) and any type SmugMug adds in future.

        Raises:
            ValueError: If the node is a folder or album but is malformed - no
                NodeID, or an album with no resolvable album key. Callers should
                catch this, log it and skip the single row, marking the listing
                partial rather than failing the whole level.
        """
        node_type_raw = data.get("Type")
        node_id = data.get("NodeID")

        if node_type_raw not in (NODE_TYPE_FOLDER, NODE_TYPE_ALBUM):
            # "System Album" (Profile Images / Cover Images) and anything new
            # SmugMug invents later are not user-processable galleries.
            logger.debug(f"Skipping node type {node_type_raw!r}: {data.get('Name')!r}")
            return None

        if not node_id:
            raise ValueError(f"node has no NodeID (Name={data.get('Name')!r})")

        if node_type_raw == NODE_TYPE_FOLDER:
            return cls(
                node_id=node_id,
                name=data.get("Name") or "",
                node_type="folder",
                url_name=data.get("UrlName", ""),
                url_path=data.get("UrlPath"),
                web_uri=data.get("WebUri"),
                description=data.get("Description") or None,
                date=data.get("DateAdded"),
                last_updated=data.get("DateModified"),
                privacy=data.get("Privacy"),
                has_children=bool(data.get("HasChildren", False)),
            )

        uris = data.get("Uris", {}) or {}
        expanded = cls._expanded_album(uris)
        album_key = expanded.get("AlbumKey") or cls._album_key_from_uris(uris)
        if not album_key:
            raise ValueError(
                f"album node {node_id} ({data.get('Name')!r}) exposes no album key"
            )

        image_count = expanded.get("ImageCount")
        if image_count is not None:
            try:
                image_count = int(image_count)
            except (TypeError, ValueError):
                image_count = None

        return cls(
            node_id=node_id,
            name=expanded.get("Name") or data.get("Name") or "",
            node_type="album",
            album_key=album_key,
            url_name=expanded.get("UrlName") or data.get("UrlName", ""),
            url_path=expanded.get("UrlPath") or data.get("UrlPath"),
            web_uri=expanded.get("WebUri") or data.get("WebUri"),
            image_count=image_count,
            description=expanded.get("Description") or data.get("Description") or None,
            date=expanded.get("Date") or data.get("DateAdded"),
            last_updated=expanded.get("LastUpdated") or data.get("DateModified"),
            privacy=expanded.get("Privacy") or data.get("Privacy"),
            has_children=False,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for a JSON API response.

        ``album_key`` is omitted for folders so a client cannot accidentally
        submit a folder for processing.

        Returns:
            JSON-safe dictionary
        """
        payload: Dict[str, Any] = {
            "node_id": self.node_id,
            "name": self.name,
            "type": self.node_type,
            "url_name": self.url_name,
            "url_path": self.url_path,
            "web_uri": self.web_uri,
            "description": self.description,
            "date": self.date,
            "last_updated": self.last_updated,
            "privacy": self.privacy,
        }
        if self.is_album:
            payload["album_key"] = self.album_key
            payload["image_count"] = self.image_count
        else:
            payload["has_children"] = self.has_children
        return payload

    def __str__(self) -> str:
        """Return string representation of the browse node."""
        if self.is_album:
            count = "?" if self.image_count is None else self.image_count
            return f"BrowseNode(album {self.name}, {self.album_key}, {count} images)"
        return f"BrowseNode(folder {self.name}, {self.node_id})"


@dataclass
class NodeListing:
    """One level of the SmugMug node tree, ready to render in a picker.

    Attributes:
        node: The node whose children these are
        breadcrumb: Ancestor chain ordered root-first and ending with ``node``
            itself, so a UI can render it left to right
        folders: Child folders, navigable
        albums: Child albums, selectable for processing
        partial: True when at least one piece of the listing could not be
            retrieved (e.g. the breadcrumb lookup failed) and what is returned
            is incomplete
    """

    node: NodeRef
    breadcrumb: List[NodeRef] = field(default_factory=list)
    folders: List[BrowseNode] = field(default_factory=list)
    albums: List[BrowseNode] = field(default_factory=list)
    partial: bool = False

    @property
    def total(self) -> int:
        """Total number of listed children (folders plus albums)."""
        return len(self.folders) + len(self.albums)

    @property
    def total_images(self) -> int:
        """Sum of image counts across child albums, ignoring unknown counts."""
        return sum(a.image_count or 0 for a in self.albums)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for a JSON API response.

        Returns:
            JSON-safe dictionary with ``node``, ``breadcrumb``, ``folders``,
            ``albums``, ``total`` and ``partial`` keys
        """
        return {
            "node": self.node.to_dict(),
            "breadcrumb": [c.to_dict() for c in self.breadcrumb],
            "folders": [f.to_dict() for f in self.folders],
            "albums": [a.to_dict() for a in self.albums],
            "total": self.total,
            "partial": self.partial,
        }

    def __str__(self) -> str:
        """Return string representation of the listing."""
        return (
            f"NodeListing({self.node.name}: {len(self.folders)} folders, "
            f"{len(self.albums)} albums)"
        )


@dataclass
class AlbumSearchResult:
    """Result of a bounded recursive walk of the node tree.

    ``truncated`` is the important field: a walk that hit one of its bounds
    returns a PARTIAL album list, and a caller that ignores this will silently
    show the user an incomplete tree.

    Attributes:
        albums: Albums found, in discovery order (breadth-first)
        folders_scanned: Number of folders whose children were listed
        max_depth_reached: Deepest level actually visited (root is 0)
        truncated: True when a bound stopped the walk early
        truncation_reason: Human-readable reason when ``truncated`` is True
        errors: Names/ids of nodes that could not be listed and were skipped
    """

    albums: List[BrowseNode] = field(default_factory=list)
    folders_scanned: int = 0
    max_depth_reached: int = 0
    truncated: bool = False
    truncation_reason: Optional[str] = None
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for a JSON API response.

        Returns:
            JSON-safe dictionary
        """
        return {
            "albums": [a.to_dict() for a in self.albums],
            "folders_scanned": self.folders_scanned,
            "max_depth_reached": self.max_depth_reached,
            "truncated": self.truncated,
            "truncation_reason": self.truncation_reason,
            "errors": self.errors,
        }

    def __str__(self) -> str:
        """Return string representation of the search result."""
        suffix = f", TRUNCATED: {self.truncation_reason}" if self.truncated else ""
        return (
            f"AlbumSearchResult({len(self.albums)} albums, "
            f"{self.folders_scanned} folders scanned{suffix})"
        )

