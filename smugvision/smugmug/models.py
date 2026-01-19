"""Data models for SmugMug API resources."""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


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

