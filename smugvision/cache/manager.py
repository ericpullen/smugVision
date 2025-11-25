"""Cache manager for local image storage."""

import logging
from pathlib import Path
from typing import Optional, List
import shutil

from smugvision.smugmug.models import AlbumImage

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages local cache of downloaded images.
    
    This class handles downloading images from SmugMug and organizing them
    in a local cache directory, optionally preserving the folder structure.
    
    Attributes:
        cache_dir: Base directory for cached images
        preserve_structure: Whether to mirror SmugMug folder structure
    """
    
    def __init__(
        self,
        cache_dir: str,
        preserve_structure: bool = True
    ) -> None:
        """Initialize cache manager.
        
        Args:
            cache_dir: Base directory for cached images
            preserve_structure: Whether to preserve folder/album structure
        """
        self.cache_dir = Path(cache_dir).expanduser()
        self.preserve_structure = preserve_structure
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Cache manager initialized: {self.cache_dir}")
    
    def get_album_cache_dir(
        self,
        album_name: str,
        folder_path: Optional[str] = None
    ) -> Path:
        """Get cache directory path for an album.
        
        Args:
            album_name: Album name
            folder_path: Optional folder path (e.g., "Gallery/Year")
            
        Returns:
            Path to album cache directory
        """
        if self.preserve_structure and folder_path:
            # Create nested structure
            cache_path = self.cache_dir / folder_path / album_name
        else:
            # Flat structure
            cache_path = self.cache_dir / album_name
        
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path
    
    def get_cached_image_path(
        self,
        image: AlbumImage,
        album_name: str,
        folder_path: Optional[str] = None
    ) -> Path:
        """Get the path where an image would be cached.
        
        Args:
            image: AlbumImage object
            album_name: Album name
            folder_path: Optional folder path
            
        Returns:
            Path to cached image file
        """
        album_dir = self.get_album_cache_dir(album_name, folder_path)
        return album_dir / image.file_name
    
    def is_image_cached(
        self,
        image: AlbumImage,
        album_name: str,
        folder_path: Optional[str] = None
    ) -> bool:
        """Check if an image is already cached.
        
        Args:
            image: AlbumImage object
            album_name: Album name
            folder_path: Optional folder path
            
        Returns:
            True if image exists in cache
        """
        cache_path = self.get_cached_image_path(image, album_name, folder_path)
        return cache_path.exists()
    
    def clear_album_cache(
        self,
        album_name: str,
        folder_path: Optional[str] = None
    ) -> None:
        """Clear cache for a specific album.
        
        Args:
            album_name: Album name
            folder_path: Optional folder path
        """
        album_dir = self.get_album_cache_dir(album_name, folder_path)
        
        if album_dir.exists():
            logger.info(f"Clearing cache: {album_dir}")
            shutil.rmtree(album_dir)
            logger.debug(f"Removed {album_dir}")
    
    def clear_all_cache(self) -> None:
        """Clear entire cache directory.
        
        Warning: This removes all cached images!
        """
        if self.cache_dir.exists():
            logger.warning(f"Clearing entire cache: {self.cache_dir}")
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cache cleared")
    
    def get_cache_stats(self) -> dict:
        """Get statistics about the cache.
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.cache_dir.exists():
            return {
                "total_size": 0,
                "file_count": 0,
                "album_count": 0
            }
        
        total_size = 0
        file_count = 0
        albums = set()
        
        for item in self.cache_dir.rglob("*"):
            if item.is_file():
                total_size += item.stat().st_size
                file_count += 1
                # Count parent directory as album
                albums.add(item.parent)
        
        return {
            "total_size": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "file_count": file_count,
            "album_count": len(albums)
        }
    
    def list_cached_albums(self) -> List[str]:
        """List all cached albums.
        
        Returns:
            List of album directory names
        """
        if not self.cache_dir.exists():
            return []
        
        albums = []
        for item in self.cache_dir.iterdir():
            if item.is_dir():
                albums.append(item.name)
        
        return sorted(albums)

