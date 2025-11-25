"""SmugMug API client for gallery and image operations."""

import logging
import time
from typing import List, Optional, Dict, Any
from pathlib import Path

import requests
from requests_oauthlib import OAuth1

from smugvision.smugmug.models import Album, AlbumImage
from smugvision.smugmug.exceptions import (
    SmugMugAPIError,
    SmugMugAuthError,
    SmugMugNotFoundError,
    SmugMugRateLimitError,
)

logger = logging.getLogger(__name__)


class SmugMugClient:
    """Client for interacting with SmugMug API.
    
    This class handles authentication via OAuth 1.0a, API requests, and data
    retrieval from SmugMug galleries and images. It provides methods for
    listing albums, retrieving images, and updating image metadata.
    
    Attributes:
        api_key: SmugMug API key
        api_secret: SmugMug API secret
        access_token: OAuth access token
        access_token_secret: OAuth access token secret
        base_url: Base URL for SmugMug API v2
        auth: OAuth1 authentication object
    """
    
    API_VERSION = "v2"
    BASE_URL = f"https://api.smugmug.com/api/{API_VERSION}"
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        access_token: str,
        access_token_secret: str,
        timeout: int = 30
    ) -> None:
        """Initialize SmugMug client with OAuth credentials.
        
        Args:
            api_key: SmugMug API key
            api_secret: SmugMug API secret
            access_token: OAuth access token
            access_token_secret: OAuth access token secret
            timeout: Request timeout in seconds
            
        Raises:
            SmugMugAuthError: If credentials are invalid
        """
        if not all([api_key, api_secret, access_token, access_token_secret]):
            raise SmugMugAuthError(
                "All OAuth credentials are required: "
                "api_key, api_secret, access_token, access_token_secret"
            )
        
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.access_token_secret = access_token_secret
        self.timeout = timeout
        
        # Create OAuth1 authentication
        self.auth = OAuth1(
            api_key,
            api_secret,
            access_token,
            access_token_secret,
            signature_type='auth_header'
        )
        
        logger.info("SmugMug client initialized")
        
        # Verify authentication by getting user info
        try:
            self._verify_authentication()
        except Exception as e:
            raise SmugMugAuthError(
                f"Failed to authenticate with SmugMug: {e}"
            ) from e
    
    def _verify_authentication(self) -> Dict[str, Any]:
        """Verify authentication by retrieving authenticated user info.
        
        Returns:
            User data dictionary
            
        Raises:
            SmugMugAuthError: If authentication fails
        """
        try:
            response = self._request("GET", "/api/v2!authuser")
            user = response.get("Response", {}).get("User", {})
            nickname = user.get("NickName", "Unknown")
            logger.info(f"Successfully authenticated as: {nickname}")
            return user
        except SmugMugAPIError as e:
            raise SmugMugAuthError(
                f"Authentication failed: {e}"
            ) from e
    
    def get_user_root_node(self) -> str:
        """Get the authenticated user's root node ID.
        
        Returns:
            Root node ID
            
        Raises:
            SmugMugAPIError: If request fails
        """
        try:
            response = self._request("GET", "/api/v2!authuser")
            user = response.get("Response", {}).get("User", {})
            
            # Get node URI
            uris = user.get("Uris", {})
            node_uri = uris.get("Node", {})
            if node_uri:
                uri = node_uri.get("Uri", "")
                # Extract node ID from URI
                if "/node/" in uri:
                    node_id = uri.split("/node/")[-1]
                    logger.debug(f"User root node: {node_id}")
                    return node_id
            
            raise SmugMugAPIError("Could not find user root node")
        except Exception as e:
            raise SmugMugAPIError(f"Failed to get user root node: {e}") from e
    
    def find_node_by_path(self, path: str) -> Optional[str]:
        """Find a node ID by navigating a path from root.
        
        Args:
            path: Path like "Gallery/Year"
            
        Returns:
            Node ID if found, None otherwise
        """
        logger.info(f"Finding node by path: {path}")
        
        # Get root node
        try:
            current_node_id = self.get_user_root_node()
        except Exception as e:
            logger.error(f"Could not get root node: {e}")
            return None
        
        # Split path and navigate
        parts = [p for p in path.split('/') if p]
        
        for part in parts:
            logger.debug(f"Looking for '{part}' under node {current_node_id}")
            
            try:
                children = self.get_node_children(current_node_id)
            except Exception as e:
                logger.error(f"Could not get children of node {current_node_id}: {e}")
                return None
            
            # Find matching folder
            found = False
            for child in children:
                child_name = child.get("Name", "")
                child_url_name = child.get("UrlName", "")
                child_type = child.get("Type")
                
                if child_type == "Folder" and (child_name == part or child_url_name == part):
                    current_node_id = child.get("NodeID")
                    if current_node_id:
                        found = True
                        logger.debug(f"Found '{part}' -> node {current_node_id}")
                        break
            
            if not found:
                logger.warning(f"Could not find '{part}' under current node")
                return None
        
        logger.info(f"Resolved path '{path}' to node: {current_node_id}")
        return current_node_id
    
    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Make authenticated request to SmugMug API.
        
        Args:
            method: HTTP method (GET, POST, PATCH, DELETE)
            endpoint: API endpoint (with or without base URL)
            params: Query parameters
            json_data: JSON body data
            headers: Additional headers
            
        Returns:
            Response data dictionary
            
        Raises:
            SmugMugAPIError: If request fails
            SmugMugNotFoundError: If resource not found (404)
            SmugMugRateLimitError: If rate limit exceeded (429)
        """
        # Build full URL
        if endpoint.startswith("http"):
            url = endpoint
        elif endpoint.startswith("/api/v2"):
            url = f"https://api.smugmug.com{endpoint}"
        else:
            url = f"{self.BASE_URL}{endpoint}"
        
        # Default headers
        request_headers = {
            "Accept": "application/json",
            "User-Agent": "smugVision/0.1.0"
        }
        if headers:
            request_headers.update(headers)
        
        # Log request
        logger.debug(f"SmugMug API {method} {url}")
        if params:
            logger.debug(f"  Params: {params}")
        
        try:
            response = requests.request(
                method=method,
                url=url,
                auth=self.auth,
                params=params,
                json=json_data,
                headers=request_headers,
                timeout=self.timeout
            )
            
            # Log response status
            logger.debug(f"  Response: {response.status_code}")
            
            # Handle error status codes
            if response.status_code == 404:
                raise SmugMugNotFoundError(
                    f"Resource not found: {endpoint}",
                    status_code=404,
                    response=response.json() if response.content else None
                )
            elif response.status_code == 429:
                # Rate limit exceeded
                retry_after = response.headers.get("Retry-After")
                retry_seconds = int(retry_after) if retry_after else None
                raise SmugMugRateLimitError(
                    "Rate limit exceeded",
                    retry_after=retry_seconds
                )
            elif response.status_code == 401:
                raise SmugMugAuthError(
                    "Authentication failed. Check your API credentials."
                )
            elif not response.ok:
                error_msg = f"API request failed with status {response.status_code}"
                try:
                    error_data = response.json()
                    if "Message" in error_data:
                        error_msg = error_data["Message"]
                except:
                    pass
                raise SmugMugAPIError(
                    error_msg,
                    status_code=response.status_code,
                    response=response.json() if response.content else None
                )
            
            # Parse JSON response
            return response.json()
            
        except requests.exceptions.Timeout as e:
            raise SmugMugAPIError(
                f"Request timeout after {self.timeout} seconds: {endpoint}"
            ) from e
        except requests.exceptions.RequestException as e:
            raise SmugMugAPIError(
                f"Request failed: {e}"
            ) from e
    
    def get_node_children(
        self,
        node_id: str,
        start: int = 1,
        count: int = 100
    ) -> List[Dict[str, Any]]:
        """Get children (albums and folders) of a node.
        
        This method handles pagination automatically to retrieve all children.
        
        Args:
            node_id: Node ID (e.g., from URL like n-ABC123)
            start: Starting position (1-indexed)
            count: Number of children per page (max 100)
            
        Returns:
            List of child nodes with their details
            
        Raises:
            SmugMugNotFoundError: If node not found
            SmugMugAPIError: If request fails
        """
        logger.info(f"Fetching children of node: {node_id}")
        
        all_children = []
        current_start = start
        
        while True:
            endpoint = f"/node/{node_id}!children"
            params = {
                "start": current_start,
                "count": min(count, 100)  # Max 100 per page
            }
            
            response = self._request("GET", endpoint, params=params)
            
            response_data = response.get("Response", {})
            children = response_data.get("Node", [])
            
            if not children:
                break
            
            all_children.extend(children)
            logger.debug(f"Retrieved {len(children)} children (start={current_start})")
            
            # Check if there are more pages
            pages = response_data.get("Pages", {})
            if not pages.get("NextPage"):
                break
            
            current_start += len(children)
        
        logger.debug(f"Found {len(all_children)} total children under node {node_id}")
        return all_children
    
    def find_albums_by_name(
        self,
        node_id: str,
        album_name: str,
        exact_match: bool = False,
        recursive: bool = True,
        max_depth: int = 3
    ) -> List[Album]:
        """Find albums by name under a specific node.
        
        Args:
            node_id: Node ID to search under
            album_name: Album name to search for
            exact_match: If True, require exact name match (case-insensitive)
            recursive: If True, search in subfolders too
            max_depth: Maximum depth to recurse (default 3)
            
        Returns:
            List of matching Album objects
            
        Raises:
            SmugMugAPIError: If request fails
        """
        logger.info(f"Searching for album '{album_name}' under node {node_id} (recursive={recursive})")
        
        def search_node(current_node_id: str, depth: int = 0) -> List[Album]:
            """Recursively search for albums."""
            if depth > max_depth:
                return []
            
            try:
                children = self.get_node_children(current_node_id)
            except Exception as e:
                logger.warning(f"Could not access node {current_node_id}: {e}")
                return []
            
            matching_albums = []
            search_lower = album_name.lower()
            folders_to_search = []
            
            for child in children:
                child_type = child.get("Type")
                
                if child_type == "Album":
                    child_name = child.get("Name", "")
                    url_name = child.get("UrlName", "")
                    
                    # Check if it matches
                    matches = False
                    if exact_match:
                        matches = (child_name.lower() == search_lower or 
                                  url_name.lower() == search_lower)
                    else:
                        matches = (search_lower in child_name.lower() or 
                                  search_lower in url_name.lower())
                    
                    if matches:
                        # Extract album key from URI
                        uris = child.get("Uris", {})
                        album_uri = uris.get("Album", {})
                        if album_uri:
                            uri = album_uri.get("Uri", "")
                            if "/album/" in uri:
                                album_key = uri.split("/album/")[-1]
                                try:
                                    album = self.get_album(album_key)
                                    matching_albums.append(album)
                                    logger.debug(f"Found matching album: {album.name} ({album_key})")
                                except Exception as e:
                                    logger.warning(f"Could not fetch album {album_key}: {e}")
                
                elif child_type == "Folder" and recursive:
                    # Add folder to search list
                    child_node_id = child.get("NodeID")
                    if child_node_id:
                        folders_to_search.append(child_node_id)
            
            # Search subfolders
            if recursive and folders_to_search:
                logger.debug(f"Searching {len(folders_to_search)} subfolder(s) at depth {depth + 1}")
                for folder_node_id in folders_to_search:
                    matching_albums.extend(search_node(folder_node_id, depth + 1))
            
            return matching_albums
        
        matching_albums = search_node(node_id)
        logger.info(f"Found {len(matching_albums)} matching album(s)")
        return matching_albums
    
    def resolve_album_key(
        self,
        identifier: str,
        node_id: Optional[str] = None
    ) -> str:
        """Resolve an album identifier to an album key.
        
        This method tries to determine if the identifier is:
        1. Already an album key (try to fetch it)
        2. An album name (search for it under node_id)
        
        Args:
            identifier: Album key or album name
            node_id: Node ID to search under (required if identifier is a name)
            
        Returns:
            Album key
            
        Raises:
            SmugMugNotFoundError: If album not found
            SmugMugAPIError: If resolution fails
        """
        # First, try as album key
        try:
            album = self.get_album(identifier)
            logger.debug(f"'{identifier}' is a valid album key")
            return identifier
        except SmugMugNotFoundError:
            # Not a valid album key, try as name
            if not node_id:
                raise SmugMugAPIError(
                    f"'{identifier}' is not a valid album key. "
                    "Provide node_id to search by name."
                )
            
            # Search for album by name
            logger.debug(f"'{identifier}' not found as key, searching as name...")
            albums = self.find_albums_by_name(node_id, identifier)
            
            if not albums:
                raise SmugMugNotFoundError(
                    f"No albums found matching '{identifier}' under node {node_id}",
                    status_code=404
                )
            
            if len(albums) > 1:
                names = [a.name for a in albums]
                raise SmugMugAPIError(
                    f"Multiple albums found matching '{identifier}': {names}\n"
                    f"Please be more specific or use the album key directly."
                )
            
            logger.info(f"Resolved '{identifier}' to album: {albums[0].name}")
            return albums[0].album_key
    
    def get_album(self, album_key: str) -> Album:
        """Get album details by album key.
        
        Args:
            album_key: Album key (unique identifier)
            
        Returns:
            Album object
            
        Raises:
            SmugMugNotFoundError: If album not found
            SmugMugAPIError: If request fails
        """
        logger.info(f"Fetching album: {album_key}")
        
        endpoint = f"/album/{album_key}"
        response = self._request("GET", endpoint)
        
        album_data = response.get("Response", {}).get("Album", {})
        album = Album.from_api_response(album_data)
        
        logger.debug(f"Retrieved album: {album.name} ({album.image_count} images)")
        return album
    
    def get_album_images(
        self,
        album_key: str,
        start: int = 1,
        count: int = 100
    ) -> List[AlbumImage]:
        """Get images from an album.
        
        This method handles pagination automatically to retrieve all images.
        SmugMug returns images in pages (default 100 per page).
        The _expandmethod parameter requests full image size URIs.
        
        Args:
            album_key: Album key (unique identifier)
            start: Starting position (1-indexed)
            count: Number of images per page (max 100)
            
        Returns:
            List of AlbumImage objects
            
        Raises:
            SmugMugAPIError: If request fails
        """
        logger.info(f"Fetching images from album: {album_key}")
        
        all_images = []
        current_start = start
        
        while True:
            endpoint = f"/album/{album_key}!images"
            params = {
                "start": current_start,
                "count": min(count, 100),  # Max 100 per page
                "_expandmethod": "inline",  # Request full URIs
                "_expand": "ImageDownload"  # Include download URLs
            }
            
            response = self._request("GET", endpoint, params=params)
            
            response_data = response.get("Response", {})
            images_data = response_data.get("AlbumImage", [])
            
            if not images_data:
                break
            
            # Convert to AlbumImage objects
            for image_data in images_data:
                image = AlbumImage.from_api_response(image_data, album_key)
                all_images.append(image)
            
            logger.debug(f"Retrieved {len(images_data)} images (start={current_start})")
            
            # Check if there are more pages
            pages = response_data.get("Pages", {})
            if not pages.get("NextPage"):
                break
            
            current_start += len(images_data)
        
        logger.info(f"Retrieved {len(all_images)} total images from album {album_key}")
        return all_images
    
    def get_image(self, image_key: str) -> AlbumImage:
        """Get image details by image key.
        
        Args:
            image_key: Image key (unique identifier)
            
        Returns:
            AlbumImage object
            
        Raises:
            SmugMugNotFoundError: If image not found
            SmugMugAPIError: If request fails
        """
        logger.info(f"Fetching image: {image_key}")
        
        endpoint = f"/image/{image_key}"
        response = self._request("GET", endpoint)
        
        image_data = response.get("Response", {}).get("Image", {})
        image = AlbumImage.from_api_response(image_data)
        
        logger.debug(f"Retrieved image: {image.file_name}")
        return image
    
    def update_image_metadata(
        self,
        image_key: str,
        caption: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        title: Optional[str] = None
    ) -> AlbumImage:
        """Update image metadata (caption, keywords, title).
        
        Args:
            image_key: Image key (unique identifier)
            caption: New caption (optional)
            keywords: New keywords list (optional)
            title: New title (optional)
            
        Returns:
            Updated AlbumImage object
            
        Raises:
            SmugMugAPIError: If update fails
        """
        logger.info(f"Updating metadata for image: {image_key}")
        
        # Build update data
        update_data = {}
        if caption is not None:
            update_data["Caption"] = caption
        if keywords is not None:
            # Convert list to comma-separated string
            update_data["Keywords"] = ", ".join(keywords) if keywords else ""
        if title is not None:
            update_data["Title"] = title
        
        if not update_data:
            logger.warning("No metadata to update")
            return self.get_image(image_key)
        
        logger.debug(f"Update data: {update_data}")
        
        endpoint = f"/image/{image_key}"
        response = self._request("PATCH", endpoint, json_data=update_data)
        
        image_data = response.get("Response", {}).get("Image", {})
        image = AlbumImage.from_api_response(image_data)
        
        logger.info(f"Successfully updated metadata for: {image.file_name}")
        return image
    
    def download_image(
        self,
        image: AlbumImage,
        destination: str,
        size: str = "Medium",
        skip_if_exists: bool = True
    ) -> Optional[Path]:
        """Download image or video to local file.
        
        Args:
            image: AlbumImage object (can also be a video)
            destination: Destination directory path
            size: Image size (Medium, Large, XLarge, X2Large, X3Large, Original, etc.)
                  Note: For videos, only Original is supported
            skip_if_exists: If True, skip download if file already exists
            
        Returns:
            Path to downloaded file, or None if skipped
            
        Raises:
            SmugMugAPIError: If download fails
        """
        dest_path = Path(destination) / image.file_name
        
        # Check if already exists
        if skip_if_exists and dest_path.exists():
            logger.debug(f"Skipping {image.file_name} (already cached)")
            return None
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get download URL
        download_url = None
        
        # For videos, use the LargestVideo endpoint to get the actual video file
        # The ArchivedUri and ImageSizes endpoints only provide thumbnail/poster images for videos
        if image.is_video:
            if image.uris and "LargestVideo" in image.uris:
                largest_video_data = image.uris["LargestVideo"]
                if isinstance(largest_video_data, dict):
                    video_uri = largest_video_data.get("Uri")
                    if video_uri:
                        try:
                            # Convert relative URI to full URL
                            if video_uri.startswith("/"):
                                video_uri = f"https://api.smugmug.com{video_uri}"
                            
                            # Fetch the video details
                            video_response = self._request("GET", video_uri)
                            response_data = video_response.get("Response", video_response)
                            video_data = response_data.get("LargestVideo", {})
                            
                            # Get the actual video URL
                            download_url = video_data.get("Url")
                            if download_url:
                                video_size_mb = video_data.get("Size", 0) / (1024 * 1024)
                                logger.info(
                                    f"Downloading video {image.file_name} "
                                    f"({video_size_mb:.1f} MB) to {dest_path}"
                                )
                        except Exception as e:
                            logger.warning(f"Could not fetch video details: {e}")
            
            if not download_url:
                raise SmugMugAPIError(
                    f"No video download URL available for {image.file_name}. "
                    f"Video may still be processing or download may not be enabled."
                )
        else:
            # For images, use the ImageSizes endpoint
            if image.uris and "ImageSizes" in image.uris:
                image_sizes_data = image.uris["ImageSizes"]
                if isinstance(image_sizes_data, dict):
                    sizes_uri = image_sizes_data.get("Uri")
                    if sizes_uri:
                        # Fetch the available sizes for this image
                        try:
                            # Convert relative URI to full URL
                            if sizes_uri.startswith("/"):
                                sizes_uri = f"https://api.smugmug.com{sizes_uri}"
                            
                            sizes_response = self._request("GET", sizes_uri)
                            # Extract ImageSizes from Response wrapper
                            response_data = sizes_response.get("Response", sizes_response)
                            sizes_data = response_data.get("ImageSizes", {})
                            
                            # Try to find the requested size
                            size_key = f"{size}ImageUrl"
                            if size_key in sizes_data:
                                download_url = sizes_data[size_key]
                            # Fall back to LargestImageUrl if requested size not available
                            elif "LargestImageUrl" in sizes_data:
                                logger.warning(f"Size '{size}' not available for {image.file_name}, using Largest")
                                download_url = sizes_data["LargestImageUrl"]
                        except Exception as e:
                            logger.warning(f"Could not fetch image sizes: {e}")
            
            # For Original size, try ArchivedUri as fallback
            if not download_url and size == "Original" and image.archived_uri:
                download_url = image.archived_uri
            
            # Last resort: try the largest available size from image metadata
            if not download_url:
                # Check if we have direct size URLs in the image data
                for size_attr in ["original_url", "largest_url", "large_url", "medium_url"]:
                    url = getattr(image, size_attr, None)
                    if url:
                        download_url = url
                        break
            
            if download_url:
                logger.info(f"Downloading {image.file_name} ({size}) to {dest_path}")
        
        if not download_url:
            media_type = "video" if image.is_video else "image"
            raise SmugMugAPIError(
                f"No download URL available for {media_type} {image.file_name}. "
                f"Try a different size or check permissions."
            )
        
        try:
            # Download media - use authenticated request since we have OAuth
            response = requests.get(
                download_url,
                auth=self.auth,
                stream=True,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Verify we got media data, not HTML
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' in content_type:
                raise SmugMugAPIError(
                    f"Received HTML instead of media for {image.file_name}. "
                    f"URL: {download_url}"
                )
            
            # Write to file
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            file_size = dest_path.stat().st_size
            logger.debug(f"Downloaded {dest_path} ({file_size} bytes)")
            return dest_path
            
        except Exception as e:
            # Clean up partial file
            if dest_path.exists():
                dest_path.unlink()
            raise SmugMugAPIError(
                f"Failed to download image {image.file_name}: {e}"
            ) from e
    
    def download_album_images(
        self,
        album_key: str,
        destination: str,
        size: str = "Medium",
        skip_if_exists: bool = True,
        skip_videos: bool = True,
        progress_callback: Optional[callable] = None
    ) -> List[Path]:
        """Download all images from an album.
        
        Args:
            album_key: Album key
            destination: Destination directory path
            size: Image size (Medium, Large, XLarge, X2Large, Original, etc.)
            skip_if_exists: If True, skip download if file already exists
            skip_videos: If True, skip video files (default: True)
            progress_callback: Optional callback function(current, total, image)
            
        Returns:
            List of downloaded file paths (excludes skipped files)
            
        Raises:
            SmugMugAPIError: If download fails
        """
        logger.info(f"Downloading images from album {album_key}")
        
        # Get album and images
        album = self.get_album(album_key)
        all_items = self.get_album_images(album_key)
        
        if not all_items:
            logger.warning(f"No images found in album {album_key}")
            return []
        
        # Filter out videos if requested
        images = all_items
        if skip_videos:
            images = [img for img in all_items if not img.is_video]
            videos_skipped = len(all_items) - len(images)
            if videos_skipped > 0:
                logger.info(f"Skipping {videos_skipped} video file(s)")
        
        if not images:
            logger.warning(f"No images to download from album {album_key} after filtering")
            return []
        
        logger.info(f"Downloading {len(images)} images from '{album.name}'")
        
        downloaded_paths = []
        skipped_count = 0
        error_count = 0
        
        for i, image in enumerate(images, 1):
            try:
                if progress_callback:
                    progress_callback(i, len(images), image)
                
                path = self.download_image(
                    image,
                    destination,
                    size,
                    skip_if_exists
                )
                
                if path:
                    downloaded_paths.append(path)
                else:
                    skipped_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to download {image.file_name}: {e}")
                error_count += 1
                # Continue with next image
        
        logger.info(
            f"Download complete: {len(downloaded_paths)} downloaded, "
            f"{skipped_count} skipped, {error_count} errors"
        )
        
        return downloaded_paths

