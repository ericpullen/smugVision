"""Preview processing service for web UI.

This service wraps the ImageProcessor to run in dry-run mode and collect
results for display in the web interface.
"""

import logging
import time
import uuid
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Generator, Any
from urllib.parse import urlparse
from pathlib import Path

from ...config import ConfigManager
from ...smugmug import SmugMugClient, AlbumImage, Album
from ...smugmug.exceptions import SmugMugError
from ...cache import CacheManager
from ...vision import VisionModelFactory
from ...vision.base import VisionModel
from ...utils.exif import extract_exif_location, reverse_geocode
from ...face.recognizer import FaceRecognizer

logger = logging.getLogger(__name__)


@dataclass
class ImagePreviewResult:
    """Result of previewing a single image."""
    image_key: str
    filename: str
    thumbnail_url: str
    web_uri: str
    status: str  # "processed", "skipped", "error"
    current_caption: Optional[str] = None
    current_keywords: List[str] = field(default_factory=list)
    proposed_caption: Optional[str] = None
    proposed_keywords: List[str] = field(default_factory=list)
    faces_detected: List[str] = field(default_factory=list)
    location: Optional[str] = None
    error: Optional[str] = None
    skip_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "image_key": self.image_key,
            "filename": self.filename,
            "thumbnail_url": self.thumbnail_url,
            "web_uri": self.web_uri,
            "status": self.status,
            "current": {
                "caption": self.current_caption,
                "keywords": self.current_keywords,
            },
            "proposed": {
                "caption": self.proposed_caption,
                "keywords": self.proposed_keywords,
            },
            "details": {
                "faces_detected": self.faces_detected,
                "location": self.location,
            },
            "error": self.error,
            "skip_reason": self.skip_reason,
        }


@dataclass
class PreviewJob:
    """Represents an active preview job."""
    job_id: str
    album_key: str
    album_name: str
    status: str  # "processing", "complete", "error"
    total_images: int
    current_image: int = 0
    current_filename: str = ""
    results: List[ImagePreviewResult] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None
    
    # Statistics
    processed_count: int = 0
    skipped_count: int = 0
    error_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "job_id": self.job_id,
            "album_key": self.album_key,
            "album_name": self.album_name,
            "status": self.status,
            "total_images": self.total_images,
            "current_image": self.current_image,
            "current_filename": self.current_filename,
            "stats": {
                "total": self.total_images,
                "processed": self.processed_count,
                "skipped": self.skipped_count,
                "errors": self.error_count,
            },
            "error": self.error,
        }


class PreviewService:
    """Service for generating preview of metadata changes."""
    
    # In-memory storage of preview jobs
    _jobs: Dict[str, PreviewJob] = {}
    
    def __init__(self, config: ConfigManager):
        """Initialize preview service.
        
        Args:
            config: smugVision configuration manager
        """
        self.config = config
        self._smugmug: Optional[SmugMugClient] = None
        self._vision: Optional[VisionModel] = None
        self._cache: Optional[CacheManager] = None
        self._face_recognizer: Optional[FaceRecognizer] = None
        self._face_recognizer_loaded: bool = False
        self._init_lock = threading.Lock()
    
    @property
    def smugmug(self) -> SmugMugClient:
        """Get or create SmugMug client."""
        if self._smugmug is None:
            self._smugmug = SmugMugClient(
                api_key=self.config.get("smugmug.api_key"),
                api_secret=self.config.get("smugmug.api_secret"),
                access_token=self.config.get("smugmug.user_token"),
                access_token_secret=self.config.get("smugmug.user_secret")
            )
        return self._smugmug
    
    @property
    def vision(self) -> VisionModel:
        """Get or create vision model."""
        if self._vision is None:
            model_name = self.config.get("vision.model", "llama3.2-vision")
            endpoint = self.config.get("vision.endpoint", "http://localhost:11434")
            self._vision = VisionModelFactory.create(
                model_name=model_name,
                endpoint=endpoint
            )
        return self._vision
    
    @property
    def cache(self) -> CacheManager:
        """Get or create cache manager."""
        if self._cache is None:
            cache_dir = self.config.get("cache.directory", "~/.smugvision/cache")
            self._cache = CacheManager(cache_dir)
        return self._cache
    
    @property
    def face_recognizer(self) -> Optional[FaceRecognizer]:
        """Get or create face recognizer (thread-safe)."""
        # Quick check without lock
        if self._face_recognizer_loaded:
            return self._face_recognizer
        
        with self._init_lock:
            # Double-check after acquiring lock
            if self._face_recognizer_loaded:
                return self._face_recognizer
            
            if self.config.get("face_recognition.enabled", True):
                try:
                    reference_faces_dir = self.config.get(
                        "face_recognition.reference_faces_dir",
                        "~/.smugvision/reference_faces"
                    )
                    reference_faces_path = Path(reference_faces_dir).expanduser()
                    
                    if reference_faces_path.exists():
                        logger.info(f"Loading face recognizer from {reference_faces_path}...")
                        self._face_recognizer = FaceRecognizer(str(reference_faces_path))
                        logger.info(f"Face recognition enabled with {len(self._face_recognizer.reference_faces)} person(s)")
                except Exception as e:
                    logger.warning(f"Could not initialize face recognizer: {e}")
            
            self._face_recognizer_loaded = True
        
        return self._face_recognizer
    
    def resolve_album_from_url(self, url: str) -> tuple:
        """Resolve album key and name from SmugMug URL.
        
        Args:
            url: SmugMug album URL
            
        Returns:
            Tuple of (album_key, album_name)
            
        Raises:
            ValueError: If URL cannot be parsed
            SmugMugError: If album cannot be found
        """
        # Parse URL to extract node ID
        node_match = re.search(r'/n-([a-zA-Z0-9]+)', url)
        if not node_match:
            raise ValueError(
                "Could not extract node ID from URL. "
                "Expected format: .../n-XXXXX/album-name"
            )
        
        node_id = node_match.group(1)
        
        # Get album name from URL
        path = urlparse(url).path
        parts = [p for p in path.split('/') if p and not p.startswith('n-')]
        if not parts:
            raise ValueError(
                "Could not extract album name from URL. "
                "Expected format: .../n-XXXXX/album-name"
            )
        
        album_identifier = parts[-1]
        
        logger.info(f"Resolving album from URL: node={node_id}, identifier={album_identifier}")
        
        # Resolve to album key
        album_key = self.smugmug.resolve_album_key(album_identifier, node_id)
        album = self.smugmug.get_album(album_key)
        
        return album_key, album.name
    
    def create_preview_job(self, url: str, force_reprocess: bool = False) -> PreviewJob:
        """Create a new preview job for an album.
        
        Args:
            url: SmugMug album URL
            force_reprocess: Whether to reprocess already-tagged images
            
        Returns:
            New PreviewJob instance
        """
        # Resolve album
        album_key, album_name = self.resolve_album_from_url(url)
        
        # Get album and count images
        album = self.smugmug.get_album(album_key)
        images = self.smugmug.get_album_images(album_key)
        
        # Filter to images only (no videos for now)
        images = [img for img in images if not img.is_video]
        
        # Create job
        job = PreviewJob(
            job_id=str(uuid.uuid4())[:8],
            album_key=album_key,
            album_name=album_name,
            status="processing",
            total_images=len(images),
        )
        
        # Store job
        self._jobs[job.job_id] = job
        
        logger.info(f"Created preview job {job.job_id} for album {album_name} ({len(images)} images)")
        
        return job
    
    def get_job(self, job_id: str) -> Optional[PreviewJob]:
        """Get a preview job by ID."""
        return self._jobs.get(job_id)
    
    def process_preview(
        self,
        job_id: str,
        force_reprocess: bool = False
    ) -> Generator[Dict[str, Any], None, None]:
        """Process album preview, yielding progress events.
        
        This is a generator that yields SSE events as processing progresses.
        
        Args:
            job_id: Preview job ID
            force_reprocess: Whether to reprocess already-tagged images
            
        Yields:
            Event dictionaries with type and data
        """
        job = self._jobs.get(job_id)
        if not job:
            yield {"event": "error", "data": {"message": f"Job {job_id} not found"}}
            return
        
        try:
            # Get album and images
            album = self.smugmug.get_album(job.album_key)
            images = self.smugmug.get_album_images(job.album_key)
            
            # Filter to images only
            images = [img for img in images if not img.is_video]
            
            marker_tag = self.config.get("processing.marker_tag", "smugvision")
            
            for i, image in enumerate(images, 1):
                job.current_image = i
                job.current_filename = image.file_name
                
                # Yield progress event
                yield {
                    "event": "progress",
                    "data": {
                        "current": i,
                        "total": job.total_images,
                        "filename": image.file_name,
                        "percent": int((i / job.total_images) * 100),
                    }
                }
                
                # Process this image
                result = self._process_single_image(
                    image=image,
                    album=album,
                    force_reprocess=force_reprocess,
                    marker_tag=marker_tag,
                )
                
                job.results.append(result)
                
                # Update stats
                if result.status == "processed":
                    job.processed_count += 1
                elif result.status == "skipped":
                    job.skipped_count += 1
                else:
                    job.error_count += 1
                
                # Yield image complete event
                yield {
                    "event": "image_complete",
                    "data": {
                        "image_key": result.image_key,
                        "filename": result.filename,
                        "status": result.status,
                    }
                }
            
            # Mark job complete
            job.status = "complete"
            
            yield {
                "event": "complete",
                "data": {
                    "processed": job.processed_count,
                    "skipped": job.skipped_count,
                    "errors": job.error_count,
                }
            }
            
        except Exception as e:
            logger.error(f"Preview processing failed: {e}", exc_info=True)
            job.status = "error"
            job.error = str(e)
            yield {"event": "error", "data": {"message": str(e)}}
    
    def _extract_thumbnail_url(self, image: AlbumImage) -> Optional[str]:
        """Extract thumbnail URL from image data if available.
        
        Args:
            image: AlbumImage with Uris data
            
        Returns:
            Thumbnail URL if found, None otherwise
        """
        if not image.uris:
            return None
        
        # Check for ImageSizes with expanded data
        if "ImageSizes" in image.uris:
            sizes_data = image.uris["ImageSizes"]
            if isinstance(sizes_data, dict):
                # Try various size keys
                for size_key in ["ThumbImageUrl", "SmallImageUrl", "TinyImageUrl"]:
                    if size_key in sizes_data:
                        return sizes_data[size_key]
        
        return None
    
    def _process_single_image(
        self,
        image: AlbumImage,
        album: Album,
        force_reprocess: bool,
        marker_tag: str,
    ) -> ImagePreviewResult:
        """Process a single image for preview.
        
        Args:
            image: Image to process
            album: Parent album
            force_reprocess: Whether to reprocess tagged images
            marker_tag: Marker tag to check for
            
        Returns:
            ImagePreviewResult
        """
        # Try to extract actual thumbnail URL from image data
        cached_thumb_url = self._extract_thumbnail_url(image)
        
        result = ImagePreviewResult(
            image_key=image.image_key,
            filename=image.file_name,
            thumbnail_url=f"/api/thumbnail/{image.image_key}",
            web_uri=image.web_uri or "",
            status="processing",
            current_caption=image.caption,
            current_keywords=list(image.keywords),
        )
        
        # Store cached URL for later use by thumbnail endpoint
        if cached_thumb_url:
            result._thumbnail_url_cached = cached_thumb_url
        
        try:
            # Check if already processed
            if not force_reprocess and image.has_marker_tag(marker_tag):
                result.status = "skipped"
                result.skip_reason = f"Already has '{marker_tag}' marker tag"
                result.proposed_caption = image.caption
                result.proposed_keywords = list(image.keywords)
                return result
            
            # Download image to cache
            album_cache_dir = self.cache.get_album_cache_dir(
                album_name=album.name,
                folder_path=None
            )
            
            size = self.config.get("processing.image_size", "Medium")
            image_path = self.smugmug.download_image(
                image=image,
                destination=str(album_cache_dir),
                size=size,
                skip_if_exists=True
            )
            
            # If download returned None, image is already cached
            if not image_path:
                image_path = album_cache_dir / image.file_name
            
            if not image_path.exists():
                raise ValueError(f"Failed to download image: {image.file_name}")
            
            # Extract EXIF location
            exif_location = extract_exif_location(str(image_path))
            location_string = None
            
            if exif_location.has_coordinates:
                if self.config.get("exif.enable_geocoding", True):
                    exif_location = reverse_geocode(exif_location, self.config)
                location_string = exif_location.location_name
            
            result.location = location_string
            
            # Detect faces
            person_names = []
            if self.face_recognizer:
                raw_names = self.face_recognizer.get_person_names(str(image_path))
                person_names = [name.replace('_', ' ') for name in raw_names]
                result.faces_detected = person_names
            
            # Generate caption
            caption_prompt = self._build_caption_prompt(location_string, person_names)
            ai_caption = self.vision.generate_caption(
                image_path=str(image_path),
                prompt=caption_prompt
            )
            
            # Generate tags
            tags_prompt = self._build_tags_prompt(location_string, person_names)
            ai_tags = self.vision.generate_tags(
                image_path=str(image_path),
                prompt=tags_prompt
            )
            
            # Format proposed metadata
            result.proposed_caption = self._format_caption(
                ai_caption=ai_caption,
                existing_caption=image.caption,
                location=location_string,
                person_names=person_names
            )
            
            result.proposed_keywords = self._format_tags(
                ai_tags=ai_tags,
                existing_tags=image.keywords,
                person_names=person_names,
                marker_tag=marker_tag
            )
            
            result.status = "processed"
            
        except Exception as e:
            logger.error(f"Error processing {image.file_name}: {e}", exc_info=True)
            result.status = "error"
            result.error = str(e)
            result.proposed_caption = image.caption
            result.proposed_keywords = list(image.keywords)
        
        return result
    
    def _build_caption_prompt(
        self,
        location: Optional[str] = None,
        person_names: Optional[List[str]] = None
    ) -> str:
        """Build caption prompt with context."""
        base_prompt = self.config.get(
            "prompts.caption",
            "Analyze this image and provide a concise, descriptive caption."
        )
        
        context_parts = []
        if location:
            context_parts.append(f"This photo was taken at: {location}")
        if person_names:
            context_parts.append(f"People in photo: {', '.join(person_names)}")
        
        if context_parts:
            return f"{base_prompt}\n\nContext:\n" + "\n".join(context_parts)
        return base_prompt
    
    def _build_tags_prompt(
        self,
        location: Optional[str] = None,
        person_names: Optional[List[str]] = None
    ) -> str:
        """Build tags prompt with context."""
        base_prompt = self.config.get(
            "prompts.tags",
            "Generate 5-10 relevant keyword tags for this image."
        )
        
        context_parts = []
        if location:
            context_parts.append(f"Location: {location}")
        if person_names:
            context_parts.append(f"People: {', '.join(person_names)}")
        
        if context_parts:
            return f"{base_prompt}\n\nContext: " + " | ".join(context_parts)
        return base_prompt
    
    def _format_caption(
        self,
        ai_caption: str,
        existing_caption: Optional[str],
        location: Optional[str],
        person_names: Optional[List[str]]
    ) -> str:
        """Format the final caption."""
        preserve_existing = self.config.get("processing.preserve_existing", True)
        
        if preserve_existing and existing_caption:
            # Append AI caption to existing
            return f"{existing_caption}\n\n{ai_caption}"
        
        return ai_caption
    
    def _format_tags(
        self,
        ai_tags: List[str],
        existing_tags: List[str],
        person_names: Optional[List[str]],
        marker_tag: str
    ) -> List[str]:
        """Format the final tags list."""
        preserve_existing = self.config.get("processing.preserve_existing", True)
        
        # Start with existing tags if preserving
        if preserve_existing:
            tags = list(existing_tags)
        else:
            tags = []
        
        # Add AI-generated tags
        for tag in ai_tags:
            tag_lower = tag.lower()
            if tag_lower not in [t.lower() for t in tags]:
                tags.append(tag)
        
        # Add person names as tags
        if person_names:
            for name in person_names:
                if name.lower() not in [t.lower() for t in tags]:
                    tags.append(name)
        
        # Add marker tag
        if marker_tag.lower() not in [t.lower() for t in tags]:
            tags.append(marker_tag)
        
        return tags
    
    def commit_changes(self, job_id: str) -> Dict[str, Any]:
        """Commit previewed changes to SmugMug.
        
        Args:
            job_id: Preview job ID
            
        Returns:
            Dictionary with commit results
        """
        job = self._jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        if job.status != "complete":
            raise ValueError(f"Job {job_id} is not complete (status: {job.status})")
        
        committed = 0
        errors = 0
        
        for result in job.results:
            if result.status != "processed":
                continue
            
            try:
                self.smugmug.update_image_metadata(
                    image_key=result.image_key,
                    caption=result.proposed_caption,
                    keywords=result.proposed_keywords
                )
                committed += 1
                logger.info(f"Committed changes for {result.filename}")
                
            except Exception as e:
                logger.error(f"Failed to commit changes for {result.filename}: {e}")
                errors += 1
        
        return {
            "status": "success" if errors == 0 else "partial",
            "committed": committed,
            "errors": errors,
        }
