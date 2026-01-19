"""Preview processing service for web UI.

This service wraps the ImageProcessor to run in dry-run mode and collect
results for display in the web interface.
"""

import logging
import uuid
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Generator, Any
from urllib.parse import urlparse

from ...config import ConfigManager
from ...smugmug import SmugMugClient, AlbumImage, Album
from ...cache import CacheManager
from ...processing import ImageProcessor
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
    """Service for generating preview of metadata changes.
    
    This service uses the main ImageProcessor in dry-run mode to ensure
    100% consistent behavior between CLI and Web UI. All processing logic
    is delegated to the core library.
    """
    
    # In-memory storage of preview jobs
    _jobs: Dict[str, PreviewJob] = {}
    
    def __init__(self, config: ConfigManager):
        """Initialize preview service.
        
        Args:
            config: smugVision configuration manager
        """
        self.config = config
        self._processor: Optional[ImageProcessor] = None
        self._processor_loaded: bool = False
        self._init_lock = threading.Lock()
    
    @property
    def processor(self) -> ImageProcessor:
        """Get or create the ImageProcessor (thread-safe, lazy initialization).
        
        The processor is always in dry_run mode for preview operations.
        """
        if self._processor_loaded:
            return self._processor
        
        with self._init_lock:
            if self._processor_loaded:
                return self._processor
            
            logger.info("Initializing ImageProcessor for preview service...")
            self._processor = ImageProcessor(
                config=self.config,
                dry_run=True  # Always dry-run for preview
            )
            self._processor_loaded = True
            logger.info("ImageProcessor initialized")
        
        return self._processor
    
    @property
    def smugmug(self) -> SmugMugClient:
        """Get SmugMug client from processor."""
        return self.processor.smugmug
    
    @property
    def cache(self) -> CacheManager:
        """Get cache manager from processor."""
        return self.processor.cache
    
    @property
    def face_recognizer(self) -> Optional[FaceRecognizer]:
        """Get face recognizer from processor."""
        return self.processor.face_recognizer
    
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
        
        This uses the main ImageProcessor to ensure 100% consistent behavior
        with the CLI tool. All processing is delegated to the core library.
        
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
                
                # Process using the main ImageProcessor (dry_run=True)
                # This ensures 100% identical behavior to CLI
                proc_result = self.processor.process_image(
                    image=image,
                    album=album,
                    force_reprocess=force_reprocess
                )
                
                # Convert ProcessingResult to ImagePreviewResult for UI
                preview_result = self._convert_to_preview_result(image, proc_result, marker_tag)
                job.results.append(preview_result)
                
                # Update stats
                if preview_result.status == "processed":
                    job.processed_count += 1
                elif preview_result.status == "skipped":
                    job.skipped_count += 1
                else:
                    job.error_count += 1
                
                # Yield image complete event
                yield {
                    "event": "image_complete",
                    "data": {
                        "image_key": preview_result.image_key,
                        "filename": preview_result.filename,
                        "status": preview_result.status,
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
    
    def _convert_to_preview_result(
        self,
        image: AlbumImage,
        proc_result,
        marker_tag: str
    ) -> ImagePreviewResult:
        """Convert a ProcessingResult from ImageProcessor to ImagePreviewResult for UI.
        
        This simply extracts the data from the ProcessingResult - no additional
        processing is done here to ensure consistency.
        
        Args:
            image: Original AlbumImage
            proc_result: ProcessingResult from ImageProcessor
            marker_tag: Marker tag for skip reason
            
        Returns:
            ImagePreviewResult for UI display
        """
        # Determine status
        if proc_result.skipped:
            status = "skipped"
        elif proc_result.error:
            status = "error"
        elif proc_result.success:
            status = "processed"
        else:
            status = "error"
        
        return ImagePreviewResult(
            image_key=proc_result.image_key,
            filename=proc_result.filename,
            thumbnail_url=f"/api/thumbnail/{proc_result.image_key}",
            web_uri=image.web_uri or "",
            status=status,
            # Current metadata from the original image
            current_caption=proc_result.current_caption,
            current_keywords=proc_result.current_keywords or [],
            # Proposed metadata from ImageProcessor
            proposed_caption=proc_result.proposed_caption if proc_result.success else proc_result.current_caption,
            proposed_keywords=proc_result.proposed_keywords if proc_result.success else (proc_result.current_keywords or []),
            # Details from ImageProcessor
            faces_detected=proc_result.detected_faces or [],
            location=proc_result.location,
            # Error/skip info
            error=proc_result.error,
            skip_reason=f"Already has '{marker_tag}' marker tag" if proc_result.skipped else None,
        )
    
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
