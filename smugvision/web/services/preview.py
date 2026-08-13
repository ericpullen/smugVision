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
from typing import Dict, List, Optional, Generator, Any, Tuple
from urllib.parse import urlparse

from ...config import ConfigManager
from ...smugmug import SmugMugClient, AlbumImage, Album
from ...cache import CacheManager
from ...processing import HintManager, ImageProcessor
from ...face.recognizer import FaceRecognizer

logger = logging.getLogger(__name__)

# Listing one level of the SmugMug node tree costs ~0.3-0.7s and SmugMug sends
# "cache-control: no-store" on every response, so a short in-process TTL is the only
# thing that makes back-navigation in the gallery picker feel instant. The TTL is
# deliberately short - and bypassable with refresh=True - so an album created in
# SmugMug becomes visible within a couple of minutes instead of staying invisible for
# the whole life of the server process.
GALLERY_CACHE_TTL_SECONDS = 120
GALLERY_CACHE_MAX_ENTRIES = 50


class PreviewServiceError(Exception):
    """Base class for preview service failures a caller is expected to distinguish."""


class JobNotFoundError(PreviewServiceError):
    """Raised when a job ID is unknown, or the job has been evicted from memory."""


class JobNotReadyError(PreviewServiceError):
    """Raised when a job exists but its state does not allow the requested operation."""


class ImageNotInJobError(PreviewServiceError):
    """Raised when an image key is not part of a job's preview results."""


class ConfirmationRequiredError(PreviewServiceError):
    """Raised when a write to SmugMug was requested without explicit confirmation."""


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
    proposed_title: Optional[str] = None
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
                "title": self.proposed_title,
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
    # True = proof every image, including ones already carrying the marker tag. Stored on
    # the job because the image list is selected when the job is created: the streaming
    # request cannot change its mind later without the count and the loop disagreeing.
    force_reprocess: bool = False
    # Images left out of this run because they were already tagged. Not the same as
    # skipped_count, which counts images the processor skipped mid-run.
    excluded_count: int = 0
    # None = use processing.preserve_existing from config; False = replace the existing
    # caption and keywords instead of merging into them, for this job only.
    preserve_existing: Optional[bool] = None
    # None = use processing.generate_titles from config.
    generate_titles: Optional[bool] = None
    
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
                "excluded": self.excluded_count,
            },
            "error": self.error,
        }


class PreviewService:
    """Service for generating preview of metadata changes.

    This service uses the main ImageProcessor in dry-run mode to ensure
    100% consistent behavior between CLI and Web UI. All processing logic
    is delegated to the core library.

    It orchestrates and adapts only: album browsing comes from
    :class:`~smugvision.smugmug.client.SmugMugClient`, captions and tags come from
    :class:`~smugvision.processing.processor.ImageProcessor`, and hints come from
    :class:`~smugvision.processing.hints.HintManager`. The single write path,
    :meth:`commit_changes`, refuses to run without explicit confirmation.
    """

    def __init__(self, config: ConfigManager):
        """Initialize preview service.

        Args:
            config: smugVision configuration manager
        """
        self.config = config
        self._processor: Optional[ImageProcessor] = None
        self._processor_loaded: bool = False
        self._init_lock = threading.Lock()

        # Standalone hint manager, used only until the processor exists so that the
        # hint editor does not have to pay for loading the vision model and face
        # encodings (see the `hints` property).
        self._hints: Optional[HintManager] = None
        self._hints_lock = threading.Lock()

        # Short-lived cache of node listings for the gallery picker:
        # {node_id or "": (monotonic timestamp, listing payload)}
        self._gallery_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._gallery_lock = threading.Lock()

        # Instance-level job storage (not class-level to avoid memory leak)
        # Only keep the most recent jobs to limit memory usage
        self._jobs: Dict[str, PreviewJob] = {}
        self._max_jobs = 5  # Keep at most 5 jobs in memory

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

    def processor_for(
        self,
        preserve_existing: Optional[bool] = None,
        generate_titles: Optional[bool] = None,
    ) -> ImageProcessor:
        """Return the processor matching one job's merge/replace mode.

        Both flags are constructor arguments, so each distinct combination needs its own
        ImageProcessor rather than mutating the shared one - mutating it would change the
        mode under any job running concurrently. The second instance INJECTS the first's
        collaborators, which is what they are injectable for: building it fresh would
        re-encode every reference face (~5s and a second copy in memory) and open a
        second Ollama client for no reason.

        Args:
            preserve_existing: ``None`` for the configured behaviour, ``False`` to
                replace existing caption and keywords
            generate_titles: ``None`` for the configured behaviour, or an explicit
                override for this job

        Returns:
            An ImageProcessor in dry_run mode with the requested merge behaviour
        """
        base = self.processor
        wanted = (
            base.preserve_existing if preserve_existing is None else bool(preserve_existing),
            base.generate_titles if generate_titles is None else bool(generate_titles),
        )
        if wanted == (base.preserve_existing, base.generate_titles):
            return base

        with self._init_lock:
            cache = getattr(self, "_variant_processors", None)
            if cache is None:
                cache = {}
                self._variant_processors = cache
            if wanted in cache:
                return cache[wanted]

            logger.info(
                f"Building an ImageProcessor variant preserve_existing={wanted[0]}, "
                f"generate_titles={wanted[1]} (sharing vision model, faces, cache, hints)"
            )
            processor = ImageProcessor(
                config=self.config,
                smugmug_client=base.smugmug,
                vision_model=base.vision,
                cache_manager=base.cache,
                face_recognizer=base.face_recognizer,
                hint_manager=base.hints,
                dry_run=True,
                preserve_existing=wanted[0],
                generate_titles=wanted[1],
            )
            cache[wanted] = processor
            return processor
    
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

    @property
    def hints(self) -> Optional[HintManager]:
        """Get the hint manager, or None when hints are disabled or unavailable.

        Once the processor exists this returns the processor's own manager, so the
        editor and the pipeline share one in-memory copy. Before that it creates a
        standalone manager over the same ``hints.file`` - reading or editing hints must
        not force the vision model and face encodings to load. The two instances stay
        consistent because :class:`~smugvision.processing.hints.HintManager` reloads on
        an mtime change, and only this service-side instance ever writes.

        Returns:
            HintManager instance, or None if ``hints.enabled`` is false or the manager
            could not be created
        """
        if not self.config.get("hints.enabled", True):
            return None

        if self._processor_loaded and self._processor is not None:
            return self._processor.hints

        with self._hints_lock:
            if self._hints is None:
                try:
                    self._hints = HintManager(self.config.get("hints.file"))
                except Exception as e:
                    # A hint is an optimization, never a reason to fail a request.
                    logger.warning(f"Could not initialize hint manager: {e}")
                    return None
            return self._hints

    def list_galleries(
        self,
        node_id: Optional[str] = None,
        refresh: bool = False
    ) -> Dict[str, Any]:
        """List one level of the SmugMug folder tree for the gallery picker.

        Delegates to :meth:`SmugMugClient.list_node_children`, which never descends, so
        the cost is independent of account size. Results are cached in-process for
        ``GALLERY_CACHE_TTL_SECONDS``.

        Args:
            node_id: Node to list. None starts at the authenticated user's root node.
            refresh: If True, bypass the cache and re-fetch from SmugMug

        Returns:
            ``NodeListing.to_dict()`` (node, breadcrumb, folders, albums, total,
            partial) plus ``cached`` and ``cache_ttl_seconds``

        Raises:
            SmugMugNotFoundError: If the node does not exist
            SmugMugAPIError: If node_id refers to an album, or the request fails
        """
        cache_key = node_id or ""
        now = time.monotonic()

        if not refresh:
            with self._gallery_lock:
                entry = self._gallery_cache.get(cache_key)
            if entry is not None and (now - entry[0]) < GALLERY_CACHE_TTL_SECONDS:
                logger.debug(f"Serving cached gallery listing for node {cache_key or '<root>'}")
                return self._gallery_payload(entry[1], cached=True)

        listing = self.smugmug.list_node_children(node_id)
        payload = listing.to_dict()

        with self._gallery_lock:
            self._gallery_cache[cache_key] = (now, payload)
            self._trim_gallery_cache()

        return self._gallery_payload(payload, cached=False)

    @staticmethod
    def _gallery_payload(payload: Dict[str, Any], cached: bool) -> Dict[str, Any]:
        """Annotate a cached listing payload without mutating the cached copy.

        Args:
            payload: Listing dictionary as returned by ``NodeListing.to_dict()``
            cached: Whether this response came from the in-process cache

        Returns:
            Shallow copy of the payload with cache metadata added
        """
        annotated = dict(payload)
        annotated["cached"] = cached
        annotated["cache_ttl_seconds"] = GALLERY_CACHE_TTL_SECONDS
        return annotated

    def _trim_gallery_cache(self) -> None:
        """Drop the oldest gallery cache entries once the cache exceeds its cap.

        Must be called while holding ``self._gallery_lock``.
        """
        excess = len(self._gallery_cache) - GALLERY_CACHE_MAX_ENTRIES
        if excess <= 0:
            return

        oldest = sorted(self._gallery_cache.items(), key=lambda item: item[1][0])[:excess]
        for key, _ in oldest:
            self._gallery_cache.pop(key, None)
        logger.debug(f"Trimmed {len(oldest)} gallery cache entry(ies)")

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
    
    def images_to_proof(
        self, album_key: str, force_reprocess: bool
    ) -> Tuple[List[AlbumImage], int]:
        """Select the images a proof run should actually work on.

        Videos are never proofed. Unless ``force_reprocess`` is set, images that already
        carry the marker tag are left out entirely rather than being handed to the
        processor and coming back as skips - proofing an album a second time should
        present only the frames that still need attention.

        Both the count reported when a job is created and the loop that processes it go
        through here, so they can never disagree about what is in the run.

        Args:
            album_key: SmugMug album key
            force_reprocess: If True, keep already-tagged images in the run

        Returns:
            Tuple of (images to proof, number of already-tagged images excluded)

        Raises:
            SmugMugError: If the album's images cannot be read
        """
        images = [img for img in self.smugmug.get_album_images(album_key) if not img.is_video]

        if force_reprocess:
            return images, 0

        pending = [img for img in images if self.processor.needs_processing(img)]
        return pending, len(images) - len(pending)

    def create_preview_job(
        self,
        url: Optional[str] = None,
        force_reprocess: bool = False,
        album_key: Optional[str] = None,
        preserve_existing: Optional[bool] = None,
        generate_titles: Optional[bool] = None
    ) -> PreviewJob:
        """Create a new preview job for an album.

        The album is identified either by its SmugMug URL (what the paste-a-link box
        sends) or directly by its album key (what the gallery picker sends). Exactly one
        of the two is required.

        Args:
            url: SmugMug album URL
            force_reprocess: Whether to include images that already carry the marker
                tag. When False they are left out of the job entirely, so
                ``total_images`` counts only the frames that still need proofing.
            album_key: SmugMug album key, as an alternative to ``url``

        Returns:
            New PreviewJob instance

        Raises:
            ValueError: If neither or both of ``url`` and ``album_key`` were given, or
                the URL cannot be parsed
            SmugMugNotFoundError: If the album does not exist
            SmugMugError: If the album cannot be read
        """
        clean_url = (url or "").strip()
        resolved_key = (album_key or "").strip()

        if clean_url and resolved_key:
            raise ValueError("Provide either 'url' or 'album_key', not both")
        if not clean_url and not resolved_key:
            raise ValueError("Provide either 'url' or 'album_key'")

        # Resolve album
        if clean_url:
            resolved_key, _ = self.resolve_album_from_url(clean_url)

        # Get album and select the images this run will work on
        album = self.smugmug.get_album(resolved_key)
        album_name = album.name
        images, excluded = self.images_to_proof(resolved_key, force_reprocess)

        # Clean up old jobs to limit memory usage
        self._cleanup_old_jobs()

        # Create job
        job = PreviewJob(
            job_id=str(uuid.uuid4())[:8],
            album_key=resolved_key,
            album_name=album_name,
            status="processing",
            total_images=len(images),
            preserve_existing=preserve_existing,
            generate_titles=generate_titles,
            force_reprocess=force_reprocess,
            excluded_count=excluded,
        )

        # Store job
        self._jobs[job.job_id] = job

        logger.info(
            f"Created preview job {job.job_id} for album {album_name} "
            f"({len(images)} to proof, {excluded} already tagged and left out)"
        )

        return job
    
    def get_job(self, job_id: str) -> Optional[PreviewJob]:
        """Get a preview job by ID."""
        return self._jobs.get(job_id)
    
    def _cleanup_old_jobs(self) -> None:
        """Remove old jobs to limit memory usage.
        
        Keeps the most recent jobs up to _max_jobs limit.
        """
        if len(self._jobs) >= self._max_jobs:
            # Sort jobs by creation time, remove oldest
            sorted_jobs = sorted(
                self._jobs.items(),
                key=lambda x: x[1].created_at,
                reverse=True
            )
            
            # Keep only the newest jobs
            jobs_to_keep = dict(sorted_jobs[:self._max_jobs - 1])
            removed_count = len(self._jobs) - len(jobs_to_keep)
            
            # Clear results from removed jobs to free memory
            for job_id, job in self._jobs.items():
                if job_id not in jobs_to_keep:
                    job.results.clear()
            
            self._jobs = jobs_to_keep
            logger.debug(f"Cleaned up {removed_count} old preview jobs")
    
    def process_preview(self, job_id: str) -> Generator[Dict[str, Any], None, None]:
        """Process album preview, yielding progress events.

        This uses the main ImageProcessor to ensure 100% consistent behavior
        with the CLI tool. All processing is delegated to the core library.

        Whether already-tagged images are included was decided when the job was
        created, and is read back off the job here. It deliberately is not a
        parameter: the image list and ``total_images`` were fixed at creation time, so
        a caller passing a different value could only make the progress counter lie.

        Args:
            job_id: Preview job ID

        Yields:
            Event dictionaries with type and data
        """
        job = self._jobs.get(job_id)
        if not job:
            yield {"event": "error", "data": {"message": f"Job {job_id} not found"}}
            return

        try:
            # Get album and the same image selection the job was sized from
            album = self.smugmug.get_album(job.album_key)
            images, _ = self.images_to_proof(job.album_key, job.force_reprocess)

            if not images:
                job.status = "complete"
                logger.info(
                    f"Preview job {job_id} had nothing to proof "
                    f"({job.excluded_count} already tagged)"
                )
                yield {
                    "event": "complete",
                    "data": {
                        "processed": 0,
                        "skipped": 0,
                        "errors": 0,
                        "excluded": job.excluded_count,
                    },
                }
                return

            marker_tag = self.config.get("processing.marker_tag", "smugvision")

            # The album can change between creating the job and streaming it, so size the
            # progress bar from what is actually being processed now.
            total = len(images)
            job.total_images = total

            for i, image in enumerate(images, 1):
                job.current_image = i
                job.current_filename = image.file_name

                # Yield progress event
                yield {
                    "event": "progress",
                    "data": {
                        "current": i,
                        "total": total,
                        "filename": image.file_name,
                        "percent": int((i / total) * 100),
                    }
                }

                # Process using the main ImageProcessor (dry_run=True)
                # This ensures 100% identical behavior to CLI
                processor = self.processor_for(job.preserve_existing, job.generate_titles)
                proc_result = processor.process_image(
                    image=image,
                    album=album,
                    force_reprocess=job.force_reprocess
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
            
            logger.info(
                f"Preview job {job_id} complete: {job.processed_count} processed, "
                f"{job.skipped_count} skipped, {job.error_count} errors, "
                f"{job.excluded_count} already tagged and left out"
            )

            yield {
                "event": "complete",
                "data": {
                    "processed": job.processed_count,
                    "skipped": job.skipped_count,
                    "errors": job.error_count,
                    "excluded": job.excluded_count,
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
            proposed_title=proc_result.proposed_title if proc_result.success else None,
            # Details from ImageProcessor
            faces_detected=proc_result.detected_faces or [],
            location=proc_result.location,
            # Error/skip info
            error=proc_result.error,
            skip_reason=f"Already has '{marker_tag}' marker tag" if proc_result.skipped else None,
        )
    
    def regenerate_image(
        self,
        job_id: str,
        image_key: str,
        force_reprocess: bool = True
    ) -> ImagePreviewResult:
        """Re-run a single image of a finished job through the same ImageProcessor.

        This is the edit-a-hint-then-try-again loop: it re-processes exactly one image
        and replaces that one entry in the job, leaving every other result untouched.
        Hints are resolved inside ``ImageProcessor.process_image`` and HintManager
        reloads on an mtime change, so a hint saved a moment ago is applied here without
        any cache busting.

        Still dry-run: the processor is the preview processor, so nothing is written to
        SmugMug. ``force_reprocess`` defaults to True because clicking "regenerate" is
        an explicit request for a new result, even for an image that carries the marker
        tag and was skipped by the album run.

        Args:
            job_id: Preview job ID
            image_key: SmugMug image key of the image to re-run
            force_reprocess: If False, an image carrying the marker tag is skipped again

        Returns:
            The replacement ImagePreviewResult (which may itself have status "error" if
            the vision model failed)

        Raises:
            JobNotFoundError: If the job ID is unknown or has been evicted
            JobNotReadyError: If the job is still processing
            ImageNotInJobError: If the image is not part of the job, or has since been
                removed from the album
            SmugMugError: If the album or image could not be re-read
        """
        job = self._jobs.get(job_id)
        if not job:
            raise JobNotFoundError(f"Job {job_id} not found")

        if job.status == "processing":
            raise JobNotReadyError(
                f"Job {job_id} is still processing; wait for it to finish before "
                "regenerating a single image"
            )

        image_key = (image_key or "").strip()
        index = next(
            (i for i, result in enumerate(job.results) if result.image_key == image_key),
            None
        )
        if index is None:
            raise ImageNotInJobError(f"Job {job_id} has no preview result for image {image_key}")

        # Re-read album and image so the "current" metadata shown after a regenerate is
        # the live state, not a snapshot from when the album run started.
        album = self.smugmug.get_album(job.album_key)
        image = next(
            (
                candidate
                for candidate in self.smugmug.get_album_images(job.album_key)
                if candidate.image_key == image_key
            ),
            None
        )
        if image is None:
            raise ImageNotInJobError(
                f"Image {image_key} is no longer in album {job.album_key}"
            )

        logger.info(f"Regenerating {image.file_name} ({image_key}) for job {job_id}")

        proc_result = self.processor_for(job.preserve_existing, job.generate_titles).process_image(
            image=image,
            album=album,
            force_reprocess=force_reprocess
        )

        marker_tag = self.config.get("processing.marker_tag", "smugvision")
        preview_result = self._convert_to_preview_result(image, proc_result, marker_tag)

        # Replace just this entry, then recount so the job stats cannot drift.
        job.results[index] = preview_result
        self._recount_job(job)

        return preview_result

    @staticmethod
    def _recount_job(job: PreviewJob) -> None:
        """Recompute a job's statistics from its current results.

        Args:
            job: Job whose processed/skipped/error counters should be rebuilt
        """
        job.processed_count = sum(1 for r in job.results if r.status == "processed")
        job.skipped_count = sum(1 for r in job.results if r.status == "skipped")
        job.error_count = sum(1 for r in job.results if r.status not in ("processed", "skipped"))

    def commit_changes(
        self,
        job_id: str,
        confirm: bool = False,
        image_keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Commit previewed changes to SmugMug.

        This is the ONLY method in the web layer that writes to the user's real SmugMug
        account, so it is deliberately hard to trigger: without ``confirm=True`` it
        raises before touching the network.

        Args:
            job_id: Preview job ID
            confirm: Must be True. Anything else refuses the write.
            image_keys: Optional subset of image keys to commit. None commits every
                successfully processed image in the job. Order and duplicates in the
                caller's list do not matter; unknown keys are an error rather than a
                silent no-op.

        Returns:
            Dictionary with commit results: ``status``, ``committed``, ``errors`` plus
            the ``committed_keys`` / ``skipped_keys`` / ``failed_keys`` breakdown

        Raises:
            JobNotFoundError: If the job ID is unknown or has been evicted
            ConfirmationRequiredError: If ``confirm`` is not True (nothing is sent)
            JobNotReadyError: If the job has not finished
            ValueError: If ``image_keys`` is malformed or names an image the job does
                not have a preview result for
        """
        job = self._jobs.get(job_id)
        if not job:
            raise JobNotFoundError(f"Job {job_id} not found")

        if confirm is not True:
            # Checked before anything else that could write. Nothing has been sent to
            # SmugMug at this point, and nothing will be.
            raise ConfirmationRequiredError(
                "Refusing to write to SmugMug without explicit confirmation: "
                'send {"confirm": true} to commit these changes'
            )

        if job.status != "complete":
            raise JobNotReadyError(f"Job {job_id} is not complete (status: {job.status})")

        selected = self._select_commit_targets(job, image_keys)

        committed = 0
        errors = 0
        committed_keys: List[str] = []
        skipped_keys: List[str] = []
        failed_keys: List[str] = []

        logger.warning(
            f"COMMIT confirmed for job {job_id}: writing {len(selected)} selected "
            f"result(s) to SmugMug"
        )

        for result in selected:
            if result.status != "processed":
                skipped_keys.append(result.image_key)
                continue

            if not result.proposed_caption and not result.proposed_keywords:
                # Both fields would be sent as None (see the `or None` guards below), so
                # there is nothing to write. Reporting it as skipped keeps the committed
                # count honest instead of claiming a write that never happened.
                logger.info(f"Nothing to commit for {result.filename}: empty proposal")
                skipped_keys.append(result.image_key)
                continue

            try:
                # Mirror the CLI write path (ImageProcessor.process_image): passing None
                # leaves a field untouched, so an empty proposed value must never be sent
                # as "" - that would wipe the user's existing caption/keywords.
                self.smugmug.update_image_metadata(
                    image_key=result.image_key,
                    caption=result.proposed_caption or None,
                    keywords=result.proposed_keywords or None,
                    title=result.proposed_title or None,
                )
                committed += 1
                committed_keys.append(result.image_key)
                logger.info(f"Committed changes for {result.filename}")

            except Exception as e:
                logger.error(f"Failed to commit changes for {result.filename}: {e}")
                errors += 1
                failed_keys.append(result.image_key)

        return {
            "status": "success" if errors == 0 else "partial",
            "committed": committed,
            "errors": errors,
            "committed_keys": committed_keys,
            "skipped_keys": skipped_keys,
            "failed_keys": failed_keys,
            "requested": len(selected),
        }

    @staticmethod
    def _select_commit_targets(
        job: PreviewJob,
        image_keys: Optional[List[str]]
    ) -> List[ImagePreviewResult]:
        """Resolve which of a job's results a commit request refers to.

        Args:
            job: Job being committed
            image_keys: Optional subset of image keys, or None for the whole job

        Returns:
            The results to attempt, in job order (or caller order for a subset)

        Raises:
            ValueError: If ``image_keys`` is not a list, is empty once cleaned, or names
                an image the job has no preview result for
        """
        if image_keys is None:
            return list(job.results)

        if not isinstance(image_keys, (list, tuple)):
            raise ValueError("'image_keys' must be a list of SmugMug image keys")

        requested = [str(key).strip() for key in image_keys if str(key).strip()]
        if not requested:
            raise ValueError("'image_keys' was provided but contained no usable image keys")

        by_key = {result.image_key: result for result in job.results}
        unknown = [key for key in requested if key not in by_key]
        if unknown:
            # Refusing beats silently committing fewer images than the user selected.
            raise ValueError(
                f"Job {job.job_id} has no preview result for: {', '.join(unknown)}"
            )

        selected: List[ImagePreviewResult] = []
        seen = set()
        for key in requested:
            if key not in seen:
                seen.add(key)
                selected.append(by_key[key])
        return selected
