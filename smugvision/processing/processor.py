"""Main image processing orchestrator."""

from typing import List, Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass
import logging
import time

from ..config import ConfigManager
from ..smugmug import SmugMugClient, AlbumImage, Album
from ..smugmug.exceptions import SmugMugError
from ..cache import CacheManager
from ..vision import VisionModelFactory
from ..vision.base import VisionModel
from ..utils.exif import extract_exif_location, reverse_geocode, resolve_location_with_custom
from ..face.recognizer import FaceRecognizer
from .metadata import MetadataFormatter

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of processing a single image.
    
    Attributes:
        image_key: SmugMug image key
        filename: Image filename
        success: Whether processing succeeded
        skipped: Whether image was skipped (already processed)
        caption_generated: Whether caption was generated
        tags_generated: Number of tags generated
        faces_detected: Number of faces detected
        processing_time: Time taken to process (seconds)
        error: Error message if failed
        
        # Detailed results for UI/inspection
        current_caption: Original caption before processing
        current_keywords: Original keywords before processing
        proposed_caption: Generated caption (what was/would be written)
        proposed_keywords: Generated keywords (what was/would be written)
        detected_faces: List of person names detected
        location: Resolved location string
        location_aliases: Location aliases for tags
    """
    image_key: str
    filename: str
    success: bool
    skipped: bool = False
    caption_generated: bool = False
    tags_generated: int = 0
    faces_detected: int = 0
    processing_time: float = 0.0
    error: Optional[str] = None
    
    # Detailed results for UI/inspection
    current_caption: Optional[str] = None
    current_keywords: Optional[List[str]] = None
    proposed_caption: Optional[str] = None
    proposed_keywords: Optional[List[str]] = None
    detected_faces: Optional[List[str]] = None
    location: Optional[str] = None
    location_aliases: Optional[List[str]] = None


@dataclass
class BatchProcessingStats:
    """Statistics for batch processing.
    
    Attributes:
        total_images: Total number of images in album
        processed: Number successfully processed
        skipped: Number skipped (already processed)
        errors: Number that failed
        total_time: Total processing time (seconds)
        results: List of individual processing results
    """
    total_images: int
    processed: int = 0
    skipped: int = 0
    errors: int = 0
    total_time: float = 0.0
    results: List[ProcessingResult] = None
    
    def __post_init__(self):
        if self.results is None:
            self.results = []


class ImageProcessor:
    """Orchestrates the complete image processing pipeline.
    
    This class coordinates:
    - Image download and caching
    - EXIF data extraction
    - Face detection and recognition
    - AI caption and tag generation
    - Metadata formatting
    - SmugMug updates
    """
    
    def __init__(
        self,
        config: ConfigManager,
        smugmug_client: Optional[SmugMugClient] = None,
        vision_model: Optional[VisionModel] = None,
        cache_manager: Optional[CacheManager] = None,
        face_recognizer: Optional[FaceRecognizer] = None,
        dry_run: bool = False
    ) -> None:
        """Initialize image processor.
        
        Args:
            config: Configuration manager
            smugmug_client: SmugMug API client (created if not provided)
            vision_model: Vision model for caption/tag generation (created if not provided)
            cache_manager: Cache manager (created if not provided)
            face_recognizer: Face recognizer (created if not provided)
            dry_run: If True, don't update SmugMug
        """
        self.config = config
        self.dry_run = dry_run
        
        # Initialize SmugMug client
        if smugmug_client:
            self.smugmug = smugmug_client
        else:
            self.smugmug = SmugMugClient(
                api_key=config.get("smugmug.api_key"),
                api_secret=config.get("smugmug.api_secret"),
                access_token=config.get("smugmug.user_token"),
                access_token_secret=config.get("smugmug.user_secret")
            )
        
        # Initialize cache manager
        if cache_manager:
            self.cache = cache_manager
        else:
            cache_dir = config.get("cache.directory", "~/.smugvision/cache")
            self.cache = CacheManager(cache_dir)
        
        # Initialize vision model
        if vision_model:
            self.vision = vision_model
        else:
            model_name = config.get("vision.model", "llama3.2-vision")
            endpoint = config.get("vision.endpoint", "http://localhost:11434")
            self.vision = VisionModelFactory.create(
                model_name=model_name,
                endpoint=endpoint
            )
        
        # Initialize face recognizer if enabled
        self.face_recognizer = None
        if config.get("face_recognition.enabled", True):
            if face_recognizer:
                self.face_recognizer = face_recognizer
            else:
                try:
                    from pathlib import Path as PathLib
                    reference_faces_dir = config.get(
                        "face_recognition.reference_faces_dir",
                        "~/.smugvision/reference_faces"
                    )
                    # Expand ~ and convert to absolute path
                    reference_faces_path = PathLib(reference_faces_dir).expanduser()
                    
                    # Get cache settings
                    use_cache = config.get("face_recognition.use_cache", True)
                    cache_dir = config.get(
                        "face_recognition.cache_dir",
                        "~/.smugvision/cache/face_encodings"
                    )
                    cache_dir_path = PathLib(cache_dir).expanduser()
                    
                    # Only initialize if directory exists
                    if reference_faces_path.exists():
                        self.face_recognizer = FaceRecognizer(
                            str(reference_faces_path),
                            cache_dir=str(cache_dir_path),
                            use_cache=use_cache
                        )
                        logger.info(f"Face recognition enabled with {len(self.face_recognizer.reference_faces)} person(s)")
                    else:
                        logger.info(f"Face recognition disabled: reference faces directory not found at {reference_faces_path}")
                except Exception as e:
                    logger.warning(f"Could not initialize face recognizer: {e}")
        
        # Initialize metadata formatter
        self.formatter = MetadataFormatter(
            preserve_existing=config.get("processing.preserve_existing", True),
            marker_tag=config.get("processing.marker_tag", "smugvision")
        )
        
        logger.info(
            f"ImageProcessor initialized: model={self.vision.model_name}, "
            f"dry_run={dry_run}"
        )
    
    def process_album(
        self,
        album_key: str,
        force_reprocess: bool = False,
        skip_videos: bool = True
    ) -> BatchProcessingStats:
        """Process all images in an album.
        
        Args:
            album_key: SmugMug album key
            force_reprocess: If True, reprocess images with marker tag
            skip_videos: If True, skip video files
            
        Returns:
            BatchProcessingStats with results
            
        Raises:
            SmugMugError: If album cannot be accessed
        """
        logger.info(f"Starting album processing: {album_key}")
        start_time = time.time()
        
        # Get album info
        album = self.smugmug.get_album(album_key)
        logger.info(f"Processing album: {album.name} ({album.image_count} items)")
        
        # Get all images
        all_items = self.smugmug.get_album_images(album_key)
        
        # Filter videos if requested
        if skip_videos:
            images = [img for img in all_items if not img.is_video]
            videos_skipped = len(all_items) - len(images)
            if videos_skipped > 0:
                logger.info(f"Skipping {videos_skipped} video file(s)")
        else:
            images = all_items
        
        # Initialize stats
        stats = BatchProcessingStats(total_images=len(images))
        
        if not images:
            logger.warning("No images to process in album")
            return stats
        
        # Process each image
        for i, image in enumerate(images, 1):
            logger.info(f"[{i}/{len(images)}] Processing: {image.file_name}")
            
            result = self.process_image(
                image=image,
                album=album,
                force_reprocess=force_reprocess
            )
            
            stats.results.append(result)
            
            if result.success:
                stats.processed += 1
            elif result.skipped:
                stats.skipped += 1
            else:
                stats.errors += 1
            
            # Log progress
            logger.info(
                f"  Result: {'✓ Success' if result.success else '○ Skipped' if result.skipped else '✗ Error'} "
                f"({result.processing_time:.1f}s)"
            )
        
        stats.total_time = time.time() - start_time
        
        logger.info(
            f"Album processing complete: {stats.processed} processed, "
            f"{stats.skipped} skipped, {stats.errors} errors "
            f"(Total time: {stats.total_time:.1f}s)"
        )
        
        return stats
    
    def process_image(
        self,
        image: AlbumImage,
        album: Album,
        force_reprocess: bool = False
    ) -> ProcessingResult:
        """Process a single image.
        
        Args:
            image: AlbumImage to process
            album: Parent album
            force_reprocess: If True, process even if already marked
            
        Returns:
            ProcessingResult
        """
        start_time = time.time()
        result = ProcessingResult(
            image_key=image.image_key,
            filename=image.file_name,
            success=False,
            current_caption=image.caption,
            current_keywords=list(image.keywords) if image.keywords else [],
        )
        
        try:
            # Check if already processed
            marker_tag = self.config.get("processing.marker_tag", "smugvision")
            if not force_reprocess and image.has_marker_tag(marker_tag):
                logger.debug(f"Image {image.file_name} already has marker tag, skipping")
                result.skipped = True
                result.processing_time = time.time() - start_time
                return result
            
            # Download image to cache
            logger.debug(f"Downloading image: {image.file_name}")
            image_path = self._download_image(image, album)
            
            # If download returned None, image is already cached - build path
            if not image_path:
                album_cache_dir = self.cache.get_album_cache_dir(
                    album_name=album.name,
                    folder_path=None
                )
                image_path = album_cache_dir / image.file_name
            
            if not image_path.exists():
                raise ValueError(f"Failed to download image: {image.file_name}")
            
            # Get GPS coordinates - prefer SmugMug API data over EXIF from downloaded file
            # (SmugMug strips GPS from downloaded images for privacy, but provides it via API)
            latitude = None
            longitude = None
            gps_source = None
            exif_location = None
            
            if image.has_gps:
                # Use GPS data from SmugMug API
                latitude = image.latitude
                longitude = image.longitude
                gps_source = "SmugMug API"
                logger.debug(f"  GPS from SmugMug API: {latitude:.6f}, {longitude:.6f}")
            else:
                # Fall back to EXIF extraction from downloaded file
                logger.debug("Extracting EXIF data from downloaded file")
                exif_location = extract_exif_location(str(image_path))
                if exif_location.has_coordinates:
                    latitude = exif_location.latitude
                    longitude = exif_location.longitude
                    gps_source = "EXIF"
                    logger.debug(f"  GPS from EXIF: {latitude:.6f}, {longitude:.6f}")
            
            # Get location string with custom locations and reverse geocoding
            location_string = None
            location_aliases = []
            is_custom = False
            
            if latitude is not None and longitude is not None:
                logger.info(
                    f"  GPS coordinates ({gps_source}): {latitude:.6f}, {longitude:.6f}"
                )
                if self.config.get("processing.use_exif_location", True):
                    # Use the new unified location resolution
                    check_custom = self.config.get("location.check_custom_first", True)
                    custom_file = self.config.get("location.custom_locations_file")
                    
                    location_string, location_aliases, is_custom = resolve_location_with_custom(
                        latitude,
                        longitude,
                        check_custom_first=check_custom,
                        custom_locations_file=custom_file,
                        interactive=False
                    )
                    
                    if is_custom:
                        logger.info(f"  Location (custom): {location_string}")
                        if location_aliases:
                            logger.info(f"  Location aliases: {', '.join(location_aliases)}")
                    elif location_string:
                        logger.info(f"  Location (geocoded): {location_string}")
                    else:
                        logger.info("  Location: Could not resolve location name")
            else:
                logger.debug("  No GPS coordinates available")
            
            # Create exif_location object for downstream use (for location tags extraction)
            if exif_location is None:
                # We got GPS from SmugMug API, create a minimal ExifLocation object
                from ..utils.exif import ExifLocation
                exif_location = ExifLocation(
                    latitude=latitude,
                    longitude=longitude,
                    has_coordinates=(latitude is not None and longitude is not None)
                )
            
            # Update with resolved location data
            exif_location.location_name = location_string
            exif_location.location_aliases = location_aliases
            exif_location.is_custom_location = is_custom
            
            # Detect and identify faces
            person_names = []
            if self.face_recognizer:
                logger.debug("Detecting faces")
                raw_names = self.face_recognizer.get_person_names(str(image_path))
                # Format names: replace underscores with spaces
                person_names = [name.replace('_', ' ') for name in raw_names]
                result.faces_detected = len(person_names)
                if person_names:
                    logger.info(f"  Identified: {', '.join(person_names)}")
            
            # Generate caption
            logger.debug("Generating caption")
            caption_prompt = self._build_caption_prompt(
                location=location_string,
                person_names=person_names,
                album_name=album.name
            )
            ai_caption = self.vision.generate_caption(
                image_path=str(image_path),
                prompt=caption_prompt
            )
            
            # Generate tags
            logger.debug("Generating tags")
            tags_prompt = self._build_tags_prompt(
                location=location_string,
                person_names=person_names,
                album_name=album.name
            )
            ai_tags = self.vision.generate_tags(
                image_path=str(image_path),
                prompt=tags_prompt
            )
            
            # Format metadata
            final_caption = self.formatter.format_caption(
                ai_caption=ai_caption,
                existing_caption=image.caption,
                location=location_string,
                person_names=person_names
            )
            
            # Extract location tags, including aliases from custom locations
            location_tags = self._extract_location_tags(exif_location) if exif_location.has_coordinates else None
            
            # Add location aliases as tags if configured
            if self.config.get("location.use_aliases_as_tags", True) and location_aliases:
                if location_tags is None:
                    location_tags = []
                location_tags.extend(location_aliases)
            
            final_tags = self.formatter.format_tags(
                ai_tags=ai_tags,
                existing_tags=image.keywords,
                person_names=person_names,
                location_tags=location_tags
            )
            
            # Store the generated metadata in the result for inspection/UI
            result.proposed_caption = final_caption
            result.proposed_keywords = final_tags
            result.detected_faces = person_names
            result.location = location_string
            result.location_aliases = location_aliases
            
            # Update SmugMug
            if not self.dry_run:
                logger.debug("Updating SmugMug")
                payload = self.formatter.create_update_payload(
                    caption=final_caption,
                    tags=final_tags
                )
                self.smugmug.update_image_metadata(image.image_key, payload)
            else:
                logger.info("  [DRY RUN] Would update with:")
                logger.info(f"    Caption: {final_caption}")
                logger.info(f"    Tags ({len(final_tags)}): {', '.join(final_tags)}")
            
            result.success = True
            result.caption_generated = bool(ai_caption)
            result.tags_generated = len(ai_tags) if ai_tags else 0
            
        except Exception as e:
            logger.error(f"Error processing {image.file_name}: {e}", exc_info=True)
            result.error = str(e)
        
        result.processing_time = time.time() - start_time
        return result
    
    def _download_image(self, image: AlbumImage, album: Album) -> Optional[Path]:
        """Download image to cache.
        
        Args:
            image: AlbumImage to download
            album: Parent album
            
        Returns:
            Path to cached image or None if failed
        """
        try:
            # Get album cache directory
            album_cache_dir = self.cache.get_album_cache_dir(
                album_name=album.name,
                folder_path=None  # Could extract from album path in future
            )
            
            # Download using SmugMug client
            size = self.config.get("processing.image_size", "Medium")
            path = self.smugmug.download_image(
                image=image,
                destination=str(album_cache_dir),
                size=size,
                skip_if_exists=True
            )
            
            return path
        except Exception as e:
            logger.error(f"Failed to download {image.file_name}: {e}")
            return None
    
    def _build_caption_prompt(
        self,
        location: Optional[str] = None,
        person_names: Optional[List[str]] = None,
        album_name: Optional[str] = None
    ) -> str:
        """Build caption prompt with context.
        
        Args:
            location: Location string from EXIF
            person_names: Identified person names
            album_name: Name of the album this image belongs to
            
        Returns:
            Enhanced prompt string
        """
        base_prompt = self.config.get(
            "prompts.caption",
            "Analyze this image and provide a concise, descriptive caption."
        )
        
        context_parts = []
        
        if album_name:
            context_parts.append(
                f"Album name: {album_name}\n"
                f"(The album name often describes the event, occasion, or subject - "
                f"e.g., pet names, birthdays, trips, celebrations. Use this context "
                f"to inform your caption.)"
            )
        
        if location:
            context_parts.append(f"Location: {location}")
        
        if person_names:
            names_str = ", ".join(person_names)
            context_parts.append(f"People identified: {names_str}")
        
        if context_parts:
            context = "\n".join(context_parts)
            return f"{base_prompt}\n\nContext:\n{context}"
        
        return base_prompt
    
    def _build_tags_prompt(
        self,
        location: Optional[str] = None,
        person_names: Optional[List[str]] = None,
        album_name: Optional[str] = None
    ) -> str:
        """Build tags prompt with context.
        
        Args:
            location: Location string from EXIF
            person_names: Identified person names
            album_name: Name of the album this image belongs to
            
        Returns:
            Enhanced prompt string
        """
        base_prompt = self.config.get(
            "prompts.tags",
            "Generate 5-10 relevant keyword tags for this image."
        )
        
        context_parts = []
        
        if album_name:
            context_parts.append(
                f"Album: {album_name} (may contain event/occasion info like pet names, birthdays, trips)"
            )
        
        if location:
            context_parts.append(f"Location: {location}")
        
        if person_names:
            context_parts.append(f"People: {', '.join(person_names)}")
        
        if context_parts:
            context = " | ".join(context_parts)
            return f"{base_prompt}\n\nContext: {context}"
        
        return base_prompt
    
    def _extract_location_tags(self, exif_location) -> Optional[List[str]]:
        """Extract location-based tags from EXIF location data.
        
        Args:
            exif_location: ExifLocation object with location data
            
        Returns:
            List of location tags or None
        """
        tags = []
        
        # Extract location components from the location_name string if available
        # For now, just add the full location name as a tag
        # In the future, could parse location_name to extract city, state, etc.
        if exif_location.location_name:
            # Split on common separators and add significant parts as tags
            parts = exif_location.location_name.replace(',', '|').split('|')
            for part in parts:
                part = part.strip()
                if part and len(part) > 2:  # Skip very short parts
                    tags.append(part)
        
        return tags if tags else None

