"""Main image processing orchestrator."""

import gc
import re
from typing import List, Optional
from pathlib import Path
from dataclasses import dataclass
import logging
import time

from ..config import ConfigManager
from ..smugmug import SmugMugClient, AlbumImage, Album
from ..cache import CacheManager
from ..vision import VisionModelFactory
from ..vision.base import VisionModel
from ..utils.exif import extract_exif_location, resolve_location_with_custom
from ..face.recognizer import FaceRecognizer
from .hints import HintManager
from .metadata import MetadataFormatter

logger = logging.getLogger(__name__)

# Fallbacks used when prompts.caption / prompts.tags are absent from the config.
DEFAULT_CAPTION_PROMPT = "Analyze this image and provide a concise, descriptive caption."
DEFAULT_TAGS_PROMPT = "Generate 5-10 relevant keyword tags for this image."


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
        faces_detected: Number of faces detected in the image, including faces that
            could not be matched to a reference person
        processing_time: Time taken to process (seconds)
        error: Error message if failed

        # Detailed results for UI/inspection
        current_caption: Original caption before processing
        current_keywords: Original keywords before processing
        proposed_caption: Generated caption (what was/would be written)
        proposed_keywords: Generated keywords (what was/would be written)
        proposed_title: Generated short title, or None when processing.generate_titles
            is off or the model returned nothing usable
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
    proposed_title: Optional[str] = None
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
        hint_manager: Optional[HintManager] = None,
        dry_run: bool = False,
        preserve_existing: Optional[bool] = None,
        generate_titles: Optional[bool] = None,
    ) -> None:
        """Initialize image processor.

        Args:
            config: Configuration manager
            smugmug_client: SmugMug API client (created if not provided)
            vision_model: Vision model for caption/tag generation (created if not provided)
            cache_manager: Cache manager (created if not provided)
            face_recognizer: Face recognizer (created if not provided)
            hint_manager: Hint manager for user-asserted facts (created if not provided,
                unless ``hints.enabled`` is false, in which case hints are off entirely)
            dry_run: If True, don't update SmugMug
            preserve_existing: Overrides ``processing.preserve_existing`` for this run.
                ``None`` (the default) uses the configured value. ``False`` replaces the
                existing caption and keywords instead of merging into them, which
                discards anything already on the image.
            generate_titles: Overrides ``processing.generate_titles`` for this run.
                ``None`` (the default) uses the configured value.
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
                access_token_secret=config.get("smugmug.user_secret"),
            )

        # Initialize cache manager
        if cache_manager:
            self.cache = cache_manager
        else:
            cache_dir = config.get("cache.directory", "~/.smugvision/cache")
            self.cache = CacheManager(cache_dir)

        # Vision model behaviour flags (read before the model is built so they can be
        # forwarded to it and used to pick the generation path in process_image)
        self.vision_temperature = config.get("vision.temperature", 0.7)
        self.vision_max_tokens = config.get("vision.max_tokens", 500)
        self.vision_single_call = bool(config.get("vision.single_call", True))

        # Initialize vision model
        if vision_model:
            self.vision = vision_model
        else:
            model_name = config.get("vision.model", "llama3.2-vision")
            # No literal fallback: an unset vision.endpoint must stay None so the ollama
            # client resolves the host itself ($OLLAMA_HOST, else http://localhost:11434).
            endpoint = config.get("vision.endpoint")
            self.vision = VisionModelFactory.create(
                model_name=model_name,
                endpoint=endpoint,
                timeout=config.get("vision.timeout", 120),
                think=config.get("vision.think", False),
                keep_alive=config.get("vision.keep_alive", "30m"),
                single_call=self.vision_single_call,
                structured_output=config.get("vision.structured_output", True),
                max_image_dimension=config.get("vision.max_image_dimension", 1568),
                jpeg_quality=config.get("vision.jpeg_quality", 85),
                validate_model=config.get("vision.validate_model", True),
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
                        "face_recognition.reference_faces_dir", "~/.smugvision/reference_faces"
                    )
                    # Expand ~ and convert to absolute path
                    reference_faces_path = PathLib(reference_faces_dir).expanduser()

                    # Get cache settings
                    use_cache = config.get("face_recognition.use_cache", True)
                    cache_dir = config.get(
                        "face_recognition.cache_dir", "~/.smugvision/cache/face_encodings"
                    )
                    cache_dir_path = PathLib(cache_dir).expanduser()

                    # Backend selection ("dlib" or "insightface"). Tuning settings may live
                    # at either depth: historically they sat at the top level of
                    # face_recognition (tolerance/model/detection_scale), and a per-backend
                    # sub-block (e.g. face_recognition.insightface) was added later. Read
                    # both here so there is one answer to "where does a backend setting
                    # live" -- the sub-block wins on conflict.
                    backend = config.get("face_recognition.backend", "dlib") or "dlib"
                    backend_options = dict(config.get(f"face_recognition.{backend}", {}) or {})
                    legacy_keys = {
                        "tolerance": config.get("face_recognition.tolerance"),
                        "model": config.get("face_recognition.model"),
                        "detection_scale": config.get("face_recognition.detection_scale"),
                    }
                    for key, value in legacy_keys.items():
                        if value is not None and key not in backend_options:
                            backend_options[key] = value

                    # Only initialize if directory exists
                    if reference_faces_path.exists():
                        self.face_recognizer = FaceRecognizer(
                            str(reference_faces_path),
                            cache_dir=str(cache_dir_path),
                            use_cache=use_cache,
                            backend=backend,
                            backend_options=backend_options,
                        )
                        logger.info(
                            f"Face recognition enabled ({backend}) with "
                            f"{len(self.face_recognizer.reference_faces)} person(s)"
                        )
                    else:
                        logger.info(
                            f"Face recognition disabled: reference faces directory "
                            f"not found at {reference_faces_path}"
                        )
                except Exception as e:
                    logger.warning(f"Could not initialize face recognizer: {e}")

        # Initialize hint manager. Hints are facts the model cannot see for itself
        # (that the white ribbed object is a dog chew, not a cracker), resolved per
        # image from the global / album / image scopes in ~/.smugvision/hints.yaml.
        # Gated the same way as face recognition: the config switch wins, and an
        # injected manager is used in place of building one.
        self.hints: Optional[HintManager] = None
        if config.get("hints.enabled", True):
            if hint_manager:
                self.hints = hint_manager
            else:
                try:
                    self.hints = HintManager(config.get("hints.file"))
                except Exception as e:
                    # A hint is an optimization, never a reason to lose a run.
                    logger.warning(f"Could not initialize hint manager: {e}")
            if self.hints:
                logger.info(
                    f"Hints enabled: {self.hints.hint_count} hint(s) from "
                    f"{self.hints.hints_file}"
                )
        else:
            logger.debug("Hints disabled by config (hints.enabled=false)")

        # Initialize metadata formatter. An explicit preserve_existing argument (from
        # --preserve-existing / --no-preserve-existing) overrides the configured value
        # for this run only; None means "whatever the config says".
        if preserve_existing is None:
            self.preserve_existing = bool(config.get("processing.preserve_existing", True))
        else:
            self.preserve_existing = bool(preserve_existing)
            logger.info(
                f"processing.preserve_existing overridden for this run: "
                f"{self.preserve_existing}"
            )
        # Titles are opt-in: smugVision has never written Title, so turning it on
        # silently would start changing a field users may curate by hand.
        if generate_titles is None:
            self.generate_titles = bool(config.get("processing.generate_titles", False))
        else:
            self.generate_titles = bool(generate_titles)
            logger.info(
                f"processing.generate_titles overridden for this run: " f"{self.generate_titles}"
            )

        self.formatter = MetadataFormatter(
            preserve_existing=self.preserve_existing,
            marker_tag=config.get("processing.marker_tag", "smugvision"),
        )

        logger.info(
            f"ImageProcessor initialized: model={self.vision.model_name}, "
            f"max_tokens={self.vision_max_tokens}, single_call={self.vision_single_call}, "
            f"dry_run={dry_run}"
        )

    def process_album(
        self, album_key: str, force_reprocess: bool = False, skip_videos: bool = True
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

            result = self.process_image(image=image, album=album, force_reprocess=force_reprocess)

            stats.results.append(result)

            if result.success:
                stats.processed += 1
            elif result.skipped:
                stats.skipped += 1
            else:
                stats.errors += 1

            # Log progress
            if result.success:
                outcome = "✓ Success"
            elif result.skipped:
                outcome = "○ Skipped"
            else:
                outcome = "✗ Error"
            logger.info(f"  Result: {outcome} ({result.processing_time:.1f}s)")

        stats.total_time = time.time() - start_time

        logger.info(
            f"Album processing complete: {stats.processed} processed, "
            f"{stats.skipped} skipped, {stats.errors} errors "
            f"(Total time: {stats.total_time:.1f}s)"
        )

        return stats

    def process_image(
        self, image: AlbumImage, album: Album, force_reprocess: bool = False
    ) -> ProcessingResult:
        """Process a single image.

        Runs the full pipeline: marker-tag skip, download/cache, GPS resolution
        (SmugMug API first, EXIF second), face recognition, caption/tag generation and
        the SmugMug write (skipped in dry-run mode). Generation uses one combined
        inference call unless vision.single_call is false.

        Args:
            image: AlbumImage to process
            album: Parent album
            force_reprocess: If True, process even if already marked

        Returns:
            ProcessingResult
        """
        start_time = time.time()
        existing_keywords = self._split_keywords(image.keywords)
        result = ProcessingResult(
            image_key=image.image_key,
            filename=image.file_name,
            success=False,
            current_caption=image.caption,
            current_keywords=existing_keywords,
        )

        try:
            # Check if already processed
            marker_tag = self.config.get("processing.marker_tag", "smugvision")
            if not force_reprocess and self._has_marker_tag(image, marker_tag):
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
                    album_name=album.name, folder_path=None
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
                logger.info(f"  GPS coordinates ({gps_source}): {latitude:.6f}, {longitude:.6f}")
                if self.config.get("processing.use_exif_location", True):
                    # Use the new unified location resolution
                    check_custom = self.config.get("location.check_custom_first", True)
                    custom_file = self.config.get("location.custom_locations_file")

                    location_string, location_aliases, is_custom = resolve_location_with_custom(
                        latitude,
                        longitude,
                        check_custom_first=check_custom,
                        custom_locations_file=custom_file,
                        interactive=False,
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
                    has_coordinates=(latitude is not None and longitude is not None),
                )

            # A location override from hints.yaml replaces the resolved value outright,
            # before anything downstream reads it. This has to happen here rather than
            # via the prompt: the geocoded name also feeds the location tags, the
            # caption suffix appended by MetadataFormatter, and result.location shown in
            # the UI, none of which the model can influence. Left in the prompt only, a
            # correction would argue with a geocoded name asserted alongside it and lose.
            location_override: Optional[str] = None
            if self.hints:
                location_override = self.hints.resolve_location(album.album_key, image.image_key)
            if location_override:
                logger.info(
                    f"  Location (hint override): {location_override} "
                    f"(was: {location_string or 'unresolved'})"
                )
                location_string = location_override
                # Aliases came from a custom-locations match that no longer applies.
                location_aliases = []
                # Treat it as user-asserted, same standing as a custom location.
                is_custom = True

            # Update with resolved location data
            exif_location.location_name = location_string
            exif_location.location_aliases = location_aliases
            exif_location.is_custom_location = is_custom

            # Detect and identify faces.
            # raw_names keep the reference-folder spelling ("John_Doe") because that is
            # what relationships.yaml is keyed on; person_names is the display form.
            raw_names: List[str] = []
            person_names: List[str] = []
            total_faces = 0
            if self.face_recognizer:
                logger.debug("Detecting faces")
                min_confidence = self.config.get("face_recognition.min_confidence", 0.25)
                raw_names = self.face_recognizer.get_person_names(
                    str(image_path), min_confidence=min_confidence
                )
                # Total faces DETECTED, which can exceed the number recognized. The
                # recognizer memoizes detection per file, so this reuses the pass
                # get_person_names() just made instead of detecting a second time.
                total_faces = self._count_detected_faces(str(image_path), len(raw_names))
                # Format names: replace underscores with spaces
                person_names = [name.replace("_", " ") for name in raw_names]
                result.faces_detected = total_faces
                if person_names:
                    logger.info(
                        f"  Identified {len(person_names)} of {total_faces} face(s): "
                        f"{', '.join(person_names)}"
                    )
                elif total_faces:
                    logger.info(f"  Detected {total_faces} face(s), none identified")

            # A people override from hints.yaml replaces the recognised set outright.
            # This has to happen here, not via a free-text note: the recognised names
            # also feed the keywords (MetadataFormatter.format_tags), the relationships
            # lookup and result.detected_faces, none of which a note can reach. Applied
            # even when face_recognition is disabled entirely, so naming people works
            # without a working recogniser.
            people_override: Optional[List[str]] = None
            if self.hints:
                people_override = self.hints.resolve_people(album.album_key, image.image_key)
            if people_override:
                previous = ", ".join(person_names) if person_names else "nobody recognised"
                raw_names = people_override
                person_names = [name.replace("_", " ") for name in raw_names]
                # Detection may have found fewer faces than the user named (a face in
                # profile, or turned away). Never claim fewer people than are named, or
                # the prompt would say "some of them are" about a complete list.
                total_faces = max(total_faces, len(raw_names))
                result.faces_detected = total_faces
                logger.info(
                    f"  People (hint override): {', '.join(person_names)} " f"(was: {previous})"
                )

            # Generate caption and tags
            caption_instruction = self.config.get("prompts.caption", DEFAULT_CAPTION_PROMPT)
            tags_instruction = self.config.get("prompts.tags", DEFAULT_TAGS_PROMPT)

            # Resolve user-asserted hints for this image (global + album + image scopes).
            # album.album_key is used deliberately: AlbumImage.album_key is empty for
            # images fetched one at a time via SmugMugClient.get_image(), which would
            # make per-album hints vanish in exactly the single-image re-run path.
            hints_text = ""
            if self.hints:
                try:
                    hints_text = self.hints.resolve(album.album_key, image.image_key)
                except Exception as e:
                    logger.warning(f"Could not resolve hints for {image.file_name}: {e}")
                if hints_text:
                    logger.info(f"  Hints applied: {hints_text}")

            # One call, whatever the request shape. `vision.single_call` decides whether the
            # vision layer issues one request or two; that is a request-shaping detail it
            # owns, so context (album, location, people, relationships.yaml) is passed as
            # arguments and injected there rather than baked into the prompt text here.
            metadata = self.vision.generate_metadata(
                str(image_path),
                caption_instruction,
                tags_instruction,
                temperature=self.vision_temperature,
                max_tokens=self.vision_max_tokens,
                location_context=location_string,
                person_names=raw_names,
                total_faces=total_faces if self.face_recognizer else None,
                album_name=album.name,
                hints=hints_text or None,
                title_instruction=(
                    self.config.get("prompts.title") if self.generate_titles else None
                ),
            )
            ai_caption = metadata.caption
            ai_tags = list(metadata.tags)
            # Never blank an existing title: a model that ignored the field, or produced
            # something over-long that the vision layer discarded, must leave Title alone.
            final_title = metadata.title.strip() if self.generate_titles else ""
            result.proposed_title = final_title or None

            # Format metadata
            final_caption = self.formatter.format_caption(
                ai_caption=ai_caption,
                existing_caption=image.caption,
                location=location_string,
                person_names=person_names,
            )

            # Extract location tags, including aliases from custom locations. When the
            # location was overridden by a hint, the geocoded components (city, county,
            # state, country) describe a place the user has told us is wrong, so they are
            # replaced by the override's own comma-separated parts rather than merged
            # with them - otherwise the corrected caption would still carry the tags it
            # was correcting.
            if location_override:
                location_tags = [
                    part.strip() for part in location_override.split(",") if part.strip()
                ]
            elif exif_location.has_coordinates:
                location_tags = self._extract_location_tags(exif_location)
            else:
                location_tags = None

            # Add location aliases as tags if configured
            if self.config.get("location.use_aliases_as_tags", True) and location_aliases:
                if location_tags is None:
                    location_tags = []
                location_tags.extend(location_aliases)

            final_tags = self.formatter.format_tags(
                ai_tags=ai_tags,
                existing_tags=existing_keywords,
                person_names=person_names,
                location_tags=location_tags,
            )

            # Store the generated metadata in the result for inspection/UI
            result.proposed_caption = final_caption
            result.proposed_keywords = final_tags
            result.detected_faces = person_names
            result.location = location_string
            result.location_aliases = location_aliases

            # Update SmugMug. update_image_metadata() takes the caption and the keyword
            # LIST as keyword arguments and builds the PATCH body itself (it joins the
            # keywords into the comma-separated string the API expects). Passing None
            # leaves a field untouched, so an empty caption never clears an existing one.
            if not self.dry_run:
                logger.debug("Updating SmugMug")
                self.smugmug.update_image_metadata(
                    image_key=image.image_key,
                    caption=final_caption or None,
                    keywords=final_tags or None,
                    title=final_title or None,
                )
            else:
                logger.info("  [DRY RUN] Would update with:")
                if final_title:
                    logger.info(f"    Title: {final_title}")
                logger.info(f"    Caption: {final_caption}")
                logger.info(f"    Tags ({len(final_tags)}): {', '.join(final_tags)}")

            result.success = True
            result.caption_generated = bool(ai_caption)
            result.tags_generated = len(ai_tags) if ai_tags else 0

        except Exception as e:
            logger.error(f"Error processing {image.file_name}: {e}", exc_info=True)
            result.error = str(e)
        finally:
            # Force garbage collection after each image to keep memory usage low
            # This is important because image processing creates large temporary objects
            gc.collect()

        result.processing_time = time.time() - start_time
        return result

    @staticmethod
    def _split_keywords(keywords: Optional[List[str]]) -> List[str]:
        """Split SmugMug keyword strings into individual tags.

        SmugMug stores keywords as one string and hands it back SEMICOLON separated
        ("dog; park; smugvision") even when it was written comma separated, but
        AlbumImage.from_api_response only splits on commas. A previously written tag
        list therefore arrives as a single blob, which hides the marker tag and
        defeats deduplication. Splitting on both separators recovers the real tags.

        Args:
            keywords: Keyword values as parsed from the SmugMug API (may be None)

        Returns:
            List of individual, non-empty tags in their original order
        """
        if not keywords:
            return []

        split_tags: List[str] = []
        for keyword in keywords:
            for part in re.split(r"[;,]", str(keyword)):
                part = part.strip()
                if part:
                    split_tags.append(part)
        return split_tags

    def _has_marker_tag(self, image: AlbumImage, marker_tag: str) -> bool:
        """Check whether an image already carries the processing marker tag.

        Falls back to a separator-aware check because SmugMug returns keywords as a
        semicolon-separated blob, which makes AlbumImage.has_marker_tag() miss the
        marker on every image smugVision has already processed. This only ever turns a
        missed marker into a detected one, so it can add skips but never remove them.

        Args:
            image: AlbumImage to inspect
            marker_tag: Marker tag configured in processing.marker_tag

        Returns:
            True if the marker tag is present in the image's keywords
        """
        if image.has_marker_tag(marker_tag):
            return True
        target = marker_tag.strip().lower()
        return any(tag.lower() == target for tag in self._split_keywords(image.keywords))

    def _count_detected_faces(self, image_path: str, recognized_count: int) -> int:
        """Count the faces detected in an image, including unrecognized ones.

        FaceRecognizer memoizes its most recent detection, so calling this straight
        after get_person_names() on the same file reuses that pass rather than running
        detection again.

        Args:
            image_path: Path to the image that was just analyzed
            recognized_count: Number of people recognized, used as a floor if the
                recognizer cannot report a detected count

        Returns:
            Number of faces detected, never less than recognized_count
        """
        if not self.face_recognizer:
            return 0
        try:
            detected = int(self.face_recognizer.get_face_count(image_path))
        except Exception as e:
            logger.debug(f"Could not determine detected face count: {e}")
            return recognized_count
        return max(detected, recognized_count)

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
                album_name=album.name, folder_path=None  # Could extract from album path in future
            )

            # Download using SmugMug client
            size = self.config.get("processing.image_size", "Medium")
            path = self.smugmug.download_image(
                image=image, destination=str(album_cache_dir), size=size, skip_if_exists=True
            )

            return path
        except Exception as e:
            logger.error(f"Failed to download {image.file_name}: {e}")
            return None

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
            parts = exif_location.location_name.replace(",", "|").split("|")
            for part in parts:
                part = part.strip()
                if part and len(part) > 2:  # Skip very short parts
                    tags.append(part)

        return tags if tags else None
