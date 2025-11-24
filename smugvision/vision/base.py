"""Abstract base class for vision models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetadataResult:
    """Generated metadata for an image.
    
    Attributes:
        caption: Generated caption text
        tags: List of generated keyword tags
        confidence: Confidence score from 0.0 to 1.0
        model_used: Name of the model that generated the metadata
        processing_time: Time taken to process in seconds
    """
    caption: str
    tags: List[str]
    confidence: float
    model_used: str
    processing_time: float


class VisionModel(ABC):
    """Abstract base class for vision models.
    
    This class defines the interface that all vision model implementations
    must follow. It provides methods for generating captions and tags from images.
    
    Attributes:
        model_name: Name identifier for the model
        endpoint: API endpoint URL for the model service
    """
    
    def __init__(self, model_name: str, endpoint: Optional[str] = None) -> None:
        """Initialize the vision model.
        
        Args:
            model_name: Name identifier for the model
            endpoint: Optional API endpoint URL (for local/remote services)
        """
        self.model_name = model_name
        self.endpoint = endpoint
        logger.info(f"Initialized {self.__class__.__name__} with model: {model_name}")
    
    @abstractmethod
    def generate_caption(
        self, 
        image_path: str, 
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 150,
        location_context: Optional[str] = None,
        person_names: Optional[List[str]] = None,
        total_faces: Optional[int] = None
    ) -> str:
        """Generate a caption for an image.
        
        Args:
            image_path: Path to the image file
            prompt: Prompt text to guide caption generation
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens in response
            location_context: Optional location information to include in prompt
            person_names: Optional list of identified person names to include in prompt
            total_faces: Optional total number of faces detected (including unrecognized)
            
        Returns:
            Generated caption text
            
        Raises:
            VisionModelError: If caption generation fails
        """
        pass
    
    @abstractmethod
    def generate_tags(
        self,
        image_path: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 150
    ) -> List[str]:
        """Generate keyword tags for an image.
        
        Args:
            image_path: Path to the image file
            prompt: Prompt text to guide tag generation
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens in response
            
        Returns:
            List of generated keyword tags
            
        Raises:
            VisionModelError: If tag generation fails
        """
        pass
    
    def process_image(
        self,
        image_path: str,
        caption_prompt: str,
        tags_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 150,
        generate_caption: bool = True,
        generate_tags: bool = True,
        use_exif_location: bool = True,
        face_recognizer: Optional[object] = None
    ) -> MetadataResult:
        """Process an image to generate both caption and tags.
        
        This is a convenience method that generates both caption and tags
        in a single call, returning a MetadataResult object. It automatically
        extracts EXIF location data if available and includes it in the caption prompt.
        
        Args:
            image_path: Path to the image file
            caption_prompt: Prompt for caption generation
            tags_prompt: Prompt for tag generation
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens in response
            generate_caption: Whether to generate caption
            generate_tags: Whether to generate tags
            use_exif_location: Whether to extract and use EXIF location data
            
        Returns:
            MetadataResult containing generated metadata
            
        Raises:
            VisionModelError: If processing fails
        """
        import time
        
        start_time = time.time()
        caption = ""
        tags: List[str] = []
        location_context: Optional[str] = None
        person_names: List[str] = []
        
        # Identify faces if face recognizer is provided
        total_faces = 0
        if face_recognizer and generate_caption:
            try:
                person_names = face_recognizer.get_person_names(image_path)
                total_faces = face_recognizer.get_face_count(image_path)
                
                if person_names:
                    recognized_count = len(person_names)
                    logger.info(
                        f"Detected {total_faces} face(s) in image, "
                        f"identified {recognized_count} person(s): "
                        f"{', '.join(person_names)}"
                    )
                    if total_faces > recognized_count:
                        logger.debug(
                            f"{total_faces - recognized_count} face(s) could not be identified"
                        )
            except Exception as e:
                logger.warning(
                    f"Failed to identify faces in {image_path}: {e}",
                    exc_info=True
                )
        
        # Extract EXIF location if requested
        if use_exif_location and generate_caption:
            try:
                from smugvision.utils.exif import extract_exif_location, reverse_geocode
                
                logger.debug(f"Extracting EXIF location from {image_path}")
                exif_location = extract_exif_location(image_path)
                
                if exif_location.has_coordinates:
                    logger.info(
                        f"Found GPS coordinates: {exif_location.latitude:.6f}, "
                        f"{exif_location.longitude:.6f}"
                    )
                    # Try reverse geocoding
                    if exif_location.latitude and exif_location.longitude:
                        logger.debug("Attempting reverse geocoding...")
                        # Use interactive=False for batch processing (can be made configurable)
                        location_name = reverse_geocode(
                            exif_location.latitude,
                            exif_location.longitude,
                            interactive=False
                        )
                        if location_name:
                            location_context = location_name
                            logger.info(
                                f"Using location context for caption: {location_context}"
                            )
                        else:
                            # Fallback to coordinates if geocoding fails
                            location_context = (
                                f"{exif_location.latitude:.6f}, "
                                f"{exif_location.longitude:.6f}"
                            )
                            logger.info(
                                f"Using GPS coordinates as location context: "
                                f"{location_context}"
                            )
                else:
                    logger.debug(f"No GPS coordinates found in EXIF data for {image_path}")
            except Exception as e:
                logger.warning(
                    f"Failed to extract EXIF location from {image_path}: {e}",
                    exc_info=True
                )
        
        try:
            if generate_caption:
                logger.debug(f"Generating caption for image: {image_path}")
                caption = self.generate_caption(
                    image_path,
                    caption_prompt,
                    temperature,
                    max_tokens,
                    location_context=location_context,
                    person_names=person_names,
                    total_faces=total_faces if face_recognizer else None
                )
            
            if generate_tags:
                logger.debug(f"Generating tags for image: {image_path}")
                tags = self.generate_tags(
                    image_path, tags_prompt, temperature, max_tokens
                )
                
                # Always add identified person names as tags
                if person_names:
                    formatted_names = [name.replace('_', ' ') for name in person_names]
                    for name in formatted_names:
                        # Add name as tag if not already present (case-insensitive)
                        name_lower = name.lower()
                        if not any(tag.lower() == name_lower for tag in tags):
                            tags.insert(0, name)  # Add at beginning
                            logger.debug(f"Added person name to tags: {name}")
            
            processing_time = time.time() - start_time
            
            return MetadataResult(
                caption=caption,
                tags=tags,
                confidence=1.0,  # Default confidence, can be overridden
                model_used=self.model_name,
                processing_time=processing_time
            )
        except Exception as e:
            logger.error(
                f"Failed to process image {image_path}: {e}", 
                exc_info=True
            )
            raise

