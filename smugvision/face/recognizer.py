"""Face recognition using reference faces."""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Suppress pkg_resources deprecation warning from face_recognition_models
warnings.filterwarnings('ignore', message='.*pkg_resources is deprecated.*', category=UserWarning)

logger = logging.getLogger(__name__)

# Try to import face_recognition, make it optional
FACE_RECOGNITION_AVAILABLE = False
FACE_RECOGNITION_ERROR = None

try:
    # Suppress pkg_resources deprecation warning before importing
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*pkg_resources is deprecated.*', category=UserWarning)
        import face_recognition
    # The library imports successfully, but models might not be available
    # We'll set it to available and let actual usage catch model errors
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    FACE_RECOGNITION_ERROR = "not_installed"
    logger.warning(
        "face_recognition library not available. "
        "Install with: pip install face_recognition"
    )


class FaceRecognizer:
    """Face recognition system using reference faces.
    
    This class loads reference face images and can identify people in new images
    by comparing detected faces to the reference set.
    
    Attributes:
        reference_faces: Dictionary mapping person names to face encodings
        tolerance: How much distance between faces to consider it a match (lower = stricter)
    """
    
    def __init__(
        self,
        reference_faces_dir: Optional[str] = None,
        tolerance: float = 0.6
    ) -> None:
        """Initialize face recognizer.
        
        Args:
            reference_faces_dir: Directory containing reference face images.
                                 Images should be named with person names
                                 (e.g., "John_Doe.jpg", "Jane_Smith.png")
            tolerance: Face recognition tolerance (0.0-1.0). Lower is stricter.
                      Default 0.6 is a good balance.
        
        Raises:
            ImportError: If face_recognition library is not installed or models are missing
        """
        if not FACE_RECOGNITION_AVAILABLE:
            if FACE_RECOGNITION_ERROR == "models_missing":
                raise ImportError(
                    "face_recognition models are required. "
                    "Install with: pip install git+https://github.com/ageitgey/face_recognition_models"
                )
            elif FACE_RECOGNITION_ERROR == "not_installed":
                raise ImportError(
                    "face_recognition library is required. "
                    "Install with: pip install face_recognition\n"
                    "Then install models with: pip install git+https://github.com/ageitgey/face_recognition_models"
                )
            else:
                raise ImportError(
                    f"face_recognition library error: {FACE_RECOGNITION_ERROR}\n"
                    "Make sure face_recognition and models are installed:\n"
                    "  pip install face_recognition\n"
                    "  pip install git+https://github.com/ageitgey/face_recognition_models"
                )
        
        self.reference_faces: Dict[str, List] = {}
        self.tolerance = tolerance
        
        if reference_faces_dir:
            self.load_reference_faces(reference_faces_dir)
    
    def load_reference_faces(self, directory: str) -> None:
        """Load reference face images from a directory.
        
        Each image file should be named with the person's name.
        Multiple images per person are supported (use same name prefix).
        
        Example directory structure:
            faces/
                John_Doe_1.jpg
                John_Doe_2.jpg
                Jane_Smith.jpg
                Bob_Johnson.png
        
        Args:
            directory: Path to directory containing reference face images
        """
        ref_dir = Path(directory)
        if not ref_dir.exists():
            logger.warning(f"Reference faces directory not found: {directory}")
            return
        
        logger.info(f"Loading reference faces from: {directory}")
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.heic', '.heif'}
        
        loaded_count = 0
        for image_path in ref_dir.iterdir():
            if not image_path.is_file():
                continue
            
            if image_path.suffix.lower() not in image_extensions:
                continue
            
            # Extract person name from filename (remove extension and numbers)
            person_name = image_path.stem
            # Remove trailing numbers/underscores (e.g., "John_Doe_1" -> "John_Doe")
            person_name = person_name.rsplit('_', 1)[0] if '_' in person_name else person_name
            
            try:
                # Load and encode the face
                encoding = self._encode_face(image_path)
                if encoding is not None:
                    if person_name not in self.reference_faces:
                        self.reference_faces[person_name] = []
                    self.reference_faces[person_name].append(encoding)
                    loaded_count += 1
                    logger.debug(f"Loaded reference face: {person_name} from {image_path.name}")
                else:
                    logger.warning(f"No face found in reference image: {image_path}")
            except Exception as e:
                logger.warning(
                    f"Failed to load reference face from {image_path}: {e}"
                )
        
        logger.info(
            f"Loaded {loaded_count} reference face(s) for "
            f"{len(self.reference_faces)} person(s)"
        )
    
    def _encode_face(self, image_path: str) -> Optional[List]:
        """Encode a face from an image file.
        
        Args:
            image_path: Path to image file containing a face
            
        Returns:
            Face encoding (128-dimensional vector), or None if no face found
            
        Raises:
            ImportError: If face_recognition models are not installed
        """
        try:
            # Load image (face_recognition handles various formats)
            image = face_recognition.load_image_file(str(image_path))
            
            # Find face locations
            face_locations = face_recognition.face_locations(image)
            
            if not face_locations or len(face_locations) == 0:
                return None
            
            # Get face encodings (use first face if multiple)
            face_encodings = face_recognition.face_encodings(
                image, 
                face_locations
            )
            
            # Check if we have any encodings (use len() to avoid NumPy boolean ambiguity)
            if face_encodings and len(face_encodings) > 0:
                return face_encodings[0]  # Return first face encoding
            
            return None
            
        except (OSError, FileNotFoundError, RuntimeError) as e:
            error_msg = str(e)
            if "face_recognition_models" in error_msg or "models" in error_msg.lower():
                raise ImportError(
                    "face_recognition models are required. "
                    "Install with: pip install git+https://github.com/ageitgey/face_recognition_models"
                ) from e
            logger.debug(f"Error encoding face from {image_path}: {e}")
            return None
        except Exception as e:
            logger.debug(f"Error encoding face from {image_path}: {e}")
            return None
    
    def identify_faces(self, image_path: str) -> List[Tuple[str, float]]:
        """Identify faces in an image.
        
        Args:
            image_path: Path to image file to analyze
            
        Returns:
            List of tuples (person_name, confidence) for each identified face.
            Confidence is 1.0 - distance (higher is better).
            Faces are returned in order of detection (top to bottom, left to right).
            
        Raises:
            ImportError: If face_recognition models are not installed
        """
        if not self.reference_faces:
            logger.debug("No reference faces loaded, cannot identify faces")
            return []
        
        try:
            # Load image
            image = face_recognition.load_image_file(str(image_path))
            
            # Find all faces in the image
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            # Check if we have any encodings (use len() to avoid NumPy boolean ambiguity)
            if not face_encodings or len(face_encodings) == 0:
                logger.debug(f"No faces detected in {image_path}")
                return []
            
            logger.debug(f"Detected {len(face_encodings)} face(s) in {image_path}")
            
            # Match each face to reference faces
            identified = []
            for face_encoding in face_encodings:
                best_match = None
                best_distance = float('inf')
                
                # Compare with all reference faces
                for person_name, reference_encodings in self.reference_faces.items():
                    # Compare with all encodings for this person (multiple reference images)
                    for ref_encoding in reference_encodings:
                        distance = face_recognition.face_distance(
                            [ref_encoding], 
                            face_encoding
                        )[0]
                        
                        if distance < best_distance:
                            best_distance = distance
                            best_match = person_name
                
                # Check if match is within tolerance
                if best_match and best_distance <= self.tolerance:
                    confidence = 1.0 - (best_distance / self.tolerance)
                    identified.append((best_match, confidence))
                    logger.debug(
                        f"Identified: {best_match} "
                        f"(distance: {best_distance:.3f}, confidence: {confidence:.2f})"
                    )
                else:
                    identified.append(("Unknown", 0.0))
                    logger.debug(
                        f"Unknown face (best match: {best_match}, "
                        f"distance: {best_distance:.3f}, tolerance: {self.tolerance})"
                    )
            
            return identified
            
        except (OSError, FileNotFoundError, RuntimeError) as e:
            error_msg = str(e)
            if "face_recognition_models" in error_msg or "models" in error_msg.lower():
                raise ImportError(
                    "face_recognition models are required. "
                    "Install with: pip install git+https://github.com/ageitgey/face_recognition_models"
                ) from e
            logger.warning(f"Error identifying faces in {image_path}: {e}")
            return []
        except Exception as e:
            logger.warning(f"Error identifying faces in {image_path}: {e}")
            return []
    
    def get_person_names(self, image_path: str, min_confidence: float = 0.5) -> List[str]:
        """Get list of identified person names from an image.
        
        Args:
            image_path: Path to image file to analyze
            min_confidence: Minimum confidence threshold (0.0-1.0)
            
        Returns:
            List of person names (without duplicates, ordered by confidence)
        """
        identified = self.identify_faces(image_path)
        
        # Filter by confidence and extract names
        names = [
            name for name, confidence in identified
            if confidence >= min_confidence and name != "Unknown"
        ]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_names = []
        for name in names:
            if name not in seen:
                seen.add(name)
                unique_names.append(name)
        
        return unique_names
    
    def get_face_count(self, image_path: str) -> int:
        """Get the total number of faces detected in an image.
        
        Args:
            image_path: Path to image file to analyze
            
        Returns:
            Total number of faces detected (including unrecognized ones)
        """
        identified = self.identify_faces(image_path)
        return len(identified)

