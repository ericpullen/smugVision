"""Face recognition using reference faces."""

import hashlib
import json
import logging
import pickle
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Suppress pkg_resources deprecation warning from face_recognition_models
warnings.filterwarnings('ignore', message='.*pkg_resources is deprecated.*', category=UserWarning)

logger = logging.getLogger(__name__)

# Cache version - increment this when encoding format changes
CACHE_VERSION = 1

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
        model: Face detection model to use ('hog' or 'cnn')
    """
    
    def __init__(
        self,
        reference_faces_dir: Optional[str] = None,
        tolerance: float = 0.6,
        model: str = "cnn",
        detection_scale: float = 0.5,
        cache_dir: Optional[str] = None,
        use_cache: bool = True
    ) -> None:
        """Initialize face recognizer.
        
        Args:
            reference_faces_dir: Directory containing person subdirectories with reference images.
            tolerance: Face recognition tolerance (0.0-1.0). Lower is stricter.
                      Default 0.6 is a good balance.
            model: Face detection model - 'hog' (faster, less accurate) or 'cnn' (slower, more accurate).
                   Default 'cnn' is recommended for better detection with glasses, shadows, angles.
            detection_scale: Scale factor for resizing images before face detection (0.0-1.0).
                           Lower values = faster but may miss small/distant faces.
                           Default 0.5 gives 3-4x speedup with minimal accuracy loss.
            cache_dir: Directory for storing face encoding cache. Defaults to 
                      ~/.smugvision/cache/face_encodings if not specified.
            use_cache: Whether to use caching for reference face encodings. Default True.
        
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
        self.model = model
        self.detection_scale = max(0.1, min(1.0, detection_scale))  # Clamp between 0.1 and 1.0
        self.use_cache = use_cache
        
        # Set up cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".smugvision" / "cache" / "face_encodings"
        
        if reference_faces_dir:
            self.load_reference_faces(reference_faces_dir)
    
    def _get_file_fingerprint(self, file_path: Path) -> str:
        """Get a fingerprint for a file based on path, size, and modification time.
        
        Args:
            file_path: Path to the file
            
        Returns:
            String fingerprint for cache invalidation
        """
        stat = file_path.stat()
        # Use path, size, and mtime for fingerprint
        fingerprint_data = f"{file_path}|{stat.st_size}|{stat.st_mtime}"
        return hashlib.md5(fingerprint_data.encode()).hexdigest()
    
    def _get_cache_paths(self, ref_dir: Path) -> Tuple[Path, Path]:
        """Get paths for cache files.
        
        Args:
            ref_dir: Reference faces directory
            
        Returns:
            Tuple of (encodings_path, manifest_path)
        """
        # Create a unique cache name based on the reference directory path
        dir_hash = hashlib.md5(str(ref_dir.resolve()).encode()).hexdigest()[:12]
        cache_name = f"face_encodings_{dir_hash}"
        
        encodings_path = self.cache_dir / f"{cache_name}.pkl"
        manifest_path = self.cache_dir / f"{cache_name}_manifest.json"
        
        return encodings_path, manifest_path
    
    def _load_cache(self, ref_dir: Path) -> Tuple[Dict[str, Any], Dict[str, List]]:
        """Load cached face encodings and manifest.
        
        Args:
            ref_dir: Reference faces directory
            
        Returns:
            Tuple of (manifest_dict, encodings_dict)
            Returns empty dicts if cache doesn't exist or is invalid
        """
        encodings_path, manifest_path = self._get_cache_paths(ref_dir)
        
        if not encodings_path.exists() or not manifest_path.exists():
            logger.debug("No cache found")
            return {}, {}
        
        try:
            # Load manifest
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            # Check cache version
            if manifest.get('version') != CACHE_VERSION:
                logger.info(f"Cache version mismatch (expected {CACHE_VERSION}, got {manifest.get('version')}), rebuilding")
                return {}, {}
            
            # Load encodings
            with open(encodings_path, 'rb') as f:
                encodings = pickle.load(f)
            
            logger.debug(f"Loaded cache with {len(manifest.get('files', {}))} file entries")
            return manifest, encodings
            
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return {}, {}
    
    def _save_cache(
        self,
        ref_dir: Path,
        manifest: Dict[str, Any],
        encodings: Dict[str, List]
    ) -> None:
        """Save face encodings and manifest to cache.
        
        Args:
            ref_dir: Reference faces directory
            manifest: File fingerprint manifest
            encodings: Face encodings by person name
        """
        try:
            # Ensure cache directory exists
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            encodings_path, manifest_path = self._get_cache_paths(ref_dir)
            
            # Save manifest
            manifest['version'] = CACHE_VERSION
            manifest['created'] = time.time()
            manifest['ref_dir'] = str(ref_dir.resolve())
            
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            # Save encodings
            with open(encodings_path, 'wb') as f:
                pickle.dump(encodings, f)
            
            logger.debug(f"Saved cache to {encodings_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def load_reference_faces(self, directory: str) -> None:
        """Load reference face images from a directory.
        
        Each subdirectory should be named after a person, containing their reference images.
        All image files within a person's directory will be loaded as reference faces.
        
        Uses caching to avoid re-encoding unchanged images. The cache stores face encodings
        and tracks file fingerprints (path, size, mtime) to detect changes.
        
        Example directory structure:
            faces/
                John_Doe/
                    photo1.jpg
                    photo2.jpg
                    vacation.png
                Jane_Smith/
                    profile.jpg
                    headshot.heic
        
        Args:
            directory: Path to directory containing person subdirectories with reference images
        """
        ref_dir = Path(directory)
        if not ref_dir.exists():
            logger.warning(f"Reference faces directory not found: {directory}")
            return
        
        start_time = time.time()
        logger.info(f"Loading reference faces from: {directory}")
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.heic', '.heif'}
        
        # Load existing cache
        cached_manifest, cached_encodings = {}, {}
        if self.use_cache:
            cached_manifest, cached_encodings = self._load_cache(ref_dir)
        
        cached_files = cached_manifest.get('files', {})
        
        # Track what we're loading
        new_manifest_files = {}
        loaded_from_cache = 0
        encoded_fresh = 0
        
        # Iterate through subdirectories (each is a person)
        for person_dir in ref_dir.iterdir():
            if not person_dir.is_dir():
                # Skip files in the root directory
                continue
            
            person_name = person_dir.name
            person_encodings = []
            
            # Load all images from this person's directory
            for image_path in person_dir.iterdir():
                if not image_path.is_file():
                    continue
                
                if image_path.suffix.lower() not in image_extensions:
                    logger.debug(f"Skipping non-image file: {image_path.name}")
                    continue
                
                # Get file fingerprint for cache lookup
                file_key = str(image_path.resolve())
                current_fingerprint = self._get_file_fingerprint(image_path)
                
                # Check if we have a valid cached encoding
                cached_fingerprint = cached_files.get(file_key, {}).get('fingerprint')
                cached_person = cached_files.get(file_key, {}).get('person')
                
                if (self.use_cache and 
                    cached_fingerprint == current_fingerprint and 
                    cached_person == person_name and
                    file_key in cached_encodings):
                    # Use cached encoding
                    encoding = cached_encodings[file_key]
                    person_encodings.append(encoding)
                    loaded_from_cache += 1
                    logger.debug(f"Loaded cached encoding for {person_name}/{image_path.name}")
                else:
                    # Need to encode fresh
                    try:
                        encoding = self._encode_face(image_path)
                        if encoding is not None:
                            person_encodings.append(encoding)
                            # Store in cache
                            if self.use_cache:
                                cached_encodings[file_key] = encoding
                            encoded_fresh += 1
                            logger.debug(f"Encoded fresh face for {person_name}/{image_path.name}")
                        else:
                            logger.warning(f"No face found in reference image: {image_path}")
                            continue  # Don't add to manifest if no face found
                    except Exception as e:
                        logger.warning(f"Failed to load reference face from {image_path}: {e}")
                        continue
                
                # Update manifest
                new_manifest_files[file_key] = {
                    'fingerprint': current_fingerprint,
                    'person': person_name,
                    'filename': image_path.name
                }
            
            if person_encodings:
                self.reference_faces[person_name] = person_encodings
                logger.debug(f"Loaded {len(person_encodings)} reference face(s) for {person_name}")
        
        # Save updated cache
        if self.use_cache and (encoded_fresh > 0 or len(new_manifest_files) != len(cached_files)):
            # Clean up cached encodings for files that no longer exist
            valid_keys = set(new_manifest_files.keys())
            cached_encodings = {k: v for k, v in cached_encodings.items() if k in valid_keys}
            
            self._save_cache(ref_dir, {'files': new_manifest_files}, cached_encodings)
        
        elapsed = time.time() - start_time
        total_faces = sum(len(faces) for faces in self.reference_faces.values())
        
        cache_status = ""
        if self.use_cache:
            cache_status = f" ({loaded_from_cache} from cache, {encoded_fresh} newly encoded)"
        
        logger.info(
            f"Loaded {total_faces} reference face(s) for "
            f"{len(self.reference_faces)} person(s) in {elapsed:.2f}s{cache_status}"
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
            # Load image and handle EXIF orientation
            from PIL import Image, ImageOps
            import numpy as np
            
            pil_image = Image.open(str(image_path))
            # Apply EXIF orientation
            pil_image = ImageOps.exif_transpose(pil_image)
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Resize reference images to reasonable size for faster processing
            # Reference faces should be clean/front-facing, so we can use smaller size
            max_dimension = 800
            if max(pil_image.size) > max_dimension:
                # Calculate new size maintaining aspect ratio
                ratio = max_dimension / max(pil_image.size)
                new_size = (int(pil_image.size[0] * ratio), int(pil_image.size[1] * ratio))
                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            image = np.array(pil_image)
            
            # Use HOG (faster) for reference faces since they should be clean, front-facing
            # We use CNN for actual image detection where faces might be at angles/shadows
            face_locations = face_recognition.face_locations(image, model="hog")
            
            if not face_locations or len(face_locations) == 0:
                return None
            
            # Get face encodings (use first face if multiple)
            face_encodings = face_recognition.face_encodings(
                image, 
                face_locations,
                model="large"  # Use large model for better accuracy
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
            # Load image and handle EXIF orientation
            # face_recognition.load_image_file doesn't handle EXIF orientation,
            # so we need to use PIL to load and rotate first
            from PIL import Image, ImageOps
            import numpy as np
            
            pil_image = Image.open(str(image_path))
            # Apply EXIF orientation (rotates/flips image if needed)
            pil_image = ImageOps.exif_transpose(pil_image)
            # Convert to RGB if needed (face_recognition requires RGB)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Resize for faster detection if scale < 1.0
            original_size = pil_image.size
            if self.detection_scale < 1.0:
                detection_size = (
                    int(original_size[0] * self.detection_scale),
                    int(original_size[1] * self.detection_scale)
                )
                detection_image = pil_image.resize(detection_size, Image.Resampling.LANCZOS)
            else:
                detection_image = pil_image
            
            # Convert to numpy array for face_recognition
            detection_array = np.array(detection_image)
            
            # Close the detection image if it's different from original
            if detection_image is not pil_image:
                detection_image.close()
            
            # Find all faces in the (possibly downscaled) image using configured model
            face_locations = face_recognition.face_locations(detection_array, model=self.model)
            
            # Scale face locations back to original size if we downscaled
            if self.detection_scale < 1.0 and face_locations:
                scale_factor = 1.0 / self.detection_scale
                face_locations = [
                    (
                        int(top * scale_factor),
                        int(right * scale_factor),
                        int(bottom * scale_factor),
                        int(left * scale_factor)
                    )
                    for top, right, bottom, left in face_locations
                ]
                # Use original image for encoding (better quality)
                original_array = np.array(pil_image)
                face_encodings = face_recognition.face_encodings(original_array, face_locations, model="large")
                del original_array  # Free memory
            else:
                face_encodings = face_recognition.face_encodings(detection_array, face_locations, model="large")
            
            # Free the detection array
            del detection_array
            
            # Close the PIL image
            pil_image.close()
            
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
    
    def get_person_names(self, image_path: str, min_confidence: float = 0.25) -> List[str]:
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
    
    def clear_cache(self) -> bool:
        """Clear the face encoding cache.
        
        Returns:
            True if cache was cleared, False if no cache existed or error occurred
        """
        try:
            if not self.cache_dir.exists():
                logger.debug("No cache directory to clear")
                return False
            
            cleared = False
            for cache_file in self.cache_dir.glob("face_encodings_*"):
                cache_file.unlink()
                cleared = True
                logger.debug(f"Deleted cache file: {cache_file}")
            
            if cleared:
                logger.info("Face encoding cache cleared")
            return cleared
            
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")
            return False
    
    def get_cache_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the current cache.
        
        Returns:
            Dictionary with cache info, or None if no cache exists
        """
        try:
            # Find any manifest files
            if not self.cache_dir.exists():
                return None
            
            manifests = list(self.cache_dir.glob("face_encodings_*_manifest.json"))
            if not manifests:
                return None
            
            info = {
                'cache_dir': str(self.cache_dir),
                'manifests': []
            }
            
            for manifest_path in manifests:
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                
                encodings_path = manifest_path.with_name(
                    manifest_path.name.replace('_manifest.json', '.pkl')
                )
                
                info['manifests'].append({
                    'ref_dir': manifest.get('ref_dir'),
                    'version': manifest.get('version'),
                    'created': manifest.get('created'),
                    'file_count': len(manifest.get('files', {})),
                    'cache_size_bytes': encodings_path.stat().st_size if encodings_path.exists() else 0
                })
            
            return info
            
        except Exception as e:
            logger.warning(f"Failed to get cache info: {e}")
            return None

