"""dlib / ``face_recognition`` face embedding backend.

This is the original smugVision face implementation, unchanged in behaviour and
still the default backend. It produces 128-dimensional float64 encodings compared
by euclidean distance, where a *lower* distance means a better match.

The ``face_recognition`` import is optional and guarded: a missing library (or
missing model files) degrades to ``FACE_RECOGNITION_AVAILABLE = False`` with an
actionable message rather than raising at import time.
"""

import logging
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np

from smugvision.face.backends.base import FaceBackend

# Suppress pkg_resources deprecation warning from face_recognition_models
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*", category=UserWarning)

logger = logging.getLogger(__name__)

# Try to import face_recognition, make it optional
FACE_RECOGNITION_AVAILABLE = False
FACE_RECOGNITION_ERROR = None

try:
    # Suppress pkg_resources deprecation warning before importing
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*pkg_resources is deprecated.*", category=UserWarning
        )
        import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    FACE_RECOGNITION_ERROR = "not_installed"
    logger.warning(
        "face_recognition library not available. Install with: pip install face_recognition"
    )
except SystemExit:
    # face_recognition's api.py calls quit() - which raises SystemExit, NOT
    # ImportError - when face_recognition_models cannot be imported. Without
    # catching it here, importing smugvision.face would terminate the whole
    # process instead of degrading to "face recognition disabled".
    FACE_RECOGNITION_AVAILABLE = False
    FACE_RECOGNITION_ERROR = "models_missing"
    logger.warning(
        "face_recognition models not available. Install with: "
        "pip install git+https://github.com/ageitgey/face_recognition_models"
    )
except Exception as e:  # pragma: no cover - defensive, depends on local install
    FACE_RECOGNITION_AVAILABLE = False
    FACE_RECOGNITION_ERROR = str(e)
    logger.warning(f"face_recognition library could not be imported: {e}")


def _raise_if_models_missing(error: Exception) -> None:
    """Translate a dlib model-file error into an actionable ImportError.

    Args:
        error: The exception raised by the face_recognition call.

    Raises:
        ImportError: If the error looks like missing face_recognition_models.
    """
    error_msg = str(error)
    if "face_recognition_models" in error_msg or "models" in error_msg.lower():
        raise ImportError(
            "face_recognition models are required. "
            "Install with: pip install git+https://github.com/ageitgey/face_recognition_models"
        ) from error


class DlibFaceBackend(FaceBackend):
    """Face embeddings via dlib's ResNet encoder, through the ``face_recognition`` library.

    Reference images are detected with the fast ``hog`` detector (they are expected
    to be clean and front-facing) while images being analyzed use the configured
    ``model`` detector, which handles angles, glasses and shadows better. Both are
    encoded with dlib's ``large`` encoder.

    Attributes:
        tolerance: Maximum euclidean distance still considered a match. Lower is
            stricter. 0.6 is the ``face_recognition`` default.
        model: Detector for images being analyzed - ``'hog'`` (fast) or ``'cnn'``
            (slower, more accurate).
        detection_scale: Downscale factor applied before detection (0.1-1.0). Faces
            are re-encoded from the full-resolution image, so this trades recall on
            small faces for speed without hurting embedding quality.
    """

    name = "dlib"
    embedding_model = "large"
    embedding_dim = 128
    metric = "euclidean"
    normalized = False

    def __init__(
        self,
        tolerance: float = 0.6,
        model: str = "cnn",
        detection_scale: float = 0.5,
    ) -> None:
        """Initialize the dlib backend.

        Args:
            tolerance: Euclidean distance ceiling for a match (lower is stricter).
            model: Face detection model, ``'hog'`` or ``'cnn'``.
            detection_scale: Pre-detection downscale factor, clamped to 0.1-1.0.

        Raises:
            ImportError: If ``face_recognition`` or its model files are unavailable.
        """
        if not FACE_RECOGNITION_AVAILABLE:
            raise ImportError(dlib_unavailable_message())

        self.tolerance = tolerance
        self.model = model
        self.detection_scale = max(0.1, min(1.0, detection_scale))

    def encode_reference(self, image_path: Path) -> Optional[np.ndarray]:
        """Encode a face from a reference image file.

        Args:
            image_path: Path to image file containing a face.

        Returns:
            Face encoding (128-dimensional vector), or None if no face found.

        Raises:
            ImportError: If face_recognition models are not installed.
        """
        try:
            # Load image and handle EXIF orientation
            from PIL import Image, ImageOps

            pil_image = Image.open(str(image_path))
            # Apply EXIF orientation
            pil_image = ImageOps.exif_transpose(pil_image)
            # Convert to RGB if needed
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

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
            # We use the configured model for actual image detection where faces might be
            # at angles/shadows
            face_locations = face_recognition.face_locations(image, model="hog")

            if not face_locations or len(face_locations) == 0:
                return None

            # Get face encodings (use first face if multiple)
            face_encodings = face_recognition.face_encodings(
                image, face_locations, model="large"  # Use large model for better accuracy
            )

            # Check if we have any encodings (use len() to avoid NumPy boolean ambiguity)
            if face_encodings and len(face_encodings) > 0:
                return face_encodings[0]  # Return first face encoding

            return None

        except (OSError, FileNotFoundError, RuntimeError) as e:
            _raise_if_models_missing(e)
            logger.debug(f"Error encoding face from {image_path}: {e}")
            return None
        except Exception as e:
            logger.debug(f"Error encoding face from {image_path}: {e}")
            return None

    def detect_and_embed(self, image_path: Path) -> List[np.ndarray]:
        """Detect and encode every face in an image.

        Args:
            image_path: Path to image file to analyze.

        Returns:
            One 128-dimensional encoding per detected face, in detection order.

        Raises:
            ImportError: If face_recognition models are not installed.
        """
        try:
            # Load image and handle EXIF orientation
            # face_recognition.load_image_file doesn't handle EXIF orientation,
            # so we need to use PIL to load and rotate first
            from PIL import Image, ImageOps

            pil_image = Image.open(str(image_path))
            # Apply EXIF orientation (rotates/flips image if needed)
            pil_image = ImageOps.exif_transpose(pil_image)
            # Convert to RGB if needed (face_recognition requires RGB)
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

            # Resize for faster detection if scale < 1.0
            original_size = pil_image.size
            if self.detection_scale < 1.0:
                detection_size = (
                    int(original_size[0] * self.detection_scale),
                    int(original_size[1] * self.detection_scale),
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
                        int(left * scale_factor),
                    )
                    for top, right, bottom, left in face_locations
                ]
                # Use original image for encoding (better quality)
                original_array = np.array(pil_image)
                face_encodings = face_recognition.face_encodings(
                    original_array, face_locations, model="large"
                )
                del original_array  # Free memory
            else:
                face_encodings = face_recognition.face_encodings(
                    detection_array, face_locations, model="large"
                )

            # Free the detection array
            del detection_array

            # Close the PIL image
            pil_image.close()

            # Check if we have any encodings (use len() to avoid NumPy boolean ambiguity)
            if not face_encodings or len(face_encodings) == 0:
                return []

            return list(face_encodings)

        except (OSError, FileNotFoundError, RuntimeError) as e:
            _raise_if_models_missing(e)
            raise

    def score(self, probe: np.ndarray, reference: np.ndarray) -> float:
        """Score a probe encoding against a reference encoding.

        Implements the historical smugVision confidence formula
        ``1.0 - distance / tolerance``, which is 1.0 for an identical face, exactly
        0.0 at the tolerance boundary, and negative beyond it. Because it is
        strictly decreasing in distance, taking the maximum score picks the same
        reference as taking the minimum distance.

        Args:
            probe: Encoding of a face detected in the analyzed image.
            reference: Encoding of a known reference face.

        Returns:
            Normalized confidence; ``>= 0.0`` iff distance is within tolerance.
        """
        distance = float(face_recognition.face_distance([reference], probe)[0])

        if self.tolerance <= 0.0:
            # Degenerate configuration: only an exact match can qualify.
            return 0.0 if distance == 0.0 else -1.0

        return 1.0 - (distance / self.tolerance)


def dlib_unavailable_message() -> str:
    """Build the actionable error message for an unusable dlib backend.

    Returns:
        Installation guidance matching the specific failure that was detected.
    """
    if FACE_RECOGNITION_ERROR == "models_missing":
        return (
            "face_recognition models are required. "
            "Install with: pip install git+https://github.com/ageitgey/face_recognition_models"
        )
    if FACE_RECOGNITION_ERROR == "not_installed":
        return (
            "face_recognition library is required. "
            "Install with: pip install face_recognition\n"
            "Then install models with: "
            "pip install git+https://github.com/ageitgey/face_recognition_models"
        )
    return (
        f"face_recognition library error: {FACE_RECOGNITION_ERROR}\n"
        "Make sure face_recognition and models are installed:\n"
        "  pip install face_recognition\n"
        "  pip install git+https://github.com/ageitgey/face_recognition_models"
    )
