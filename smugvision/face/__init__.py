"""Face recognition module for identifying people in images.

:class:`FaceRecognizer` is the entry point. Embedding backends (dlib by default,
InsightFace optionally) live in :mod:`smugvision.face.backends`; importing this
package does not import the optional InsightFace dependencies.
"""

from smugvision.face.backends import FaceBackend, create_backend, register_backend
from smugvision.face.recognizer import (
    CACHE_VERSION,
    FACE_RECOGNITION_AVAILABLE,
    INSIGHTFACE_AVAILABLE,
    FaceRecognizer,
)

__all__ = [
    "FaceRecognizer",
    "FaceBackend",
    "create_backend",
    "register_backend",
    "CACHE_VERSION",
    "FACE_RECOGNITION_AVAILABLE",
    "INSIGHTFACE_AVAILABLE",
]
