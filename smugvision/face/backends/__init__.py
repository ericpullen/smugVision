"""Pluggable face embedding backends.

Importing this package pulls in the dlib backend only. The InsightFace backend is
loaded lazily by :func:`create_backend`, so machines without ``insightface`` /
``onnxruntime`` installed - and machines that simply are not using it - never pay
the multi-second ONNX Runtime import cost.

Backends are selected by name from config (``face_recognition.backend``). Unlike a
strict allow-list, unknown names produce an error that lists what is valid rather
than silently mapping to a default implementation.
"""

import importlib.util
import logging
from typing import Callable, Dict, List, Type

from smugvision.face.backends.base import FaceBackend
from smugvision.face.backends.dlib_backend import (
    FACE_RECOGNITION_AVAILABLE,
    FACE_RECOGNITION_ERROR,
    DlibFaceBackend,
    dlib_unavailable_message,
)

logger = logging.getLogger(__name__)

#: Backend used when none is configured. dlib remains the default so that
#: upgrading smugVision never silently changes recognition results.
DEFAULT_BACKEND = "dlib"


def _load_dlib_backend() -> Type[FaceBackend]:
    """Return the dlib backend class.

    Returns:
        The DlibFaceBackend class.
    """
    return DlibFaceBackend


def _load_insightface_backend() -> Type[FaceBackend]:
    """Import and return the InsightFace backend class.

    Deferred so that ``insightface`` and ``onnxruntime`` are only imported when the
    backend is actually requested.

    Returns:
        The InsightFaceBackend class.

    Raises:
        ImportError: If insightface / onnxruntime are not installed.
    """
    from smugvision.face.backends.insightface_backend import (
        INSIGHTFACE_AVAILABLE,
        InsightFaceBackend,
        insightface_unavailable_message,
    )

    if not INSIGHTFACE_AVAILABLE:
        raise ImportError(insightface_unavailable_message())

    return InsightFaceBackend


_BACKEND_LOADERS: Dict[str, Callable[[], Type[FaceBackend]]] = {
    "dlib": _load_dlib_backend,
    "insightface": _load_insightface_backend,
}


def list_backends() -> List[str]:
    """List the names of all known backends.

    Returns:
        Sorted backend names, including any registered via :func:`register_backend`.
    """
    return sorted(_BACKEND_LOADERS)


def insightface_available() -> bool:
    """Check whether the InsightFace backend could be used, without importing it.

    Uses module-spec lookup rather than a real import so that this is cheap enough
    to call at startup. A True result means the packages are installed; the model
    pack download and ONNX session creation can still fail later, which is why
    :func:`create_backend` raises a detailed ImportError of its own.

    Returns:
        True if both insightface and onnxruntime appear to be installed.
    """
    try:
        return (
            importlib.util.find_spec("insightface") is not None
            and importlib.util.find_spec("onnxruntime") is not None
        )
    except (ImportError, ValueError):  # pragma: no cover - malformed install
        return False


def backend_available(name: str) -> bool:
    """Check whether a named backend's dependencies are installed.

    Args:
        name: Backend name, case-insensitive.

    Returns:
        True if the backend is known and its dependencies appear usable.
    """
    key = (name or "").strip().lower()
    if key == "dlib":
        return FACE_RECOGNITION_AVAILABLE
    if key == "insightface":
        return insightface_available()
    return key in _BACKEND_LOADERS


def register_backend(name: str, backend_class: Type[FaceBackend]) -> None:
    """Register a custom face backend implementation.

    The encoding cache is namespaced by the backend's *declared* identity - its
    ``name``, ``embedding_model`` and ``embedding_dim`` - not by the name it is
    registered under. A custom backend must therefore declare an identity distinct
    from the built-ins, or it will share (and overwrite) their cached vectors.

    Args:
        name: Name to register the backend under (case-insensitive).
        backend_class: A FaceBackend subclass.

    Raises:
        TypeError: If backend_class is not a FaceBackend subclass.
    """
    if not (isinstance(backend_class, type) and issubclass(backend_class, FaceBackend)):
        raise TypeError(f"{backend_class!r} is not a FaceBackend subclass")

    key = name.strip().lower()
    _BACKEND_LOADERS[key] = lambda cls=backend_class: cls  # type: ignore[misc]
    logger.debug(f"Registered face backend: {key} -> {backend_class.__name__}")


def create_backend(name: str = DEFAULT_BACKEND, **kwargs) -> FaceBackend:
    """Create a face embedding backend by name.

    Args:
        name: Backend name - ``"dlib"`` (default) or ``"insightface"``.
        **kwargs: Passed through to the backend constructor. dlib accepts
            tolerance / model / detection_scale; insightface accepts model_name /
            det_size / similarity_threshold / providers.

    Returns:
        An initialized FaceBackend. For InsightFace no model is loaded yet; the
        ONNX session is built on first use.

    Raises:
        ValueError: If the backend name is not recognized.
        ImportError: If the backend's dependencies or models are unavailable.
    """
    key = (name or DEFAULT_BACKEND).strip().lower()

    loader = _BACKEND_LOADERS.get(key)
    if loader is None:
        raise ValueError(
            f"Unknown face recognition backend: {name!r}. "
            f"Valid backends are: {', '.join(list_backends())}"
        )

    backend_class = loader()
    return backend_class(**kwargs)


__all__ = [
    "DEFAULT_BACKEND",
    "FACE_RECOGNITION_AVAILABLE",
    "FACE_RECOGNITION_ERROR",
    "DlibFaceBackend",
    "FaceBackend",
    "backend_available",
    "create_backend",
    "dlib_unavailable_message",
    "insightface_available",
    "list_backends",
    "register_backend",
]
