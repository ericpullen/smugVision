"""Face recognition using reference faces.

:class:`FaceRecognizer` is the single public entry point. It stays backend-agnostic
and owns the parts that are the same whatever produces the embeddings: the
reference-face map, the on-disk encoding cache, and the "names above a confidence
threshold" contract. Detection, embedding and scoring live in
:mod:`smugvision.face.backends`.

Two backends ship today, selected by the ``face_recognition.backend`` config key:

* ``"dlib"`` (default) - the original ``face_recognition`` implementation.
  128-d encodings, euclidean distance, lower is better.
* ``"insightface"`` - ArcFace ONNX embeddings, better on profile/angle/age
  variation. 512-d L2-normalized embeddings, cosine similarity, higher is better.

Those vectors are mathematically incomparable, so the cache is namespaced per
backend and validated against the live backend's signature on load. See
:meth:`FaceRecognizer._load_cache`.
"""

import hashlib
import json
import logging
import pickle
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from smugvision.face.backends import (
    DEFAULT_BACKEND,
    FACE_RECOGNITION_AVAILABLE,
    FACE_RECOGNITION_ERROR,
    FaceBackend,
    create_backend,
    dlib_unavailable_message,
    insightface_available,
)

# Suppress pkg_resources deprecation warning from face_recognition_models
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*", category=UserWarning)

logger = logging.getLogger(__name__)

# Cache version - increment this when encoding format changes.
# v2: encodings became backend-namespaced. The cache filename now carries the
# backend, embedding model and dimensionality, and the manifest records the full
# embedding signature, so dlib's 128-d euclidean vectors can never be compared
# against InsightFace's 512-d cosine vectors. v1 caches carry no backend marker at
# all and so cannot be migrated - they are simply ignored and re-encoded.
CACHE_VERSION = 2

# Whether the optional InsightFace backend's dependencies are installed. Computed
# by module-spec lookup, so reading this does NOT import insightface/onnxruntime.
# Mirrors FACE_RECOGNITION_AVAILABLE, which is re-exported from the dlib backend
# for backward compatibility with code that imported it from this module.
INSIGHTFACE_AVAILABLE = insightface_available()

# Option keys each built-in backend accepts via `backend_options`.
_BACKEND_OPTION_KEYS: Dict[str, set] = {
    "dlib": {"tolerance", "model", "detection_scale"},
    "insightface": {"model_name", "det_size", "similarity_threshold", "providers"},
}

# dlib-only settings, ignored when another backend is active.
_DLIB_ONLY_SETTINGS = ("tolerance", "model", "detection_scale")

__all__ = [
    "CACHE_VERSION",
    "FACE_RECOGNITION_AVAILABLE",
    "FACE_RECOGNITION_ERROR",
    "INSIGHTFACE_AVAILABLE",
    "FaceRecognizer",
]


class FaceRecognizer:
    """Face recognition system using reference faces.

    This class loads reference face images and can identify people in new images
    by comparing detected faces to the reference set.

    Attributes:
        reference_faces: Dictionary mapping person names to face encodings. Names are
            the reference subdirectory names verbatim, so underscores are preserved
            (``John_Doe``); callers that display names convert them.
        tolerance: dlib euclidean distance ceiling for a match (lower = stricter).
            Retained on the instance whatever backend is active, because it is the
            dlib interpretation and diagnostic scripts read it.
        model: dlib face detection model to use ('hog' or 'cnn').
        detection_scale: dlib pre-detection downscale factor.
        backend: The active :class:`~smugvision.face.backends.base.FaceBackend`.
        backend_name: Name of the active backend, e.g. ``"dlib"``.
    """

    def __init__(
        self,
        reference_faces_dir: Optional[str] = None,
        tolerance: float = 0.6,
        model: str = "cnn",
        detection_scale: float = 0.5,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        *,
        backend: str = DEFAULT_BACKEND,
        backend_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize face recognizer.

        Args:
            reference_faces_dir: Directory containing person subdirectories with
                reference images.
            tolerance: dlib face recognition tolerance (0.0-1.0). Lower is stricter.
                Default 0.6 is a good balance. **dlib only** - it is a euclidean
                distance ceiling and has no meaning for cosine-similarity backends,
                which take their own threshold via ``backend_options``.
            model: dlib face detection model - 'hog' (faster, less accurate) or 'cnn'
                (slower, more accurate). Default 'cnn' is recommended for better
                detection with glasses, shadows, angles. **dlib only.**
            detection_scale: Scale factor for resizing images before face detection
                (0.0-1.0). Lower values = faster but may miss small/distant faces.
                Default 0.5 gives 3-4x speedup with minimal accuracy loss.
                **dlib only** - InsightFace resizes internally to its ``det_size``.
            cache_dir: Directory for storing face encoding cache. Defaults to
                ~/.smugvision/cache/face_encodings if not specified.
            use_cache: Whether to use caching for reference face encodings. Default True.
            backend: Embedding backend - ``"dlib"`` (default) or ``"insightface"``.
                An unavailable or unknown backend logs an actionable error and falls
                back to dlib; if dlib is unavailable too, ImportError is raised so
                the caller can disable face recognition.
            backend_options: Extra backend-specific settings. For ``insightface``:
                ``model_name`` (default ``"buffalo_l"``), ``det_size`` (default
                ``(640, 640)``), ``similarity_threshold`` (default ``0.4``, cosine
                similarity, higher is stricter) and ``providers``. For ``dlib`` these
                may override tolerance / model / detection_scale. Unknown keys are
                ignored with a debug log.

        Raises:
            ImportError: If no usable backend is available - for dlib, when the
                face_recognition library or its models are missing.
        """
        self.reference_faces: Dict[str, List[np.ndarray]] = {}

        # These three keep the dlib interpretation regardless of the active backend,
        # both for backward compatibility and because diagnostic tooling reads them
        # as euclidean-distance settings.
        self.tolerance = tolerance
        self.model = model
        self.detection_scale = max(0.1, min(1.0, detection_scale))  # Clamp between 0.1 and 1.0
        self.use_cache = use_cache

        # Set up cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".smugvision" / "cache" / "face_encodings"

        # Single-entry memo so get_person_names() followed by get_face_count() on the
        # same image does not run detection twice. Keyed on path + size + mtime.
        self._memo_key: Optional[Tuple[str, int, float]] = None
        self._memo_value: Optional[List[Tuple[str, float]]] = None

        self.backend: FaceBackend = self._select_backend(backend, backend_options or {})
        self.backend_name: str = self.backend.name

        if reference_faces_dir:
            self.load_reference_faces(reference_faces_dir)

    def _select_backend(self, requested: str, options: Dict[str, Any]) -> FaceBackend:
        """Create the embedding backend, degrading gracefully when unavailable.

        Selecting a backend whose dependencies are missing is a configuration
        problem, not a reason to kill the run: it logs an actionable error and falls
        back to dlib. Only when no backend at all can be built does this raise, which
        callers treat as "face recognition disabled".

        Args:
            requested: Requested backend name.
            options: Backend-specific options from ``backend_options``.

        Returns:
            An initialized FaceBackend.

        Raises:
            ImportError: If neither the requested backend nor dlib is usable.
        """
        name = (requested or DEFAULT_BACKEND).strip().lower()

        candidates = [name]
        if name != DEFAULT_BACKEND:
            candidates.append(DEFAULT_BACKEND)

        last_error: Optional[Exception] = None
        for index, candidate in enumerate(candidates):
            try:
                backend = create_backend(candidate, **self._backend_kwargs(candidate, options))
            except (ImportError, ValueError, TypeError) as e:
                last_error = e
                if index == 0:
                    # Actionable, at error level: the user asked for this explicitly.
                    logger.error(f"Face recognition backend {candidate!r} is unavailable: {e}")
                    if len(candidates) > 1:
                        logger.warning(
                            f"Falling back to the {DEFAULT_BACKEND!r} face recognition backend"
                        )
                continue

            if candidate != name:
                logger.warning(f"Using {candidate!r} face recognition backend instead of {name!r}")
            else:
                logger.info(f"Face recognition backend: {backend.describe()}")

            if candidate != "dlib":
                ignored = [s for s in _DLIB_ONLY_SETTINGS if s in options]
                logger.debug(
                    f"dlib-only settings are not used by the {candidate!r} backend: "
                    f"{', '.join(_DLIB_ONLY_SETTINGS)}"
                    + (f" (explicitly supplied: {', '.join(ignored)})" if ignored else "")
                )

            return backend

        # Nothing worked. Surface as ImportError so callers disable face recognition.
        if isinstance(last_error, ImportError):
            raise last_error
        raise ImportError(
            f"No usable face recognition backend (requested {requested!r}): {last_error}\n"
            f"{dlib_unavailable_message()}"
        )

    def _backend_kwargs(self, backend_name: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Build constructor kwargs for a backend.

        Args:
            backend_name: Name of the backend being constructed.
            options: Backend-specific options from ``backend_options``.

        Returns:
            Keyword arguments to pass to the backend constructor.
        """
        known = _BACKEND_OPTION_KEYS.get(backend_name)

        if backend_name == "dlib":
            kwargs: Dict[str, Any] = {
                "tolerance": self.tolerance,
                "model": self.model,
                "detection_scale": self.detection_scale,
            }
        else:
            kwargs = {}

        if known is None:
            # Custom backend registered by a caller: pass everything through.
            kwargs.update(options)
            return kwargs

        for key, value in options.items():
            if key in known:
                kwargs[key] = value
            elif backend_name == "dlib" or key not in _DLIB_ONLY_SETTINGS:
                logger.debug(f"Ignoring unsupported {backend_name} backend option: {key}")

        return kwargs

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

        The filename carries the backend slug as well as the reference directory
        hash, so each backend keeps its own cache. Switching backends therefore
        neither reads nor destroys the other's encodings. The ``face_encodings_``
        prefix is preserved so :meth:`clear_cache` and :meth:`get_cache_info` keep
        matching, including legacy v1 files.

        Args:
            ref_dir: Reference faces directory

        Returns:
            Tuple of (encodings_path, manifest_path)
        """
        # Create a unique cache name based on the reference directory path
        dir_hash = hashlib.md5(str(ref_dir.resolve()).encode()).hexdigest()[:12]
        cache_name = f"face_encodings_{self.backend.cache_slug()}_{dir_hash}"

        encodings_path = self.cache_dir / f"{cache_name}.pkl"
        manifest_path = self.cache_dir / f"{cache_name}_manifest.json"

        return encodings_path, manifest_path

    def _load_cache(self, ref_dir: Path) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        """Load cached face encodings and manifest.

        The cache is rejected wholesale unless the version *and* every field of the
        embedding signature (backend, embedding model, dimensionality, metric,
        normalization) match the live backend. Version alone is not enough, because
        v1 manifests predate backends entirely and carry no such marker.

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
            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            # Check cache version
            if manifest.get("version") != CACHE_VERSION:
                logger.info(
                    f"Cache version mismatch (expected {CACHE_VERSION}, "
                    f"got {manifest.get('version')}), rebuilding"
                )
                return {}, {}

            # Check the embedding format matches this backend exactly
            expected = self.backend.cache_signature()
            stored = {key: manifest.get(key) for key in expected}
            if stored != expected:
                logger.warning(
                    f"Cache embedding signature mismatch (expected {expected}, got {stored}), "
                    "rebuilding rather than risk comparing incompatible vectors"
                )
                return {}, {}

            # Load encodings
            with open(encodings_path, "rb") as f:
                encodings = pickle.load(f)

            logger.debug(f"Loaded cache with {len(manifest.get('files', {}))} file entries")
            return manifest, encodings

        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return {}, {}

    def _save_cache(
        self, ref_dir: Path, manifest: Dict[str, Any], encodings: Dict[str, np.ndarray]
    ) -> None:
        """Save face encodings and manifest to cache.

        Args:
            ref_dir: Reference faces directory
            manifest: File fingerprint manifest
            encodings: Face encodings keyed by absolute reference image path
        """
        try:
            # Ensure cache directory exists
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            encodings_path, manifest_path = self._get_cache_paths(ref_dir)

            # Save manifest
            manifest["version"] = CACHE_VERSION
            manifest["created"] = time.time()
            manifest["ref_dir"] = str(ref_dir.resolve())
            # Embedding signature - checked on load to prevent cross-backend reads
            manifest.update(self.backend.cache_signature())

            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

            # Save encodings
            with open(encodings_path, "wb") as f:
                pickle.dump(encodings, f)

            logger.debug(f"Saved cache to {encodings_path}")

        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def load_reference_faces(self, directory: str) -> None:
        """Load reference face images from a directory.

        Each subdirectory should be named after a person, containing their reference images.
        All image files within a person's directory will be loaded as reference faces.

        Person names are taken from the subdirectory name verbatim, underscores
        included; converting ``John_Doe`` to "John Doe" is the caller's job.

        Uses caching to avoid re-encoding unchanged images. The cache stores face encodings
        and tracks file fingerprints (path, size, mtime) to detect changes, and is
        namespaced per backend so encodings are never reused across backends.

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
            directory: Path to directory containing person subdirectories with reference
                images. A leading ``~`` is expanded.
        """
        ref_dir = Path(directory).expanduser()
        if not ref_dir.exists():
            logger.warning(f"Reference faces directory not found: {directory}")
            return

        start_time = time.time()
        logger.info(f"Loading reference faces from: {directory}")

        # Reset state so a repeat call replaces rather than duplicates encodings
        self.reference_faces = {}
        self._invalidate_memo()

        # Supported image extensions
        image_extensions = {".jpg", ".jpeg", ".png", ".heic", ".heif"}

        # Load existing cache
        cached_manifest: Dict[str, Any] = {}
        cached_encodings: Dict[str, np.ndarray] = {}
        if self.use_cache:
            cached_manifest, cached_encodings = self._load_cache(ref_dir)

        cached_files = cached_manifest.get("files", {})

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
                cached_fingerprint = cached_files.get(file_key, {}).get("fingerprint")
                cached_person = cached_files.get(file_key, {}).get("person")

                if (
                    self.use_cache
                    and cached_fingerprint == current_fingerprint
                    and cached_person == person_name
                    and file_key in cached_encodings
                ):
                    # Use cached encoding
                    person_encodings.append(cached_encodings[file_key])
                    loaded_from_cache += 1
                    logger.debug(f"Loaded cached encoding for {person_name}/{image_path.name}")
                else:
                    # Need to encode fresh
                    try:
                        encoding: Optional[np.ndarray] = self.backend.encode_reference(image_path)
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
                    "fingerprint": current_fingerprint,
                    "person": person_name,
                    "filename": image_path.name,
                }

            if person_encodings:
                self.reference_faces[person_name] = person_encodings
                logger.debug(f"Loaded {len(person_encodings)} reference face(s) for {person_name}")

        # Save updated cache
        if self.use_cache and (encoded_fresh > 0 or len(new_manifest_files) != len(cached_files)):
            # Clean up cached encodings for files that no longer exist
            valid_keys = set(new_manifest_files.keys())
            cached_encodings = {k: v for k, v in cached_encodings.items() if k in valid_keys}

            self._save_cache(ref_dir, {"files": new_manifest_files}, cached_encodings)

        elapsed = time.time() - start_time
        total_faces = sum(len(faces) for faces in self.reference_faces.values())

        cache_status = ""
        if self.use_cache:
            cache_status = f" ({loaded_from_cache} from cache, {encoded_fresh} newly encoded)"

        logger.info(
            f"Loaded {total_faces} reference face(s) for "
            f"{len(self.reference_faces)} person(s) in {elapsed:.2f}s{cache_status}"
        )

    def _encode_face(self, image_path: str) -> Optional[np.ndarray]:
        """Encode a face from an image file using the active backend.

        Args:
            image_path: Path to image file containing a face

        Returns:
            Face embedding vector, or None if no face found

        Raises:
            ImportError: If the backend's model files are not installed
        """
        return self.backend.encode_reference(Path(image_path))

    def _invalidate_memo(self) -> None:
        """Drop the single-entry detection memo."""
        self._memo_key = None
        self._memo_value = None

    def _memo_fingerprint(self, image_path: str) -> Optional[Tuple[str, int, float]]:
        """Build a memo key for an image from its path, size and mtime.

        Args:
            image_path: Path to the image file

        Returns:
            A (path, size, mtime) tuple, or None if the file cannot be stat'ed.
        """
        try:
            path = Path(image_path)
            stat = path.stat()
            return (str(path.resolve()), stat.st_size, stat.st_mtime)
        except OSError:
            return None

    def identify_faces(self, image_path: str) -> List[Tuple[str, float]]:
        """Identify faces in an image.

        Results are memoized for the most recently analyzed image, so calling
        :meth:`get_person_names` and :meth:`get_face_count` back to back on the same
        file runs detection once. The memo is keyed on path, size and mtime, so an
        edited file is re-analyzed.

        Args:
            image_path: Path to image file to analyze

        Returns:
            List of tuples (person_name, confidence) for each *detected* face -
            unmatched faces appear as ("Unknown", 0.0), so the length of this list is
            the detected-face count. Confidence is normalized 0.0-1.0 by the active
            backend: 0.0 sits at that backend's match threshold and 1.0 is a perfect
            match, so the number means the same thing on every backend even though
            the underlying metrics differ.
            Faces are returned in order of detection (top to bottom, left to right).

        Raises:
            ImportError: If the backend's model files are not installed
        """
        if not self.reference_faces:
            logger.debug("No reference faces loaded, cannot identify faces")
            return []

        memo_key = self._memo_fingerprint(image_path)
        if memo_key is not None and memo_key == self._memo_key and self._memo_value is not None:
            logger.debug(f"Reusing memoized face detection for {image_path}")
            return list(self._memo_value)

        try:
            face_embeddings = self.backend.detect_and_embed(Path(image_path))

            if not face_embeddings or len(face_embeddings) == 0:
                logger.debug(f"No faces detected in {image_path}")
                identified: List[Tuple[str, float]] = []
            else:
                logger.debug(f"Detected {len(face_embeddings)} face(s) in {image_path}")
                identified = self._match_embeddings(face_embeddings)

            if memo_key is not None:
                self._memo_key = memo_key
                self._memo_value = list(identified)

            return identified

        except ImportError:
            # Missing model files are a setup problem, not a bad image - propagate.
            raise
        except Exception as e:
            logger.warning(f"Error identifying faces in {image_path}: {e}")
            return []

    def _match_embeddings(self, face_embeddings: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Match face embeddings against the reference set.

        Metric-agnostic: the backend's :meth:`score` folds distance-vs-similarity
        direction and the match threshold into one normalized number, so the best
        reference is the highest score and a match is any score at or above zero.

        Args:
            face_embeddings: One embedding per detected face, in detection order

        Returns:
            List of (person_name, confidence) tuples, one per input embedding, with
            unmatched faces as ("Unknown", 0.0).
        """
        identified: List[Tuple[str, float]] = []

        for face_embedding in face_embeddings:
            best_match: Optional[str] = None
            best_score = float("-inf")

            # Compare with all reference faces
            for person_name, reference_encodings in self.reference_faces.items():
                # Compare with all encodings for this person (multiple reference images)
                for ref_encoding in reference_encodings:
                    score = self.backend.score(face_embedding, ref_encoding)

                    if score > best_score:
                        best_score = score
                        best_match = person_name

            # A non-negative score means the backend considers this a match
            if best_match is not None and best_score >= 0.0:
                confidence = min(1.0, best_score)
                identified.append((best_match, confidence))
                logger.debug(f"Identified: {best_match} (confidence: {confidence:.2f})")
            else:
                identified.append(("Unknown", 0.0))
                logger.debug(
                    f"Unknown face (best match: {best_match}, score: {best_score:.3f}, "
                    f"backend: {self.backend.describe()})"
                )

        return identified

    def get_person_names(self, image_path: str, min_confidence: float = 0.25) -> List[str]:
        """Get list of identified person names from an image.

        Args:
            image_path: Path to image file to analyze
            min_confidence: Minimum confidence threshold (0.0-1.0). Because every
                backend normalizes confidence the same way - 0.0 at its own match
                threshold, 1.0 at a perfect match - this value keeps its meaning
                when the backend changes.

        Returns:
            List of person names (without duplicates, in detection order). Names keep
            the underscores of their reference directory (``John_Doe``).
        """
        identified = self.identify_faces(image_path)

        # Filter by confidence and extract names
        names = [
            name
            for name, confidence in identified
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

        Removes every backend's cache files, plus any legacy pre-v2 files.

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
                self._invalidate_memo()
            return cleared

        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")
            return False

    def get_cache_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the current cache.

        Returns:
            Dictionary with cache info, or None if no cache exists. Each manifest
            entry reports the backend and embedding format it was built with;
            legacy v1 entries report None for those fields.
        """
        try:
            # Find any manifest files
            if not self.cache_dir.exists():
                return None

            manifests = list(self.cache_dir.glob("face_encodings_*_manifest.json"))
            if not manifests:
                return None

            info: Dict[str, Any] = {"cache_dir": str(self.cache_dir), "manifests": []}

            for manifest_path in manifests:
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)

                encodings_path = manifest_path.with_name(
                    manifest_path.name.replace("_manifest.json", ".pkl")
                )

                info["manifests"].append(
                    {
                        "ref_dir": manifest.get("ref_dir"),
                        "version": manifest.get("version"),
                        "created": manifest.get("created"),
                        "file_count": len(manifest.get("files", {})),
                        "cache_size_bytes": (
                            encodings_path.stat().st_size if encodings_path.exists() else 0
                        ),
                        "backend": manifest.get("backend"),
                        "embedding_model": manifest.get("embedding_model"),
                        "dim": manifest.get("dim"),
                    }
                )

            return info

        except Exception as e:
            logger.warning(f"Failed to get cache info: {e}")
            return None
