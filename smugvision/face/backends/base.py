"""Abstract base class for face-embedding backends.

A backend owns everything that is specific to one face-embedding implementation:
detecting faces, turning them into embedding vectors, and scoring a probe vector
against a reference vector. :class:`~smugvision.face.recognizer.FaceRecognizer`
stays backend-agnostic and only orchestrates caching, the reference map, and the
name-filtering contract.

Backends differ in embedding dimensionality (dlib: 128, ArcFace: 512) and in the
metric used to compare them (dlib: euclidean distance, lower is better; ArcFace:
cosine similarity, higher is better). To keep ``FaceRecognizer`` free of that
knowledge, :meth:`FaceBackend.score` folds the metric *and* its direction into a
single normalized number - see the method docstring for the exact contract.
"""

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class FaceBackend(ABC):
    """Base class for a face detection + embedding implementation.

    Subclasses must set the five class/instance attributes below, because they are
    written into the encoding cache manifest and used to namespace cache files.
    Vectors produced by two different backends must never be compared, so these
    values are the cache's collision guard.

    Attributes:
        name: Short backend identifier, e.g. ``"dlib"`` or ``"insightface"``.
        embedding_model: Identifier of the specific embedding model, e.g.
            ``"large"`` for dlib's large encoder or ``"buffalo_l"`` for InsightFace.
        embedding_dim: Length of an embedding vector (128 for dlib, 512 for ArcFace).
        metric: Comparison metric name, ``"euclidean"`` or ``"cosine"``.
        normalized: True if embeddings are L2-normalized, so cosine similarity is a
            plain dot product.
    """

    name: str = "base"
    embedding_model: str = "unknown"
    embedding_dim: int = 0
    metric: str = "unknown"
    normalized: bool = False

    @abstractmethod
    def encode_reference(self, image_path: Path) -> Optional[np.ndarray]:
        """Encode the face in a reference image.

        Reference images are expected to contain one clear photo of a single person.

        Args:
            image_path: Path to the reference image file.

        Returns:
            An embedding vector of length :attr:`embedding_dim`, or None if no face
            could be detected.

        Raises:
            ImportError: If the backend's model files are missing at runtime.
        """

    @abstractmethod
    def detect_and_embed(self, image_path: Path) -> List[np.ndarray]:
        """Detect every face in an image and embed each one.

        Args:
            image_path: Path to the image file to analyze.

        Returns:
            One embedding vector per *detected* face, in detection order (roughly
            top to bottom, left to right). Faces that cannot be matched to a
            reference are still included - the caller labels those "Unknown" - so
            ``len()`` of this list is the true detected-face count.

        Raises:
            ImportError: If the backend's model files are missing at runtime.
        """

    @abstractmethod
    def score(self, probe: np.ndarray, reference: np.ndarray) -> float:
        """Score a probe embedding against a reference embedding.

        This is the one place a backend's metric and threshold semantics are
        expressed, which is what lets the caller stay metric-agnostic. The
        returned value must satisfy all of:

        * **Monotonic in similarity** - a more similar pair scores higher, so the
          caller can pick the best reference with a plain ``max()``.
        * **Sign encodes the match decision** - ``>= 0.0`` means "within this
          backend's threshold", ``< 0.0`` means "not a match". The caller accepts
          on ``>= 0.0`` and never sees the raw distance or the threshold.
        * **Normalized on match** - an accepted score lies in ``0.0..1.0``, where
          0.0 sits exactly at the threshold and 1.0 is a perfect match.

        That last property is what keeps ``min_confidence`` meaning the same thing
        on every backend even though the underlying metrics are unrelated.

        Args:
            probe: Embedding of a face found in the image being analyzed.
            reference: Embedding of a known reference face.

        Returns:
            Normalized confidence as described above. May be negative for a
            non-match; the caller only compares and never displays negatives.
        """

    def cache_signature(self) -> Dict[str, Any]:
        """Describe the embedding format for cache validation.

        Every field here affects the *meaning* of a stored vector. The recognizer
        writes this into the cache manifest and rejects a cache when any field
        disagrees with the live backend, so a dlib 128-d euclidean vector can never
        be scored as an ArcFace 512-d cosine vector.

        Note that matching thresholds are deliberately absent: ``tolerance`` and
        ``similarity_threshold`` change how vectors are compared, not how they are
        computed, so changing them must not invalidate the cache.

        Returns:
            Dictionary with backend, embedding_model, dim, metric and normalized keys.
        """
        return {
            "backend": self.name,
            "embedding_model": self.embedding_model,
            "dim": self.embedding_dim,
            "metric": self.metric,
            "normalized": self.normalized,
        }

    def cache_slug(self) -> str:
        """Build a filesystem-safe cache name fragment identifying this backend.

        Used to namespace cache filenames so switching backends does not destroy
        the other backend's cache and can never read it either.

        Returns:
            A slug such as ``"dlib_large_128d"`` or ``"insightface_buffalo_l_512d"``.
        """
        raw = f"{self.name}_{self.embedding_model}_{self.embedding_dim}d"
        return re.sub(r"[^A-Za-z0-9_]+", "_", raw).strip("_").lower()

    def describe(self) -> str:
        """Return a short human-readable description for logging.

        Returns:
            Description such as ``"insightface/buffalo_l (512-d, cosine)"``.
        """
        return f"{self.name}/{self.embedding_model} ({self.embedding_dim}-d, {self.metric})"
