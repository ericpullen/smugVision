"""InsightFace (ArcFace ONNX) face embedding backend.

ArcFace embeddings are markedly more robust than dlib's to profile views, head
angle and age variation, which is what matters for family albums spanning years.
On Apple silicon the ONNX session can run through CoreML (and therefore the ANE).

Produces 512-dimensional, L2-normalized float32 embeddings compared by **cosine
similarity, where HIGHER means more similar** - the opposite direction to dlib's
euclidean distance. Those vectors are mathematically incomparable with dlib's, so
the recognizer namespaces the encoding cache per backend; see
:meth:`~smugvision.face.backends.base.FaceBackend.cache_signature`.

``insightface`` and ``onnxruntime`` are optional dependencies. The import is
guarded so that merely importing this module on a machine without them sets
``INSIGHTFACE_AVAILABLE = False`` instead of raising.
"""

import logging
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from smugvision.face.backends.base import FaceBackend

logger = logging.getLogger(__name__)

# Try to import insightface, make it optional
INSIGHTFACE_AVAILABLE = False
INSIGHTFACE_ERROR = None

try:
    from insightface.app import FaceAnalysis

    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    INSIGHTFACE_ERROR = "not_installed"
    logger.debug(
        'insightface library not available. Install with: pip install "smugvision[insightface]"'
    )
except Exception as e:  # pragma: no cover - depends on local ONNX runtime install
    INSIGHTFACE_AVAILABLE = False
    INSIGHTFACE_ERROR = str(e)
    logger.debug(f"insightface library could not be imported: {e}")

# Default cosine-similarity threshold for the buffalo_l model pack. Practical
# operating range is roughly 0.35-0.45; higher is stricter. This is NOT related to
# the dlib `tolerance` value and the two must never be substituted for each other.
DEFAULT_SIMILARITY_THRESHOLD = 0.4
DEFAULT_MODEL_NAME = "buffalo_l"
DEFAULT_DET_SIZE = (640, 640)


class InsightFaceBackend(FaceBackend):
    """Face embeddings via InsightFace's ArcFace ONNX models.

    Detection uses SCRFD and recognition uses ArcFace, both bundled in the model
    pack named by ``model_name``. The model pack is downloaded on first use (a few
    hundred MB for ``buffalo_l``) into ``~/.insightface/models``, so the ONNX
    session is created lazily rather than at construction time.

    Threshold semantics, spelled out because they invert dlib's:

    * dlib compares euclidean distance against ``tolerance`` - LOWER is a better
      match, default 0.6.
    * This backend compares cosine similarity against ``similarity_threshold`` -
      HIGHER is a better match, default 0.4.

    The two numbers are on unrelated scales and are deliberately kept as separate
    settings. What *is* shared is the output of :meth:`score`, which both backends
    normalize to 0.0 at their own threshold and 1.0 at a perfect match, so
    ``min_confidence`` keeps its meaning when you switch backends.

    Attributes:
        embedding_model: Name of the InsightFace model pack, e.g. ``"buffalo_l"``.
        det_size: Detector input resolution as a (width, height) tuple.
        similarity_threshold: Minimum cosine similarity to call a pair a match.
    """

    name = "insightface"
    embedding_dim = 512
    metric = "cosine"
    normalized = True

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        det_size: Sequence[int] = DEFAULT_DET_SIZE,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        providers: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize the InsightFace backend.

        No model is loaded here; the first call to :meth:`encode_reference` or
        :meth:`detect_and_embed` builds the ONNX session.

        Args:
            model_name: InsightFace model pack name. ``"buffalo_l"`` is the
                accurate default; ``"buffalo_s"`` is smaller and faster.
            det_size: Detector input size as (width, height). Larger finds smaller
                faces at more cost.
            similarity_threshold: Cosine similarity at or above which two faces are
                considered the same person (0.0-1.0, higher is stricter).
            providers: Explicit ONNX Runtime execution providers. Defaults to
                CoreML then CPU on macOS, CPU elsewhere.

        Raises:
            ImportError: If insightface / onnxruntime are not installed.
        """
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError(insightface_unavailable_message())

        self.embedding_model = model_name
        self.det_size: Tuple[int, int] = (int(det_size[0]), int(det_size[1]))
        self.similarity_threshold = float(similarity_threshold)
        self._providers = list(providers) if providers else None
        self._app: Optional[Any] = None

    def _resolve_providers(self) -> List[str]:
        """Pick ONNX Runtime execution providers available on this machine.

        Returns:
            Provider names in priority order, always ending with a CPU fallback.
        """
        if self._providers:
            return self._providers

        preferred = []
        if sys.platform == "darwin":
            # CoreML routes to the Neural Engine / GPU on Apple silicon.
            preferred.append("CoreMLExecutionProvider")
        preferred.append("CPUExecutionProvider")

        try:
            import onnxruntime

            available = set(onnxruntime.get_available_providers())
        except Exception as e:  # pragma: no cover - onnxruntime is a hard requirement
            logger.debug(f"Could not query onnxruntime providers: {e}")
            return ["CPUExecutionProvider"]

        resolved = [p for p in preferred if p in available]
        return resolved or ["CPUExecutionProvider"]

    def _ensure_app(self) -> Any:
        """Build the InsightFace model session on first use.

        Returns:
            The prepared ``FaceAnalysis`` instance.

        Raises:
            ImportError: If the model pack cannot be loaded or downloaded.
        """
        if self._app is not None:
            return self._app

        providers = self._resolve_providers()
        logger.info(
            f"Initializing InsightFace model '{self.embedding_model}' "
            f"(det_size={self.det_size}, providers={providers}). "
            "First run downloads the model pack to ~/.insightface/models."
        )

        try:
            app = FaceAnalysis(name=self.embedding_model, providers=providers)
            # ctx_id=-1 selects CPU-side context; GPU selection is via providers.
            app.prepare(ctx_id=-1, det_size=self.det_size)
        except Exception as e:
            raise ImportError(
                f"Failed to initialize InsightFace model '{self.embedding_model}': {e}\n"
                "Check that onnxruntime is installed and that the model pack could be "
                "downloaded to ~/.insightface/models."
            ) from e

        self._app = app
        return app

    def _load_bgr(self, image_path: Path) -> np.ndarray:
        """Load an image as a BGR numpy array with EXIF orientation applied.

        Args:
            image_path: Path to the image file.

        Returns:
            HxWx3 uint8 array in BGR channel order.
        """
        from PIL import Image, ImageOps

        pil_image: Image.Image = Image.open(str(image_path))
        # Apply EXIF orientation (rotates/flips image if needed). exif_transpose
        # returns a new image, and only returns None in its in_place mode.
        rotated = ImageOps.exif_transpose(pil_image)
        if rotated is not None:
            pil_image = rotated
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        rgb: np.ndarray = np.array(pil_image)
        pil_image.close()

        # insightface follows the OpenCV convention and expects BGR. Convert in place
        # with OpenCV rather than `ascontiguousarray(rgb[:, :, ::-1])`: the slice-reverse
        # allocates a second full-resolution buffer (~33MB on a 12MP photo, held alongside
        # the first) and measures ~22ms, against ~1ms and zero extra allocation here.
        # cv2 ships as a hard dependency of insightface, so this adds nothing new.
        import cv2

        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR, dst=rgb)
        return rgb

    def _embedding(self, face: Any) -> Optional[np.ndarray]:
        """Extract the L2-normalized embedding from an InsightFace result.

        Args:
            face: A single face object returned by ``FaceAnalysis.get()``.

        Returns:
            512-d float32 unit vector, or None if the recognition model produced none.
        """
        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            return None
        vector: np.ndarray = np.asarray(embedding, dtype=np.float32)
        return vector

    def encode_reference(self, image_path: Path) -> Optional[np.ndarray]:
        """Encode the primary face in a reference image.

        When several faces are present the largest is used, on the assumption that
        the subject of a reference photo is the closest person. (The dlib backend
        instead takes the first detected face.)

        Args:
            image_path: Path to the reference image file.

        Returns:
            512-d L2-normalized float32 embedding, or None if no face was found.

        Raises:
            ImportError: If the InsightFace model pack cannot be initialized.
        """
        app = self._ensure_app()

        try:
            image = self._load_bgr(image_path)
            faces = app.get(image)
        except Exception as e:
            logger.debug(f"Error encoding face from {image_path}: {e}")
            return None

        if not faces:
            return None

        largest = max(
            faces,
            key=lambda f: (float(f.bbox[2]) - float(f.bbox[0]))
            * (float(f.bbox[3]) - float(f.bbox[1])),
        )
        return self._embedding(largest)

    def detect_and_embed(self, image_path: Path) -> List[np.ndarray]:
        """Detect and embed every face in an image.

        Args:
            image_path: Path to the image file to analyze.

        Returns:
            One 512-d L2-normalized embedding per detected face, sorted top to
            bottom then left to right to match the documented ordering.

        Raises:
            ImportError: If the InsightFace model pack cannot be initialized.
        """
        app = self._ensure_app()

        image = self._load_bgr(image_path)
        faces = app.get(image)

        if not faces:
            return []

        # bbox is (x1, y1, x2, y2); order by top edge, then left edge.
        faces = sorted(faces, key=lambda f: (float(f.bbox[1]), float(f.bbox[0])))

        embeddings = []
        for face in faces:
            embedding = self._embedding(face)
            if embedding is None:
                logger.debug(f"Face detected without an embedding in {image_path}, skipping")
                continue
            embeddings.append(embedding)

        return embeddings

    def score(self, probe: np.ndarray, reference: np.ndarray) -> float:
        """Score a probe embedding against a reference embedding by cosine similarity.

        Both vectors are already L2-normalized, so cosine similarity is a plain dot
        product. The raw similarity is then rescaled to the shared confidence
        contract - 0.0 exactly at :attr:`similarity_threshold`, 1.0 at a perfect
        match, negative below the threshold - which is what makes
        ``min_confidence`` mean the same thing here as on the dlib backend.

        Args:
            probe: Embedding of a face detected in the analyzed image.
            reference: Embedding of a known reference face.

        Returns:
            Normalized confidence; ``>= 0.0`` iff similarity meets the threshold.
        """
        similarity = float(np.dot(probe, reference))
        threshold = self.similarity_threshold

        if threshold >= 1.0:
            # Degenerate configuration: only an exact match can qualify.
            return 0.0 if similarity >= 1.0 else -1.0

        # Cap at 1.0; floating point can nudge a self-match just above unity.
        return min(1.0, (similarity - threshold) / (1.0 - threshold))


def insightface_unavailable_message() -> str:
    """Build the actionable error message for an unusable InsightFace backend.

    Returns:
        Installation guidance matching the specific failure that was detected.
    """
    if INSIGHTFACE_ERROR == "not_installed":
        return (
            "insightface backend requested but the insightface library is not installed.\n"
            'Install with: pip install "smugvision[insightface]"\n'
            "  (or: pip install insightface onnxruntime)"
        )
    return (
        f"insightface backend requested but unusable: {INSIGHTFACE_ERROR}\n"
        'Install with: pip install "smugvision[insightface]"\n'
        "  (or: pip install insightface onnxruntime)"
    )
