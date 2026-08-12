"""Factory for creating vision model instances."""

import logging
from typing import Any, Dict, List, Optional, Type

from smugvision.vision.base import VisionModel
from smugvision.vision.llama import LlamaVisionModel
from smugvision.vision.exceptions import VisionModelError

logger = logging.getLogger(__name__)

# Shown by list_models() when Ollama cannot be reached, purely as a hint about the
# kind of name that belongs in vision.model. These are NOT an allow-list.
_MODEL_HINTS: List[str] = [
    "qwen3-vl:8b",
    "gemma3:12b",
    "llama3.2-vision",
    "llava:13b",
    "minicpm-v",
]


class VisionModelFactory:
    """Factory for creating vision model instances.

    Any model name is accepted and served by :class:`LlamaVisionModel`, the generic
    Ollama adapter - if Ollama can run it, smugVision can use it. The registry is an
    override map rather than an allow-list: register a name only when it needs a
    genuinely different :class:`VisionModel` implementation.
    """

    # Default implementation used for any model name without an explicit override.
    _default_model_class: Type[VisionModel] = LlamaVisionModel

    # Explicit overrides registered via register_model(). Empty by default.
    _model_registry: Dict[str, Type[VisionModel]] = {}

    @classmethod
    def create(cls, model_name: str, endpoint: Optional[str] = None, **kwargs: Any) -> VisionModel:
        """Create a vision model instance.

        Args:
            model_name: Name of the model to create, as Ollama knows it
                (e.g. ``qwen3-vl:8b``, ``gemma4:latest``)
            endpoint: Optional API endpoint URL
            **kwargs: Additional model-specific configuration, forwarded to the
                model constructor (``timeout``, ``think``, ``keep_alive``,
                ``single_call``, ``structured_output``, ``max_image_dimension``,
                ``jpeg_quality``, ``validate_model``)

        Returns:
            VisionModel instance

        Raises:
            VisionModelError: If the model name is empty/not a string, or if the model
                could not be constructed

        Examples:
            >>> model = VisionModelFactory.create(
            ...     "qwen3-vl:8b", endpoint="http://localhost:11434"
            ... )
        """
        if not isinstance(model_name, str) or not model_name.strip():
            raise VisionModelError(
                f"Invalid model name: {model_name!r}. "
                f"Set vision.model to a model available from `ollama list`."
            )

        model_class = cls._resolve_model_class(model_name)

        logger.info(f"Creating {model_class.__name__} instance for model: {model_name}")

        try:
            # Create model instance with endpoint and any additional kwargs
            return model_class(model_name=model_name.strip(), endpoint=endpoint, **kwargs)
        except VisionModelError:
            raise
        except Exception as e:
            raise VisionModelError(f"Failed to create model {model_name}: {e}") from e

    @classmethod
    def _resolve_model_class(cls, model_name: str) -> Type[VisionModel]:
        """Resolve which implementation should serve a model name.

        Explicit registrations win over the default. A registration for a bare name
        (``myvision``) also covers its tagged variants (``myvision:7b``).

        Args:
            model_name: Requested model name

        Returns:
            The VisionModel subclass to instantiate
        """
        normalized = model_name.lower().strip()

        if normalized in cls._model_registry:
            return cls._model_registry[normalized]

        base_name = normalized.split(":", 1)[0]
        if base_name in cls._model_registry:
            return cls._model_registry[base_name]

        logger.debug(
            f"No explicit registration for '{model_name}'; using "
            f"{cls._default_model_class.__name__}"
        )
        return cls._default_model_class

    @classmethod
    def register_model(cls, name: str, model_class: Type[VisionModel]) -> None:
        """Register a custom implementation for a model name.

        Use this only for a genuinely different implementation (a non-Ollama backend,
        for instance). Ollama models need no registration - they are handled by the
        default adapter.

        Args:
            name: Model name identifier. A bare name also matches tagged variants.
            model_class: VisionModel subclass to register

        Raises:
            VisionModelError: If name is empty or model_class is not a VisionModel
        """
        if not isinstance(name, str) or not name.strip():
            raise VisionModelError(f"Invalid model name for registration: {name!r}")

        if not isinstance(model_class, type) or not issubclass(model_class, VisionModel):
            raise VisionModelError(f"Model class must be a subclass of VisionModel: {model_class}")

        cls._model_registry[name.lower().strip()] = model_class
        logger.debug(f"Registered model: {name} -> {model_class.__name__}")

    @classmethod
    def registered_models(cls) -> List[str]:
        """List names with an explicit implementation override.

        Returns:
            Sorted list of registered model names (usually empty)
        """
        return sorted(cls._model_registry.keys())

    @classmethod
    def list_models(
        cls,
        endpoint: Optional[str] = None,
        vision_only: bool = True,
        timeout: float = 5.0,
    ) -> List[str]:
        """List vision models that can actually be used right now.

        Queries the Ollama server's tag list and, where the server reports model
        capabilities, keeps only the vision-capable ones. Never raises: if the server
        is unreachable it returns any explicitly registered names, or a small static
        hint list when there are none.

        Args:
            endpoint: Optional Ollama endpoint URL (default: client default/OLLAMA_HOST)
            vision_only: Drop models the server reports as not vision-capable. Models
                whose capabilities cannot be determined are kept.
            timeout: Seconds to wait for the server before giving up

        Returns:
            List of usable model names, possibly empty
        """
        names: List[str] = cls.registered_models()

        try:
            import ollama

            client = ollama.Client(host=endpoint, timeout=timeout)
            available = LlamaVisionModel._extract_model_names(client.list())
        except Exception as e:
            logger.debug(f"Could not list models from Ollama ({e}); returning hints instead")
            return names or list(_MODEL_HINTS)

        for name in available:
            if vision_only and not cls._is_vision_capable(client, name):
                continue
            if name not in names:
                names.append(name)

        return names

    @staticmethod
    def _is_vision_capable(client: Any, model_name: str) -> bool:
        """Report whether Ollama says a model can accept images.

        Args:
            client: An ``ollama.Client`` instance
            model_name: Model name to inspect

        Returns:
            True if the model reports vision support, or if support could not be
            determined (benefit of the doubt); False only on an explicit denial
        """
        try:
            info = client.show(model_name)
        except Exception as e:
            logger.debug(f"Could not inspect capabilities of {model_name}: {e}")
            return True

        if isinstance(info, dict):
            capabilities = info.get("capabilities")
        else:
            capabilities = getattr(info, "capabilities", None)

        if not capabilities:
            # Older servers do not report capabilities at all.
            return True

        return any(str(capability).lower() == "vision" for capability in capabilities)
