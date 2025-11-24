"""Factory pattern for creating vision model instances."""

import logging
from typing import Dict, Optional, Type

from smugvision.vision.base import VisionModel
from smugvision.vision.llama import LlamaVisionModel
from smugvision.vision.exceptions import VisionModelError

logger = logging.getLogger(__name__)


class VisionModelFactory:
    """Factory for creating vision model instances.
    
    This factory supports creating different vision model implementations
    based on configuration. It provides a centralized way to instantiate
    models and supports future extensibility for additional model types.
    """
    
    # Registry of available model classes
    _model_registry: Dict[str, Type[VisionModel]] = {
        "llama3.2-vision": LlamaVisionModel,
        "llama3.2": LlamaVisionModel,  # Alias
    }
    
    @classmethod
    def create(
        cls,
        model_name: str,
        endpoint: Optional[str] = None,
        **kwargs
    ) -> VisionModel:
        """Create a vision model instance.
        
        Args:
            model_name: Name of the model to create
            endpoint: Optional API endpoint URL
            **kwargs: Additional model-specific configuration
            
        Returns:
            VisionModel instance
            
        Raises:
            VisionModelError: If model name is not supported
            
        Examples:
            >>> factory = VisionModelFactory()
            >>> model = factory.create("llama3.2-vision", endpoint="http://localhost:11434")
        """
        # Normalize model name (case-insensitive)
        model_name_lower = model_name.lower().strip()
        
        if model_name_lower not in cls._model_registry:
            available = ", ".join(cls._model_registry.keys())
            raise VisionModelError(
                f"Unsupported model: {model_name}. "
                f"Available models: {available}"
            )
        
        model_class = cls._model_registry[model_name_lower]
        
        logger.info(f"Creating {model_class.__name__} instance for model: {model_name}")
        
        try:
            # Create model instance with endpoint and any additional kwargs
            model = model_class(
                model_name=model_name,
                endpoint=endpoint,
                **kwargs
            )
            return model
        except Exception as e:
            raise VisionModelError(
                f"Failed to create model {model_name}: {e}"
            ) from e
    
    @classmethod
    def register_model(cls, name: str, model_class: Type[VisionModel]) -> None:
        """Register a new model class with the factory.
        
        This allows extending the factory with custom model implementations.
        
        Args:
            name: Model name identifier
            model_class: VisionModel subclass to register
        """
        if not issubclass(model_class, VisionModel):
            raise VisionModelError(
                f"Model class must be a subclass of VisionModel: {model_class}"
            )
        
        cls._model_registry[name.lower()] = model_class
        logger.debug(f"Registered model: {name} -> {model_class.__name__}")
    
    @classmethod
    def list_models(cls) -> list:
        """List all available model names.
        
        Returns:
            List of registered model names
        """
        return list(cls._model_registry.keys())

