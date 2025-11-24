"""Custom exceptions for vision model operations."""


class VisionModelError(Exception):
    """Base exception for vision model errors."""
    pass


class VisionModelConnectionError(VisionModelError):
    """Raised when connection to vision model service fails."""
    pass


class VisionModelTimeoutError(VisionModelError):
    """Raised when vision model request times out."""
    pass


class VisionModelInvalidResponseError(VisionModelError):
    """Raised when vision model returns invalid or unexpected response."""
    pass


class VisionModelImageError(VisionModelError):
    """Raised when image processing fails (invalid format, missing file, etc.)."""
    pass

