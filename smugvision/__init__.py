"""smugVision - AI-powered photo metadata generation for SmugMug.

Automatically generate descriptive captions and relevant tags for your SmugMug
photos using local AI vision models, face recognition, and EXIF metadata.
"""

from smugvision._version import __version__, __version_info__
from smugvision.config import ConfigManager
from smugvision.smugmug import SmugMugClient
from smugvision.processing import ImageProcessor

__author__ = "Eric Pullen"
__license__ = "MIT"
__all__ = [
    "__version__",
    "__version_info__",
    "ConfigManager",
    "SmugMugClient",
    "ImageProcessor",
]

