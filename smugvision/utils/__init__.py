"""Utility functions for smugVision."""

from smugvision.utils.exif import (
    extract_exif_location,
    reverse_geocode,
    resolve_location_with_custom,
    get_location_for_image,
    ExifLocation,
)

from smugvision.utils.locations import (
    LocationResolver,
    CustomLocation,
    LocationMatch,
    resolve_location,
    get_resolver,
)

__all__ = [
    # EXIF and location utilities
    "extract_exif_location",
    "reverse_geocode",
    "resolve_location_with_custom",
    "get_location_for_image",
    "ExifLocation",
    # Custom location utilities
    "LocationResolver",
    "CustomLocation",
    "LocationMatch",
    "resolve_location",
    "get_resolver",
]

