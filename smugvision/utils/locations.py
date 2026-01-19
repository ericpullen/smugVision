"""Custom location resolver for smugVision.

This module provides functionality to resolve GPS coordinates to custom
location names defined in a YAML configuration file. This allows users
to define friendly names for locations like homes, relatives' houses,
or frequently visited places that may not have good reverse geocoding
results.
"""

import logging
from dataclasses import dataclass, field
from math import radians, cos, sin, asin, sqrt
from pathlib import Path
from typing import Optional, List, Dict, Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class CustomLocation:
    """A custom location definition.
    
    Attributes:
        name: Human-readable name for the location
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
        radius: Match radius in meters (photos within this distance will match)
        address: Optional street address for reference
        aliases: Optional list of alternative names (useful for tags)
    """
    name: str
    latitude: float
    longitude: float
    radius: float = 50.0  # Default 50 meters
    address: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate location data after initialization."""
        if not -90 <= self.latitude <= 90:
            raise ValueError(f"Invalid latitude: {self.latitude}. Must be between -90 and 90.")
        if not -180 <= self.longitude <= 180:
            raise ValueError(f"Invalid longitude: {self.longitude}. Must be between -180 and 180.")
        if self.radius <= 0:
            raise ValueError(f"Invalid radius: {self.radius}. Must be positive.")


@dataclass
class LocationMatch:
    """Result of a location match operation.
    
    Attributes:
        location: The matched CustomLocation
        distance: Distance from the query point to the location center (meters)
        is_custom: Always True for custom location matches
    """
    location: CustomLocation
    distance: float
    is_custom: bool = True
    
    @property
    def name(self) -> str:
        """Get the location name."""
        return self.location.name
    
    @property
    def aliases(self) -> List[str]:
        """Get location aliases."""
        return self.location.aliases
    
    @property
    def address(self) -> Optional[str]:
        """Get the location address."""
        return self.location.address


class LocationResolver:
    """Resolves GPS coordinates to custom location names.
    
    This class loads custom locations from a YAML file and provides
    methods to check if given coordinates match any defined location.
    Matching is based on Haversine distance calculation.
    
    Examples:
        >>> resolver = LocationResolver("~/.smugvision/locations.yaml")
        >>> match = resolver.find_match(38.123456, -85.654321)
        >>> if match:
        ...     print(f"Location: {match.name}")
        ...     print(f"Distance: {match.distance:.1f}m")
    """
    
    def __init__(
        self,
        locations_file: Optional[str] = None,
        auto_load: bool = True
    ):
        """Initialize the location resolver.
        
        Args:
            locations_file: Path to the YAML locations file. If None,
                           defaults to ~/.smugvision/locations.yaml
            auto_load: If True, load locations from file during init
        """
        if locations_file:
            self.locations_file = Path(locations_file).expanduser()
        else:
            self.locations_file = Path.home() / ".smugvision" / "locations.yaml"
        
        self._locations: List[CustomLocation] = []
        self._loaded = False
        
        if auto_load and self.locations_file.exists():
            self.load()
    
    def load(self) -> int:
        """Load custom locations from YAML file.
        
        Returns:
            Number of locations loaded
            
        Raises:
            FileNotFoundError: If locations file doesn't exist
            ValueError: If YAML is malformed or contains invalid data
        """
        if not self.locations_file.exists():
            logger.debug(
                f"Custom locations file not found: {self.locations_file} "
                f"(create this file to define custom location names)"
            )
            self._locations = []
            self._loaded = True
            return 0
        
        try:
            with open(self.locations_file, 'r') as f:
                data = yaml.safe_load(f)
            
            if data is None:
                logger.debug("Locations file is empty")
                self._locations = []
                self._loaded = True
                return 0
            
            if not isinstance(data, dict):
                raise ValueError(
                    f"Invalid locations file format: expected a dictionary with 'locations' key"
                )
            
            locations_list = data.get('locations', [])
            if not isinstance(locations_list, list):
                raise ValueError(
                    f"Invalid 'locations' field: expected a list"
                )
            
            self._locations = []
            for i, loc_data in enumerate(locations_list):
                try:
                    location = self._parse_location(loc_data, index=i)
                    self._locations.append(location)
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(
                        f"Skipping invalid location at index {i}: {e}"
                    )
            
            self._loaded = True
            logger.info(
                f"Loaded {len(self._locations)} custom locations from {self.locations_file}"
            )
            return len(self._locations)
            
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse locations YAML: {e}") from e
        except Exception as e:
            logger.error(f"Error loading locations file: {e}")
            raise
    
    def _parse_location(self, data: Dict[str, Any], index: int = 0) -> CustomLocation:
        """Parse a location dictionary into a CustomLocation object.
        
        Args:
            data: Dictionary with location data
            index: Index in the locations list (for error messages)
            
        Returns:
            CustomLocation object
            
        Raises:
            KeyError: If required fields are missing
            ValueError: If field values are invalid
        """
        # Required fields
        if 'name' not in data:
            raise KeyError(f"Location at index {index} missing required 'name' field")
        if 'latitude' not in data:
            raise KeyError(f"Location '{data.get('name', index)}' missing required 'latitude' field")
        if 'longitude' not in data:
            raise KeyError(f"Location '{data.get('name', index)}' missing required 'longitude' field")
        
        # Parse aliases - ensure it's a list
        aliases = data.get('aliases', [])
        if isinstance(aliases, str):
            aliases = [aliases]
        elif not isinstance(aliases, list):
            aliases = []
        
        return CustomLocation(
            name=str(data['name']),
            latitude=float(data['latitude']),
            longitude=float(data['longitude']),
            radius=float(data.get('radius', 50.0)),
            address=data.get('address'),
            aliases=[str(a) for a in aliases]
        )
    
    def find_match(
        self,
        latitude: float,
        longitude: float
    ) -> Optional[LocationMatch]:
        """Find a custom location that matches the given coordinates.
        
        Checks all custom locations and returns the closest one that
        contains the given coordinates within its radius. If multiple
        locations match, the closest one is returned.
        
        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            
        Returns:
            LocationMatch if coordinates are within a custom location's radius,
            None otherwise
        """
        if not self._loaded:
            self.load()
        
        if not self._locations:
            return None
        
        best_match: Optional[LocationMatch] = None
        
        for location in self._locations:
            distance = self._haversine_distance(
                latitude, longitude,
                location.latitude, location.longitude
            )
            
            if distance <= location.radius:
                if best_match is None or distance < best_match.distance:
                    best_match = LocationMatch(
                        location=location,
                        distance=distance
                    )
        
        if best_match:
            logger.debug(
                f"Matched coordinates ({latitude:.6f}, {longitude:.6f}) "
                f"to custom location '{best_match.name}' "
                f"(distance: {best_match.distance:.1f}m)"
            )
        
        return best_match
    
    @staticmethod
    def _haversine_distance(
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """Calculate the distance between two points using the Haversine formula.
        
        Args:
            lat1, lon1: First point coordinates (decimal degrees)
            lat2, lon2: Second point coordinates (decimal degrees)
            
        Returns:
            Distance in meters
        """
        # Convert to radians
        lat1, lon1 = radians(lat1), radians(lon1)
        lat2, lon2 = radians(lat2), radians(lon2)
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        # Earth's radius in meters
        r = 6371000
        
        return r * c
    
    @property
    def locations(self) -> List[CustomLocation]:
        """Get list of all loaded custom locations."""
        if not self._loaded:
            self.load()
        return self._locations.copy()
    
    @property
    def location_count(self) -> int:
        """Get the number of loaded custom locations."""
        if not self._loaded:
            self.load()
        return len(self._locations)
    
    def reload(self) -> int:
        """Reload locations from the YAML file.
        
        Returns:
            Number of locations loaded
        """
        self._loaded = False
        return self.load()
    
    def add_location(self, location: CustomLocation) -> None:
        """Add a custom location to the in-memory list.
        
        Note: This does not persist to the YAML file. Call save() to persist.
        
        Args:
            location: CustomLocation to add
        """
        if not self._loaded:
            self.load()
        self._locations.append(location)
        logger.debug(f"Added custom location: {location.name}")
    
    def save(self) -> None:
        """Save current locations to the YAML file.
        
        Creates the parent directory if it doesn't exist.
        """
        # Ensure parent directory exists
        self.locations_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert locations to dictionary format
        locations_data = {
            'locations': [
                {
                    'name': loc.name,
                    'latitude': loc.latitude,
                    'longitude': loc.longitude,
                    'radius': loc.radius,
                    **(({'address': loc.address} if loc.address else {})),
                    **(({'aliases': loc.aliases} if loc.aliases else {})),
                }
                for loc in self._locations
            ]
        }
        
        with open(self.locations_file, 'w') as f:
            yaml.dump(
                locations_data,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True
            )
        
        logger.info(f"Saved {len(self._locations)} locations to {self.locations_file}")


# Module-level resolver instance for convenience
_default_resolver: Optional[LocationResolver] = None


def get_resolver(locations_file: Optional[str] = None) -> LocationResolver:
    """Get or create a LocationResolver instance.
    
    This function provides a convenient way to get a resolver instance
    without managing the lifecycle manually. The resolver is cached
    for subsequent calls.
    
    Args:
        locations_file: Optional path to locations file. If provided and
                       different from the cached resolver's file, a new
                       resolver is created.
                       
    Returns:
        LocationResolver instance
    """
    global _default_resolver
    
    if locations_file:
        path = Path(locations_file).expanduser()
        if _default_resolver is None or _default_resolver.locations_file != path:
            _default_resolver = LocationResolver(locations_file)
    elif _default_resolver is None:
        _default_resolver = LocationResolver()
    
    return _default_resolver


def resolve_location(
    latitude: float,
    longitude: float,
    locations_file: Optional[str] = None
) -> Optional[LocationMatch]:
    """Convenience function to resolve coordinates to a custom location.
    
    Args:
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
        locations_file: Optional path to locations file
        
    Returns:
        LocationMatch if coordinates match a custom location, None otherwise
    """
    resolver = get_resolver(locations_file)
    return resolver.find_match(latitude, longitude)
