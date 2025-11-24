"""EXIF data extraction utilities for images."""

import logging
from dataclasses import dataclass
from math import radians, cos, sin, asin, sqrt
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image
from PIL.ExifTags import GPS, TAGS

# Register HEIF/HEIC support if available
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False

logger = logging.getLogger(__name__)


@dataclass
class ExifLocation:
    """Location information extracted from EXIF data.
    
    Attributes:
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
        location_name: Human-readable location name (from reverse geocoding)
        has_coordinates: Whether GPS coordinates are available
    """
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    location_name: Optional[str] = None
    has_coordinates: bool = False
    
    def __str__(self) -> str:
        """Return location as a formatted string."""
        if self.location_name:
            return self.location_name
        elif self.has_coordinates:
            return f"{self.latitude:.6f}, {self.longitude:.6f}"
        else:
            return "Unknown location"


def _convert_to_decimal_degrees(
    degrees: Tuple[float, float, float], ref: str
) -> float:
    """Convert GPS coordinates from degrees/minutes/seconds to decimal.
    
    Args:
        degrees: Tuple of (degrees, minutes, seconds)
        ref: Reference direction ('N', 'S', 'E', 'W')
        
    Returns:
        Decimal degrees (negative for South/West)
    """
    decimal = degrees[0] + degrees[1] / 60.0 + degrees[2] / 3600.0
    
    if ref in ('S', 'W'):
        decimal = -decimal
    
    return decimal


def extract_exif_location(image_path: str) -> ExifLocation:
    """Extract GPS location data from image EXIF metadata.
    
    Supports JPEG, PNG, and HEIC/HEIF formats. For HEIC files, tries multiple
    methods including Pillow and exifread (if available).
    
    Args:
        image_path: Path to the image file
        
    Returns:
        ExifLocation object with GPS coordinates if available
        
    Examples:
        >>> location = extract_exif_location("photo.jpg")
        >>> if location.has_coordinates:
        ...     print(f"Photo taken at: {location.latitude}, {location.longitude}")
    """
    location = ExifLocation()
    
    # Try exifread first for HEIC files (often better at reading HEIC metadata)
    file_ext = Path(image_path).suffix.lower()
    is_heic = file_ext in ['.heic', '.heif']
    
    if is_heic:
        try:
            import exifread
            location = _extract_gps_with_exifread(image_path)
            if location.has_coordinates:
                logger.debug(f"Successfully extracted GPS using exifread from {image_path}")
                return location
        except ImportError:
            logger.debug("exifread not available, using Pillow method")
        except Exception as e:
            logger.debug(f"exifread extraction failed: {e}, trying Pillow method")
    
    # Use Pillow method (works for most formats)
    try:
        image_file = Path(image_path)
        if not image_file.exists():
            logger.warning(f"Image file not found: {image_path}")
            return location
        
        # Check for HEIC format
        file_ext = image_file.suffix.lower()
        is_heic = file_ext in ['.heic', '.heif']
        
        if is_heic and not HEIC_SUPPORT:
            logger.warning(
                f"HEIC format detected but pillow-heif not installed. "
                f"EXIF extraction may fail. Install with: pip install pillow-heif"
            )
        
        with Image.open(image_path) as img:
            # Get EXIF data (try modern API first, fallback to legacy)
            exif_data = None
            try:
                # Modern Pillow API (Pillow 6.0+)
                exif_data = img.getexif()
                if exif_data is None or len(exif_data) == 0:
                    # Try legacy API
                    exif_data = getattr(img, '_getexif', None)()
            except (AttributeError, TypeError):
                # Legacy API
                try:
                    exif_data = img._getexif()
                except (AttributeError, TypeError):
                    pass
            
            if exif_data is None or len(exif_data) == 0:
                logger.debug(f"No EXIF data found in {image_path}")
                return location
            
            # Extract GPS info
            gps_info = None
            
            # Try modern API: GPS IFD is at tag 34853
            try:
                if hasattr(exif_data, 'get_ifd'):
                    # Modern Pillow Exif object - use get_ifd for GPS IFD
                    gps_ifd = exif_data.get_ifd(34853)  # GPS IFD tag ID
                    if gps_ifd and len(gps_ifd) > 0:
                        gps_info = gps_ifd
                        logger.debug(f"Found GPS data via get_ifd: {len(gps_ifd)} GPS tags")
            except (AttributeError, KeyError, TypeError) as e:
                logger.debug(f"get_ifd method not available or GPS IFD not found: {e}")
            
            # Fallback: Try to find GPSInfo in main EXIF tags
            if gps_info is None:
                try:
                    items = exif_data.items() if hasattr(exif_data, 'items') else exif_data
                except (AttributeError, TypeError):
                    items = []
                
                for tag_id, value in items:
                    tag = TAGS.get(tag_id, tag_id)
                    if tag == 'GPSInfo' or tag_id == 34853:
                        if isinstance(value, dict):
                            gps_info = value
                            logger.debug(f"Found GPS data in main EXIF tags")
                            break
                        elif hasattr(exif_data, 'get_ifd'):
                            # Value might be an IFD pointer, try get_ifd
                            try:
                                gps_info = exif_data.get_ifd(tag_id)
                                if gps_info:
                                    break
                            except (KeyError, AttributeError):
                                pass
            
            if gps_info is None or len(gps_info) == 0:
                logger.debug(f"No GPS data found in EXIF for {image_path}")
                return location
            
            # Parse GPS coordinates
            latitude = None
            longitude = None
            lat_ref = None
            lon_ref = None
            
            for key, value in gps_info.items():
                gps_tag = GPS.get(key, key)
                
                if gps_tag == 'GPSLatitude':
                    latitude = value
                elif gps_tag == 'GPSLatitudeRef':
                    lat_ref = value
                elif gps_tag == 'GPSLongitude':
                    longitude = value
                elif gps_tag == 'GPSLongitudeRef':
                    lon_ref = value
            
            # Convert to decimal degrees
            if latitude and longitude and lat_ref and lon_ref:
                try:
                    lat_decimal = _convert_to_decimal_degrees(latitude, lat_ref)
                    lon_decimal = _convert_to_decimal_degrees(longitude, lon_ref)
                    
                    location.latitude = lat_decimal
                    location.longitude = lon_decimal
                    location.has_coordinates = True
                    
                    logger.debug(
                        f"Extracted GPS coordinates from {image_path}: "
                        f"{lat_decimal:.6f}, {lon_decimal:.6f}"
                    )
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Failed to convert GPS coordinates for {image_path}: {e}"
                    )
            else:
                logger.debug(
                    f"Incomplete GPS data in {image_path} "
                    f"(lat: {latitude}, lon: {longitude}, "
                    f"refs: {lat_ref}/{lon_ref})"
                )
                
    except Exception as e:
        logger.warning(f"Error extracting EXIF location from {image_path}: {e}")
    
    return location


def _extract_gps_with_exifread(image_path: str) -> ExifLocation:
    """Extract GPS coordinates using exifread library.
    
    This is often more reliable for HEIC files than Pillow.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        ExifLocation object with GPS coordinates if available
    """
    location = ExifLocation()
    
    try:
        import exifread
        
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f, details=True)
            
            # Look for GPS tags
            gps_latitude = None
            gps_latitude_ref = None
            gps_longitude = None
            gps_longitude_ref = None
            
            for tag_name, tag_value in tags.items():
                # Check various possible GPS tag name formats
                if tag_name in ['GPS GPSLatitude', 'GPS Latitude', 'EXIF GPSLatitude']:
                    # exifread returns Rational objects
                    try:
                        if hasattr(tag_value, 'values'):
                            # It's a Ratio object, convert to tuple
                            values = tag_value.values
                            gps_latitude = (float(values[0]), float(values[1]), float(values[2]))
                        else:
                            # Try to parse as string
                            parts = str(tag_value).split()
                            if len(parts) >= 3:
                                gps_latitude = (
                                    float(parts[0].split('/')[0]) / float(parts[0].split('/')[1]) if '/' in parts[0] else float(parts[0]),
                                    float(parts[1].split('/')[0]) / float(parts[1].split('/')[1]) if '/' in parts[1] else float(parts[1]),
                                    float(parts[2].split('/')[0]) / float(parts[2].split('/')[1]) if '/' in parts[2] else float(parts[2])
                                )
                    except (ValueError, AttributeError, IndexError) as e:
                        logger.debug(f"Error parsing GPS latitude from exifread: {e}")
                        
                elif tag_name in ['GPS GPSLatitudeRef', 'GPS LatitudeRef', 'EXIF GPSLatitudeRef']:
                    gps_latitude_ref = str(tag_value).strip()
                elif tag_name in ['GPS GPSLongitude', 'GPS Longitude', 'EXIF GPSLongitude']:
                    try:
                        if hasattr(tag_value, 'values'):
                            values = tag_value.values
                            gps_longitude = (float(values[0]), float(values[1]), float(values[2]))
                        else:
                            parts = str(tag_value).split()
                            if len(parts) >= 3:
                                gps_longitude = (
                                    float(parts[0].split('/')[0]) / float(parts[0].split('/')[1]) if '/' in parts[0] else float(parts[0]),
                                    float(parts[1].split('/')[0]) / float(parts[1].split('/')[1]) if '/' in parts[1] else float(parts[1]),
                                    float(parts[2].split('/')[0]) / float(parts[2].split('/')[1]) if '/' in parts[2] else float(parts[2])
                                )
                    except (ValueError, AttributeError, IndexError) as e:
                        logger.debug(f"Error parsing GPS longitude from exifread: {e}")
                elif tag_name in ['GPS GPSLongitudeRef', 'GPS LongitudeRef', 'EXIF GPSLongitudeRef']:
                    gps_longitude_ref = str(tag_value).strip()
            
            # Convert to decimal degrees
            if gps_latitude and gps_longitude and gps_latitude_ref and gps_longitude_ref:
                try:
                    lat_decimal = _convert_to_decimal_degrees(gps_latitude, gps_latitude_ref)
                    lon_decimal = _convert_to_decimal_degrees(gps_longitude, gps_longitude_ref)
                    
                    location.latitude = lat_decimal
                    location.longitude = lon_decimal
                    location.has_coordinates = True
                    
                    logger.debug(
                        f"Extracted GPS coordinates using exifread: "
                        f"{lat_decimal:.6f}, {lon_decimal:.6f}"
                    )
                except (ValueError, TypeError) as e:
                    logger.debug(f"Failed to convert GPS coordinates from exifread: {e}")
                    
    except Exception as e:
        logger.debug(f"Error using exifread for {image_path}: {e}")
    
    return location


def reverse_geocode(
    latitude: float,
    longitude: float,
    interactive: bool = False
) -> Optional[str]:
    """Convert GPS coordinates to a human-readable location name.
    
    This function attempts reverse geocoding using available services.
    Falls back gracefully if geocoding is unavailable. Tries to get
    the most specific location possible, including business/venue names.
    
    Args:
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
        interactive: If True and multiple venues found, prompt user to select
        
    Returns:
        Location name string (e.g., "The Louisville Palace Theater, Louisville, Kentucky"),
        or None if geocoding fails
        
    Note:
        This implementation uses Nominatim (OpenStreetMap) via geopy.
        For production use, consider using a dedicated geocoding service.
    """
    try:
        # Try using geopy if available (optional dependency)
        try:
            from geopy.geocoders import Nominatim
            from geopy.exc import GeocoderTimedOut, GeocoderServiceError
            
            geolocator = Nominatim(user_agent="smugvision/0.1.0")
            
            try:
                location = geolocator.reverse(
                    (latitude, longitude),
                    timeout=10,
                    language='en',
                    exactly_one=True
                )
                
                if location:
                    # Try to get a meaningful location string with hierarchy
                    raw_data = location.raw
                    address = raw_data.get('address', {})
                    
                    # Check for building/venue name in the raw response
                    building_name = raw_data.get('name', '').strip()
                    place_type = raw_data.get('type', '')
                    place_class = raw_data.get('class', '')
                    
                    # If no name, search for any nearby business/venue
                    # Search for various venue types: restaurants, theaters, businesses, etc.
                    if not building_name:
                        # Comprehensive list of venue/business types to search
                        all_venue_types = [
                            'restaurant', 'cafe', 'coffee', 'bar', 'pub', 'brewery',
                            'theater', 'theatre', 'cinema', 'venue', 'hall', 'auditorium',
                            'museum', 'gallery', 'library',
                            'school', 'university', 'college',
                            'hotel', 'motel', 'resort',
                            'shop', 'store', 'market', 'mall',
                            'gym', 'fitness', 'sports',
                            'park', 'stadium', 'arena',
                            'church', 'temple', 'mosque', 'synagogue',
                            'hospital', 'clinic', 'pharmacy',
                            'bank', 'office', 'business'
                        ]
                        
                        # Collect all nearby venues within 200m
                        nearby_venues = []
                        
                        for search_term in all_venue_types:
                            try:
                                query = f"{search_term} near {latitude},{longitude}"
                                search_results = geolocator.geocode(
                                    query,
                                    exactly_one=False,
                                    limit=5,
                                    timeout=5
                                )
                                
                                if search_results:
                                    for result in search_results:
                                        result_lat = float(result.raw.get('lat', 0))
                                        result_lon = float(result.raw.get('lon', 0))
                                        
                                        # Calculate distance
                                        lat1, lon1 = radians(latitude), radians(longitude)
                                        lat2, lon2 = radians(result_lat), radians(result_lon)
                                        dlat = lat2 - lat1
                                        dlon = lon2 - lon1
                                        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                                        c = 2 * asin(sqrt(a))
                                        distance = 6371000 * c
                                        
                                        if distance < 200:  # Within 200 meters
                                            candidate_name = result.raw.get('name', '').strip()
                                            if candidate_name:
                                                # Avoid duplicates
                                                if not any(v['name'] == candidate_name for v in nearby_venues):
                                                    nearby_venues.append({
                                                        'name': candidate_name,
                                                        'distance': distance,
                                                        'type': search_term
                                                    })
                            except Exception as e:
                                logger.debug(f"Could not search for {search_term} nearby: {e}")
                                continue
                        
                        # Sort by distance (closest first)
                        nearby_venues.sort(key=lambda x: x['distance'])
                        
                        if nearby_venues:
                            # If multiple venues found, handle selection
                            if len(nearby_venues) > 1:
                                logger.info(
                                    f"Found {len(nearby_venues)} nearby venues at this location:"
                                )
                                for i, venue in enumerate(nearby_venues[:10], 1):  # Show top 10
                                    logger.info(
                                        f"  {i}. {venue['name']} ({venue['type']}, "
                                        f"{venue['distance']:.0f}m away)"
                                    )
                                
                                if interactive:
                                    # Prompt user to select
                                    try:
                                        print(f"\nMultiple venues found at this location:")
                                        for i, venue in enumerate(nearby_venues[:10], 1):
                                            print(
                                                f"  {i}. {venue['name']} "
                                                f"({venue['type']}, {venue['distance']:.0f}m)"
                                            )
                                        print(f"  {len(nearby_venues) + 1}. Use closest venue (default)")
                                        
                                        choice = input(
                                            f"\nSelect venue (1-{len(nearby_venues) + 1}, default={len(nearby_venues) + 1}): "
                                        ).strip()
                                        
                                        if choice and choice.isdigit():
                                            choice_num = int(choice)
                                            if 1 <= choice_num <= len(nearby_venues):
                                                building_name = nearby_venues[choice_num - 1]['name']
                                                logger.info(
                                                    f"User selected: {building_name}"
                                                )
                                            else:
                                                # Default to closest
                                                building_name = nearby_venues[0]['name']
                                                logger.info(
                                                    f"Using closest venue: {building_name}"
                                                )
                                        else:
                                            # Default to closest
                                            building_name = nearby_venues[0]['name']
                                            logger.info(
                                                f"Using closest venue: {building_name}"
                                            )
                                    except (EOFError, KeyboardInterrupt):
                                        # Non-interactive or interrupted, use closest
                                        building_name = nearby_venues[0]['name']
                                        logger.info(
                                            f"Non-interactive mode: using closest venue: {building_name}"
                                        )
                                else:
                                    # Non-interactive: use closest
                                    building_name = nearby_venues[0]['name']
                                    logger.info(
                                        f"Using closest venue: {building_name} "
                                        f"(found {len(nearby_venues)} total venues)"
                                    )
                            else:
                                # Single venue found
                                building_name = nearby_venues[0]['name']
                                logger.debug(
                                    f"Found venue: {building_name} "
                                    f"(distance: {nearby_venues[0]['distance']:.1f}m)"
                                )
                    
                    # Also check address for building name (sometimes in address fields)
                    if not building_name:
                        building_name = (
                            address.get('building') or
                            address.get('amenity') or
                            address.get('school') or
                            address.get('university') or
                            address.get('college')
                        )
                    
                    # Build location string from most specific to general
                    # Priority: building/venue > city/town > county > state > country
                    parts = []
                    
                    # Most specific: building/venue name
                    if building_name:
                        parts.append(building_name)
                    
                    # City, town, village, or hamlet
                    city = (
                        address.get('city') or
                        address.get('town') or
                        address.get('village') or
                        address.get('hamlet') or
                        address.get('municipality')
                    )
                    if city and city not in parts:  # Avoid duplicates
                        parts.append(city)
                    
                    # County or region
                    county = address.get('county')
                    if county:
                        # Remove "County" suffix if present to avoid redundancy
                        county_clean = county.replace(' County', '').strip()
                        if county_clean not in parts:  # Avoid duplicates
                            parts.append(county_clean)
                    
                    # State or region
                    state = address.get('state') or address.get('region')
                    if state:
                        parts.append(state)
                    
                    # Country (always include if available)
                    country = address.get('country')
                    if country:
                        parts.append(country)
                    
                    if parts:
                        location_str = ", ".join(parts)
                        logger.debug(
                            f"Reverse geocoded {latitude:.6f}, {longitude:.6f} "
                            f"to: {location_str}"
                        )
                        return location_str
                    else:
                        # Fallback to full address if available
                        return location.address
                        
            except (GeocoderTimedOut, GeocoderServiceError) as e:
                logger.debug(f"Geocoding service error: {e}")
                return None
                
        except ImportError:
            # geopy not installed, use fallback
            logger.debug(
                "geopy not available. Install with 'pip install geopy' "
                "for reverse geocoding support."
            )
            return None
            
    except Exception as e:
        logger.debug(f"Error during reverse geocoding: {e}")
        return None
    
    return None

