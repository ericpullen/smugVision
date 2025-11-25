"""Custom exceptions for SmugMug API operations."""


class SmugMugError(Exception):
    """Base exception for SmugMug-related errors."""
    pass


class SmugMugAPIError(SmugMugError):
    """Exception raised for SmugMug API errors.
    
    Attributes:
        message: Error message
        status_code: HTTP status code
        response: Full response data (if available)
    """
    
    def __init__(self, message: str, status_code: int = None, response: dict = None):
        """Initialize SmugMug API error.
        
        Args:
            message: Error message
            status_code: HTTP status code
            response: Full response data
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response = response
    
    def __str__(self) -> str:
        """Return string representation of error."""
        if self.status_code:
            return f"SmugMug API Error ({self.status_code}): {self.message}"
        return f"SmugMug API Error: {self.message}"


class SmugMugAuthError(SmugMugError):
    """Exception raised for authentication errors."""
    pass


class SmugMugNotFoundError(SmugMugAPIError):
    """Exception raised when a resource is not found (404)."""
    pass


class SmugMugRateLimitError(SmugMugAPIError):
    """Exception raised when API rate limit is exceeded (429)."""
    
    def __init__(self, message: str, retry_after: int = None):
        """Initialize rate limit error.
        
        Args:
            message: Error message
            retry_after: Seconds to wait before retrying (if provided by API)
        """
        super().__init__(message, status_code=429)
        self.retry_after = retry_after
    
    def __str__(self) -> str:
        """Return string representation of error."""
        if self.retry_after:
            return f"SmugMug Rate Limit Exceeded. Retry after {self.retry_after} seconds."
        return "SmugMug Rate Limit Exceeded."

