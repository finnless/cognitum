class PollClassifierError(Exception):
    """Base exception for cognitum library."""
    pass

class DatasetError(PollClassifierError):
    """Raised when there are issues with dataset operations."""
    pass

class ModelError(PollClassifierError):
    """Raised when there are issues with model operations."""
    pass

class ValidationError(PollClassifierError):
    """Raised when validation fails."""
    pass
