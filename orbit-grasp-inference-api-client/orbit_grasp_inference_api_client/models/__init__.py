"""Contains all the data models used in inputs/outputs"""

from .detect_request import DetectRequest
from .detect_response import DetectResponse
from .http_validation_error import HTTPValidationError
from .validation_error import ValidationError

__all__ = (
    "DetectRequest",
    "DetectResponse",
    "HTTPValidationError",
    "ValidationError",
)
