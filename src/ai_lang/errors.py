"""Error types for ai-lang."""

from typing import Optional
from .syntax import SourceLocation


class TypeCheckError(Exception):
    """Type checking error."""
    def __init__(self, message: str, location: Optional[SourceLocation] = None):
        super().__init__(message)
        self.location = location


class EvalError(Exception):
    """Evaluation error."""
    pass