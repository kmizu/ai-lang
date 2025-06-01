"""Error types for ai-lang.

This module re-exports the enhanced error types from error_reporting
for backward compatibility, while providing the new enhanced functionality.
"""

from typing import Optional
from .syntax import SourceLocation
from .error_reporting import (
    AiLangError,
    TypeCheckError as _TypeCheckError,
    EvaluationError as _EvalError,
    ParseError as _ParseError,
    LexError as _LexError,
    ErrorContext,
    ErrorKind,
    get_trace,
    enable_trace,
    disable_trace,
    clear_trace,
)


# Re-export with original names for compatibility
class TypeCheckError(_TypeCheckError):
    """Type checking error with backward compatibility."""
    def __init__(self, message: str, location: Optional[SourceLocation] = None, context: Optional[ErrorContext] = None):
        if context is None and location is not None:
            context = ErrorContext(location=location)
        super().__init__(message, context)
        self.location = location


class EvalError(_EvalError):
    """Evaluation error with backward compatibility."""
    def __init__(self, message: str, location: Optional[SourceLocation] = None, context: Optional[ErrorContext] = None):
        if context is None and location is not None:
            context = ErrorContext(location=location)
        super().__init__(message, context)
        self.location = location


class OptimizationError(AiLangError):
    """Optimization error."""
    pass


class ParseError(_ParseError):
    """Parse error."""
    pass


class LexError(_LexError):
    """Lexical error."""
    pass