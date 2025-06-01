"""Enhanced error reporting system for ai-lang.

This module provides improved error messages with:
- Source location tracking
- Context display with highlighted error positions
- Suggestions for common mistakes
- Type derivation traces in verbose mode
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from enum import Enum, auto
import re

from .syntax import SourceLocation
from .core import Value, VType, VConstructor, VPi, VNeutral, NVar
from .colors import Colors


class ErrorKind(Enum):
    """Categories of errors for suggestion generation."""
    TYPE_MISMATCH = auto()
    UNKNOWN_VARIABLE = auto()
    UNKNOWN_CONSTRUCTOR = auto()
    WRONG_ARITY = auto()
    IMPLICIT_MISMATCH = auto()
    MISSING_TYPE_ANNOTATION = auto()
    INVALID_PATTERN = auto()
    MODULE_NOT_FOUND = auto()
    CONSTRAINT_NOT_SATISFIED = auto()


@dataclass
class ErrorContext:
    """Context information for an error."""
    source_code: Optional[str] = None
    filename: Optional[str] = None
    location: Optional[SourceLocation] = None
    kind: Optional[ErrorKind] = None
    # Additional context for generating suggestions
    expected: Optional[str] = None
    actual: Optional[str] = None
    available_names: Optional[List[str]] = None
    similar_names: Optional[List[str]] = None


@dataclass
class TypeDerivation:
    """A step in type derivation for verbose output."""
    description: str
    location: Optional[SourceLocation]
    context: Dict[str, str]  # Variable name -> type
    result: Optional[str]


class TypeDerivationTrace:
    """Accumulates type derivation steps for verbose output."""
    
    def __init__(self):
        self.steps: List[TypeDerivation] = []
        self.enabled = False
    
    def add_step(self, description: str, location: Optional[SourceLocation] = None,
                 context: Optional[Dict[str, str]] = None, result: Optional[str] = None):
        """Add a derivation step."""
        if self.enabled:
            self.steps.append(TypeDerivation(
                description=description,
                location=location,
                context=context or {},
                result=result
            ))
    
    def format(self) -> str:
        """Format the trace for display."""
        if not self.steps:
            return ""
        
        lines = [Colors.bold("\nType Derivation Trace:")]
        for i, step in enumerate(self.steps, 1):
            lines.append(f"\n{Colors.dim(f'Step {i}:')} {step.description}")
            
            if step.location:
                lines.append(f"  {Colors.dim('at')} {format_location(step.location)}")
            
            if step.context:
                lines.append(f"  {Colors.dim('context:')}")
                for var, ty in step.context.items():
                    lines.append(f"    {Colors.var_name(var)} : {Colors.type_name(ty)}")
            
            if step.result:
                lines.append(f"  {Colors.dim('result:')} {Colors.type_name(step.result)}")
        
        return "\n".join(lines)


# Global trace instance
_trace = TypeDerivationTrace()


def get_trace() -> TypeDerivationTrace:
    """Get the global type derivation trace."""
    return _trace


def enable_trace():
    """Enable type derivation tracing."""
    _trace.enabled = True


def disable_trace():
    """Disable type derivation tracing."""
    _trace.enabled = False


def clear_trace():
    """Clear the type derivation trace."""
    _trace.steps = []


def format_location(location: Optional[SourceLocation]) -> str:
    """Format a source location for display."""
    if not location:
        return "<unknown location>"
    
    parts = []
    if location.filename:
        parts.append(location.filename)
    parts.append(f"{location.line}:{location.column}")
    
    return ":".join(parts)


def show_source_context(source_code: str, location: SourceLocation, 
                       error_message: str = "", context_lines: int = 2) -> str:
    """Display source code context around an error location."""
    lines = source_code.split('\n')
    
    if not (0 < location.line <= len(lines)):
        return ""
    
    output = []
    
    # Error message
    if error_message:
        output.append(Colors.error(f"Error: {error_message}"))
    
    # Location
    output.append(f"{Colors.dim('at')} {format_location(location)}")
    output.append("")
    
    # Source context
    start_line = max(0, location.line - context_lines - 1)
    end_line = min(len(lines), location.line + context_lines)
    
    for i in range(start_line, end_line):
        line_num = i + 1
        line_content = lines[i]
        
        if line_num == location.line:
            # Highlight the error line
            output.append(f"{Colors.error('→')} {line_num:4d} │ {line_content}")
            
            # Show error position
            if location.column > 0:
                spaces = ' ' * (7 + location.column - 1)
                marker = Colors.error('^' + '~' * min(len(line_content) - location.column, 10))
                output.append(f"       │ {spaces}{marker}")
        else:
            output.append(f"  {line_num:4d} │ {line_content}")
    
    return '\n'.join(output)


def format_type_for_display(type_val: Value) -> str:
    """Format a type value for user-friendly display."""
    if isinstance(type_val, VType):
        return "Type" if type_val.level.n == 0 else f"Type{type_val.level.n}"
    
    elif isinstance(type_val, VConstructor):
        if type_val.args:
            args_str = " ".join(format_type_for_display(arg) for arg in type_val.args)
            return f"{type_val.name} {args_str}"
        return type_val.name
    
    elif isinstance(type_val, VPi):
        domain_str = format_type_for_display(type_val.domain)
        
        # For simple function types, use arrow notation
        if not type_val.implicit and type_val.name == "_":
            # Create a dummy value to peek at codomain
            dummy_val = VNeutral(NVar(9999))
            codomain = type_val.codomain_closure.apply(dummy_val)
            codomain_str = format_type_for_display(codomain)
            return f"{domain_str} → {codomain_str}"
        
        # For dependent/implicit types, show the binder
        elif type_val.implicit:
            return f"{{{type_val.name} : {domain_str}}} → ..."
        else:
            return f"({type_val.name} : {domain_str}) → ..."
    
    elif isinstance(type_val, VNeutral):
        if isinstance(type_val.neutral, NVar):
            return f"?{type_val.neutral.level}"
        else:
            return "<neutral>"
    
    else:
        return str(type_val)


def suggest_similar_names(name: str, available_names: List[str], max_suggestions: int = 3) -> List[str]:
    """Find similar names using edit distance."""
    suggestions = []
    
    for available in available_names:
        distance = edit_distance(name, available)
        if distance <= 2:  # Max edit distance of 2
            suggestions.append((distance, available))
    
    # Sort by distance and return top suggestions
    suggestions.sort(key=lambda x: x[0])
    return [name for _, name in suggestions[:max_suggestions]]


def edit_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one character longer
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def generate_suggestion(error_context: ErrorContext) -> Optional[str]:
    """Generate a helpful suggestion based on the error context."""
    if not error_context.kind:
        return None
    
    suggestions = []
    
    if error_context.kind == ErrorKind.UNKNOWN_VARIABLE:
        if error_context.similar_names:
            names = ", ".join(f"'{name}'" for name in error_context.similar_names[:3])
            suggestions.append(f"Did you mean: {names}?")
        
        if error_context.actual and error_context.actual[0].isupper():
            suggestions.append("Note: Constructor names must start with an uppercase letter")
    
    elif error_context.kind == ErrorKind.TYPE_MISMATCH:
        if error_context.expected and error_context.actual:
            # Check for common type mismatches
            if "Nat" in error_context.expected and "String" in error_context.actual:
                suggestions.append("Use 'show' to convert numbers to strings")
            elif "List" in error_context.expected and "[" in error_context.actual:
                suggestions.append("List literals need a type annotation: [1, 2, 3] : List Nat")
            elif "->" in error_context.expected and error_context.actual in ["Nat", "Bool", "String"]:
                suggestions.append("This looks like a function type. Did you forget to apply an argument?")
    
    elif error_context.kind == ErrorKind.WRONG_ARITY:
        suggestions.append("Check the number of arguments - use ':type' in the REPL to see the expected type")
    
    elif error_context.kind == ErrorKind.IMPLICIT_MISMATCH:
        suggestions.append("Use {} for implicit arguments, () for explicit arguments")
        suggestions.append("Example: id {Nat} 42 applies id with implicit type argument Nat")
    
    elif error_context.kind == ErrorKind.MISSING_TYPE_ANNOTATION:
        suggestions.append("Add a type annotation to help type inference")
        suggestions.append("Example: \\x : Nat -> x + 1")
    
    elif error_context.kind == ErrorKind.MODULE_NOT_FOUND:
        suggestions.append("Check that the module file exists and is in the module path")
        suggestions.append("Module files should have the .ai extension")
    
    elif error_context.kind == ErrorKind.CONSTRAINT_NOT_SATISFIED:
        suggestions.append("Ensure the type has an instance of the required type class")
        suggestions.append("You may need to import the instance or define it")
    
    if suggestions:
        return "\n".join(f"{Colors.hint('Hint:')} {s}" for s in suggestions)
    
    return None


class AiLangError(Exception):
    """Base class for all ai-lang errors with enhanced reporting."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None):
        super().__init__(message)
        self.context = context or ErrorContext()
        self._formatted_message = None
    
    def format_error(self) -> str:
        """Format the error with context and suggestions."""
        if self._formatted_message:
            return self._formatted_message
        
        parts = []
        
        # Show source context if available
        if self.context.source_code and self.context.location:
            parts.append(show_source_context(
                self.context.source_code,
                self.context.location,
                str(self)
            ))
        else:
            # Simple error format
            parts.append(Colors.error(f"Error: {self}"))
            if self.context.location:
                parts.append(f"{Colors.dim('at')} {format_location(self.context.location)}")
        
        # Add suggestions
        suggestion = generate_suggestion(self.context)
        if suggestion:
            parts.append("")
            parts.append(suggestion)
        
        # Add type derivation trace if enabled
        trace_output = get_trace().format()
        if trace_output:
            parts.append(trace_output)
        
        self._formatted_message = '\n'.join(parts)
        return self._formatted_message


class TypeCheckError(AiLangError):
    """Type checking error with enhanced reporting."""
    pass


class EvaluationError(AiLangError):
    """Runtime evaluation error with enhanced reporting."""
    pass


class ParseError(AiLangError):
    """Parse error with enhanced reporting."""
    pass


class LexError(AiLangError):
    """Lexical error with enhanced reporting."""
    pass