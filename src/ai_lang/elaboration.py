"""Term elaboration for ai-lang.

This module handles the elaboration of terms during type checking,
ensuring that implicit arguments are properly inserted and stored
for later evaluation.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .syntax import *
from .core import *
from .errors import TypeCheckError


@dataclass
class ElaborationResult:
    """Result of elaborating a term."""
    term: Term
    type: Value


class Elaborator:
    """Handles term elaboration during type checking."""
    
    def __init__(self, type_checker: 'TypeChecker'):
        self.type_checker = type_checker
        self.elaborated_bodies: Dict[str, Term] = {}
    
    def elaborate_and_check(self, expr: Expr, expected_type: Optional[Value], ctx: 'Context') -> ElaborationResult:
        """Elaborate an expression and check/infer its type.
        
        This combines type checking with elaboration, ensuring that
        implicit arguments are properly inserted.
        """
        # Convert to term
        term = self.type_checker.convert_expr(expr, ctx)
        
        if expected_type is not None:
            # Check mode: elaborate against expected type
            elaborated_term = self.check_and_elaborate(term, expected_type, ctx)
            return ElaborationResult(elaborated_term, expected_type)
        else:
            # Infer mode: elaborate and infer type
            elaborated_term, inferred_type = self.infer_and_elaborate(term, ctx)
            return ElaborationResult(elaborated_term, inferred_type)
    
    def check_and_elaborate(self, term: Term, expected_type: Value, ctx: 'Context') -> Term:
        """Type check a term against expected type and return elaborated term."""
        # For now, just use the regular type checker
        # In a full implementation, this would track elaboration
        self.type_checker.check_type(term, expected_type, ctx)
        return term
    
    def infer_and_elaborate(self, term: Term, ctx: 'Context') -> Tuple[Term, Value]:
        """Infer type of a term and return elaborated term with its type."""
        if isinstance(term, TApp):
            # This is where we handle implicit argument insertion
            fun_type = self.type_checker.infer_type(term.function, ctx)
            
            # If we have implicit parameters and an explicit application,
            # we need to insert implicit arguments
            if isinstance(fun_type, VPi) and fun_type.implicit and not term.implicit:
                # Use delayed inference to get the elaborated term
                from .delayed_inference import infer_with_delayed_inference
                try:
                    elaborated_term = infer_with_delayed_inference(term, ctx, self.type_checker)
                    # Now infer the type of the elaborated term
                    result_type = self.type_checker.infer_type(elaborated_term, ctx)
                    return elaborated_term, result_type
                except:
                    # Fall back to regular inference
                    pass
            
            # Regular application
            result_type = self.type_checker.infer_type(term, ctx)
            return term, result_type
        else:
            # For other terms, just infer normally
            result_type = self.type_checker.infer_type(term, ctx)
            return term, result_type


def elaborate_function_body(name: str, body: Expr, ctx: 'Context', type_checker: 'TypeChecker') -> Term:
    """Elaborate a function body and store it for evaluation."""
    elaborator = Elaborator(type_checker)
    
    # Get the expected return type from the function type
    if name in type_checker.global_types:
        func_type = type_checker.global_types[name]
        # Skip through implicit and explicit parameters to get return type
        return_type = func_type
        while isinstance(return_type, VPi):
            # Apply dummy values to get to the return type
            dummy = VNeutral(NVar(1000))
            return_type = return_type.codomain_closure.apply(dummy)
        
        # Elaborate the body
        result = elaborator.elaborate_and_check(body, None, ctx)
        
        # Store the elaborated term
        type_checker.elaborated_terms[name] = result.term
        
        return result.term
    else:
        # No type signature, just convert
        term = type_checker.convert_expr(body, ctx)
        type_checker.elaborated_terms[name] = term
        return term