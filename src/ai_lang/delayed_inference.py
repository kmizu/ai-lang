"""Delayed inference for handling partial applications with implicit parameters.

This module implements a mechanism to delay implicit parameter inference
until we have enough information from subsequent applications.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .core import *
from .syntax import *
from .errors import TypeCheckError


@dataclass
class PartialApp:
    """Represents a partial application that may need implicit arguments."""
    base_function: Term
    implicit_count: int
    explicit_args: List[Term]
    function_type: Value
    
    def needs_more_args(self) -> bool:
        """Check if we need more arguments to infer all implicits."""
        # For const : {A : Type} -> {B : Type} -> A -> B -> A
        # We need at least 2 explicit args to infer both A and B
        return len(self.explicit_args) < self.implicit_count


def collect_full_application(term: Term) -> Tuple[Term, List[Term]]:
    """Collect all arguments from a nested application.
    
    Returns (base_function, [arg1, arg2, ...])
    """
    args = []
    current = term
    
    while isinstance(current, TApp) and not current.implicit:
        args.insert(0, current.argument)
        current = current.function
    
    return current, args


def infer_with_delayed_inference(
    term: TApp,
    ctx: 'Context',
    type_checker: 'TypeChecker'
) -> Term:
    """Infer implicit arguments using delayed inference.
    
    This handles cases like `const Z True` where we need both arguments
    to infer both implicit type parameters.
    """
    # First, collect the full application chain
    base_fun, all_args = collect_full_application(term)
    
    # Get the type of the base function
    base_type = type_checker.infer_type(base_fun, ctx)
    
    # Count implicit parameters
    implicit_count = 0
    temp_type = base_type
    while isinstance(temp_type, VPi) and temp_type.implicit:
        implicit_count += 1
        # Apply dummy to continue
        dummy = VNeutral(NVar(1000 + implicit_count))
        temp_type = temp_type.codomain_closure.apply(dummy)
    
    # print(f"DEBUG delayed: base_fun={base_fun}, args={all_args}, implicit_count={implicit_count}")
    
    if implicit_count == 0:
        # No implicit parameters, return as-is
        return term
    
    # Now we have all the information we need
    # Use constraint-based inference with all arguments
    from .constraints import infer_implicit_args_with_constraints
    
    try:
        implicit_args, substitution = infer_implicit_args_with_constraints(
            term, base_type, all_args, ctx, type_checker
        )
        
        # Build the result with implicit arguments first
        result = base_fun
        for imp_arg in implicit_args:
            result = TApp(result, imp_arg, implicit=True)
        
        # Then add explicit arguments
        for exp_arg in all_args:
            result = TApp(result, exp_arg, implicit=False)
        
        return result
        
    except TypeCheckError as e:
        # If we can't infer all parameters with the given arguments,
        # fall back to partial inference
        # print(f"DEBUG delayed: Failed full inference: {e}")
        
        # Try to infer as many as possible
        if len(all_args) >= 1:
            # At least try to infer the first implicit from the first arg
            try:
                # Use single argument inference
                implicit_args, _ = infer_implicit_args_with_constraints(
                    term, base_type, all_args[:1], ctx, type_checker
                )
                
                # Build partial result
                result = base_fun
                for imp_arg in implicit_args:
                    result = TApp(result, imp_arg, implicit=True)
                
                # Add first explicit argument
                result = TApp(result, all_args[0], implicit=False)
                
                # Add remaining arguments without inference
                for arg in all_args[1:]:
                    result = TApp(result, arg, implicit=False)
                
                return result
            except:
                pass
        
        # Complete failure - return original
        raise e