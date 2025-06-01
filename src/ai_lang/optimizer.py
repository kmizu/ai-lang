"""Optimization passes for ai-lang.

This module implements various optimization passes that transform the typed AST
to improve runtime performance. Each optimization is implemented as a separate
pass that can be enabled or disabled.
"""

from __future__ import annotations
from dataclasses import dataclass, replace
from typing import List, Dict, Set, Optional, Union, Tuple
from abc import ABC, abstractmethod
import copy

from .syntax import *
from .core import *
from .typechecker import TypeChecker, Context
from .errors import OptimizationError


class OptimizationPass(ABC):
    """Base class for optimization passes."""
    
    @abstractmethod
    def name(self) -> str:
        """Name of the optimization pass."""
        pass
    
    @abstractmethod
    def optimize_module(self, module: Module, type_checker: TypeChecker) -> Module:
        """Apply optimization to a module."""
        pass


@dataclass
class OptimizationContext:
    """Context for optimization passes."""
    bindings: Dict[str, Term]  # Variable name to term mapping
    used_vars: Set[str]  # Set of used variables
    inline_candidates: Dict[str, Term]  # Functions that can be inlined
    
    def __init__(self):
        self.bindings = {}
        self.used_vars = set()
        self.inline_candidates = {}
    
    def mark_used(self, name: str) -> None:
        """Mark a variable as used."""
        self.used_vars.add(name)
    
    def is_used(self, name: str) -> bool:
        """Check if a variable is used."""
        return name in self.used_vars
    
    def add_binding(self, name: str, term: Term) -> None:
        """Add a binding to the context."""
        self.bindings[name] = term
    
    def get_binding(self, name: str) -> Optional[Term]:
        """Get the binding for a variable."""
        return self.bindings.get(name)


class EtaReduction(OptimizationPass):
    """Eta-reduction: Convert \\x -> f x to f when x doesn't appear free in f."""
    
    def name(self) -> str:
        return "eta-reduction"
    
    def optimize_module(self, module: Module, type_checker: TypeChecker) -> Module:
        """Apply eta-reduction to all declarations in a module."""
        optimized_decls = []
        
        for decl in module.declarations:
            if isinstance(decl, FunctionDef):
                optimized_decl = self._optimize_function(decl, type_checker)
                optimized_decls.append(optimized_decl)
            else:
                optimized_decls.append(decl)
        
        return replace(module, declarations=optimized_decls)
    
    def _optimize_function(self, func: FunctionDef, type_checker: TypeChecker) -> FunctionDef:
        """Optimize a function definition."""
        optimized_clauses = []
        
        for clause in func.clauses:
            optimized_body = self._optimize_expr(clause.body)
            optimized_clause = replace(clause, body=optimized_body)
            optimized_clauses.append(optimized_clause)
        
        return replace(func, clauses=optimized_clauses)
    
    def _optimize_expr(self, expr: Expr) -> Expr:
        """Apply eta-reduction to an expression."""
        if isinstance(expr, Lambda):
            # Check if this is eta-reducible: \\x -> f x
            if isinstance(expr.body, App) and not expr.body.implicit:
                if isinstance(expr.body.argument, Var) and expr.body.argument.name.value == expr.param.value:
                    # Check if x appears free in f
                    if not self._appears_free(expr.param.value, expr.body.function):
                        # Eta-reduce: return f instead of \\x -> f x
                        return self._optimize_expr(expr.body.function)
            
            # Otherwise, optimize the body
            optimized_body = self._optimize_expr(expr.body)
            return replace(expr, body=optimized_body)
        
        elif isinstance(expr, App):
            optimized_fun = self._optimize_expr(expr.function)
            optimized_arg = self._optimize_expr(expr.argument)
            return replace(expr, function=optimized_fun, argument=optimized_arg)
        
        elif isinstance(expr, Let):
            optimized_value = self._optimize_expr(expr.value)
            optimized_body = self._optimize_expr(expr.body)
            return replace(expr, value=optimized_value, body=optimized_body)
        
        elif isinstance(expr, Case):
            optimized_scrutinee = self._optimize_expr(expr.scrutinee)
            optimized_branches = []
            for branch in expr.branches:
                optimized_branch_body = self._optimize_expr(branch.body)
                optimized_branch = replace(branch, body=optimized_branch_body)
                optimized_branches.append(optimized_branch)
            return replace(expr, scrutinee=optimized_scrutinee, branches=optimized_branches)
        
        else:
            # No optimization for other expressions
            return expr
    
    def _appears_free(self, var_name: str, expr: Expr) -> bool:
        """Check if a variable appears free in an expression."""
        if isinstance(expr, Var):
            return expr.name.value == var_name
        
        elif isinstance(expr, Lambda):
            # If the lambda binds this variable, it's not free in the body
            if expr.param.value == var_name:
                return False
            return self._appears_free(var_name, expr.body)
        
        elif isinstance(expr, App):
            return (self._appears_free(var_name, expr.function) or 
                   self._appears_free(var_name, expr.argument))
        
        elif isinstance(expr, Let):
            if expr.name.value == var_name:
                # Variable is bound by let, not free in body
                return self._appears_free(var_name, expr.value)
            return (self._appears_free(var_name, expr.value) or
                   self._appears_free(var_name, expr.body))
        
        elif isinstance(expr, Case):
            if self._appears_free(var_name, expr.scrutinee):
                return True
            
            for branch in expr.branches:
                # Check if pattern binds the variable
                if self._pattern_binds(var_name, branch.pattern):
                    continue
                if self._appears_free(var_name, branch.body):
                    return True
            
            return False
        
        elif isinstance(expr, Literal):
            return False
        
        else:
            # Conservative: assume it might appear
            return True
    
    def _pattern_binds(self, var_name: str, pattern: Pattern) -> bool:
        """Check if a pattern binds a variable."""
        if isinstance(pattern, PatternVar):
            return pattern.name.value == var_name
        elif isinstance(pattern, PatternConstructor):
            return any(self._pattern_binds(var_name, arg) for arg in pattern.args)
        else:
            return False


class DeadCodeElimination(OptimizationPass):
    """Remove unused bindings and functions."""
    
    def name(self) -> str:
        return "dead-code-elimination"
    
    def optimize_module(self, module: Module, type_checker: TypeChecker) -> Module:
        """Remove unused declarations from a module."""
        # First pass: collect all used names
        used_names = set()
        
        # Always keep exported names
        for export in module.exports:
            for name in export.names:
                used_names.add(name.value)
        
        # Always keep main function if it exists
        for decl in module.declarations:
            if isinstance(decl, FunctionDef) and decl.name.value == "main":
                used_names.add("main")
        
        # Analyze usage starting from used names
        changed = True
        while changed:
            changed = False
            for decl in module.declarations:
                if isinstance(decl, FunctionDef) and decl.name.value in used_names:
                    # Analyze function body for references
                    for clause in decl.clauses:
                        new_refs = self._collect_references(clause.body)
                        for ref in new_refs:
                            if ref not in used_names:
                                used_names.add(ref)
                                changed = True
                
                elif isinstance(decl, TypeSignature) and decl.name.value in used_names:
                    # Type signatures don't introduce new references
                    pass
                
                elif isinstance(decl, DataDecl):
                    # Keep data declarations if any constructor is used
                    for ctor in decl.constructors:
                        if ctor.name.value in used_names:
                            used_names.add(decl.name.value)
                            # Add all constructors of this type
                            for c in decl.constructors:
                                used_names.add(c.name.value)
                            break
        
        # Second pass: filter declarations
        optimized_decls = []
        for decl in module.declarations:
            if isinstance(decl, FunctionDef):
                if decl.name.value in used_names:
                    # Also optimize the function body to remove dead let bindings
                    optimized_decl = self._optimize_function(decl)
                    optimized_decls.append(optimized_decl)
            elif isinstance(decl, TypeSignature):
                if decl.name.value in used_names:
                    optimized_decls.append(decl)
            elif isinstance(decl, DataDecl):
                if decl.name.value in used_names:
                    optimized_decls.append(decl)
            else:
                # Keep other declarations (imports, exports, etc.)
                optimized_decls.append(decl)
        
        return replace(module, declarations=optimized_decls)
    
    def _optimize_function(self, func: FunctionDef) -> FunctionDef:
        """Optimize a function by removing dead code."""
        optimized_clauses = []
        
        for clause in func.clauses:
            optimized_body = self._optimize_expr(clause.body)
            optimized_clause = replace(clause, body=optimized_body)
            optimized_clauses.append(optimized_clause)
        
        return replace(func, clauses=optimized_clauses)
    
    def _optimize_expr(self, expr: Expr) -> Expr:
        """Remove dead code from an expression."""
        if isinstance(expr, Let):
            # Check if the let binding is used
            if self._appears_free(expr.name.value, expr.body):
                # Binding is used, optimize both value and body
                optimized_value = self._optimize_expr(expr.value)
                optimized_body = self._optimize_expr(expr.body)
                return replace(expr, value=optimized_value, body=optimized_body)
            else:
                # Binding is not used, remove it
                # But still evaluate the value for side effects (if any)
                # For now, just return the body
                return self._optimize_expr(expr.body)
        
        elif isinstance(expr, Lambda):
            optimized_body = self._optimize_expr(expr.body)
            return replace(expr, body=optimized_body)
        
        elif isinstance(expr, App):
            optimized_fun = self._optimize_expr(expr.function)
            optimized_arg = self._optimize_expr(expr.argument)
            return replace(expr, function=optimized_fun, argument=optimized_arg)
        
        elif isinstance(expr, Case):
            optimized_scrutinee = self._optimize_expr(expr.scrutinee)
            optimized_branches = []
            for branch in expr.branches:
                optimized_branch_body = self._optimize_expr(branch.body)
                optimized_branch = replace(branch, body=optimized_branch_body)
                optimized_branches.append(optimized_branch)
            return replace(expr, scrutinee=optimized_scrutinee, branches=optimized_branches)
        
        else:
            return expr
    
    def _collect_references(self, expr: Expr) -> Set[str]:
        """Collect all variable references in an expression."""
        refs = set()
        
        if isinstance(expr, Var):
            refs.add(expr.name.value)
        
        elif isinstance(expr, Lambda):
            # Don't include the bound variable
            body_refs = self._collect_references(expr.body)
            body_refs.discard(expr.param.value)
            refs.update(body_refs)
        
        elif isinstance(expr, App):
            refs.update(self._collect_references(expr.function))
            refs.update(self._collect_references(expr.argument))
        
        elif isinstance(expr, Let):
            refs.update(self._collect_references(expr.value))
            body_refs = self._collect_references(expr.body)
            body_refs.discard(expr.name.value)
            refs.update(body_refs)
        
        elif isinstance(expr, Case):
            refs.update(self._collect_references(expr.scrutinee))
            for branch in expr.branches:
                branch_refs = self._collect_references(branch.body)
                # Remove pattern-bound variables
                bound_vars = self._collect_pattern_vars(branch.pattern)
                for var in bound_vars:
                    branch_refs.discard(var)
                refs.update(branch_refs)
        
        return refs
    
    def _collect_pattern_vars(self, pattern: Pattern) -> Set[str]:
        """Collect all variables bound by a pattern."""
        vars = set()
        
        if isinstance(pattern, PatternVar):
            vars.add(pattern.name.value)
        elif isinstance(pattern, PatternConstructor):
            for arg in pattern.args:
                vars.update(self._collect_pattern_vars(arg))
        
        return vars
    
    def _appears_free(self, var_name: str, expr: Expr) -> bool:
        """Check if a variable appears free in an expression."""
        return var_name in self._collect_references(expr)


class Inlining(OptimizationPass):
    """Inline simple functions to reduce overhead."""
    
    def name(self) -> str:
        return "inlining"
    
    def __init__(self, max_inline_size: int = 10):
        """Initialize inlining pass.
        
        Args:
            max_inline_size: Maximum size of expressions to inline
        """
        self.max_inline_size = max_inline_size
    
    def optimize_module(self, module: Module, type_checker: TypeChecker) -> Module:
        """Apply inlining to a module."""
        # First, identify inline candidates
        inline_candidates = {}
        
        for decl in module.declarations:
            if isinstance(decl, FunctionDef):
                if self._is_inline_candidate(decl):
                    # Store the function body for inlining
                    if len(decl.clauses) == 1 and not decl.clauses[0].patterns:
                        # Simple function without pattern matching
                        inline_candidates[decl.name.value] = decl.clauses[0].body
        
        # Second, apply inlining to all functions
        optimized_decls = []
        
        for decl in module.declarations:
            if isinstance(decl, FunctionDef):
                optimized_decl = self._inline_in_function(decl, inline_candidates)
                optimized_decls.append(optimized_decl)
            else:
                optimized_decls.append(decl)
        
        return replace(module, declarations=optimized_decls)
    
    def _is_inline_candidate(self, func: FunctionDef) -> bool:
        """Check if a function is a good candidate for inlining."""
        # Only inline simple functions
        if len(func.clauses) != 1:
            return False
        
        clause = func.clauses[0]
        
        # Don't inline functions with pattern matching
        if clause.patterns:
            return False
        
        # Don't inline recursive functions
        if self._is_recursive(func.name.value, clause.body):
            return False
        
        # Check size
        size = self._expr_size(clause.body)
        return size <= self.max_inline_size
    
    def _is_recursive(self, func_name: str, expr: Expr) -> bool:
        """Check if an expression contains a recursive call."""
        if isinstance(expr, Var):
            return expr.name.value == func_name
        
        elif isinstance(expr, Lambda):
            return self._is_recursive(func_name, expr.body)
        
        elif isinstance(expr, App):
            return (self._is_recursive(func_name, expr.function) or
                   self._is_recursive(func_name, expr.argument))
        
        elif isinstance(expr, Let):
            return (self._is_recursive(func_name, expr.value) or
                   self._is_recursive(func_name, expr.body))
        
        elif isinstance(expr, Case):
            if self._is_recursive(func_name, expr.scrutinee):
                return True
            return any(self._is_recursive(func_name, branch.body) 
                      for branch in expr.branches)
        
        else:
            return False
    
    def _expr_size(self, expr: Expr) -> int:
        """Calculate the size of an expression."""
        if isinstance(expr, (Var, Literal)):
            return 1
        
        elif isinstance(expr, Lambda):
            return 1 + self._expr_size(expr.body)
        
        elif isinstance(expr, App):
            return 1 + self._expr_size(expr.function) + self._expr_size(expr.argument)
        
        elif isinstance(expr, Let):
            return 1 + self._expr_size(expr.value) + self._expr_size(expr.body)
        
        elif isinstance(expr, Case):
            size = 1 + self._expr_size(expr.scrutinee)
            for branch in expr.branches:
                size += 1 + self._expr_size(branch.body)
            return size
        
        else:
            return 1
    
    def _inline_in_function(self, func: FunctionDef, inline_candidates: Dict[str, Expr]) -> FunctionDef:
        """Apply inlining within a function."""
        optimized_clauses = []
        
        for clause in func.clauses:
            optimized_body = self._inline_expr(clause.body, inline_candidates)
            optimized_clause = replace(clause, body=optimized_body)
            optimized_clauses.append(optimized_clause)
        
        return replace(func, clauses=optimized_clauses)
    
    def _inline_expr(self, expr: Expr, inline_candidates: Dict[str, Expr]) -> Expr:
        """Inline function calls in an expression."""
        if isinstance(expr, Var):
            # Check if this is an inline candidate
            if expr.name.value in inline_candidates:
                # Return a copy of the inlined expression
                return copy.deepcopy(inline_candidates[expr.name.value])
            return expr
        
        elif isinstance(expr, App):
            # First, optimize the function and argument
            optimized_fun = self._inline_expr(expr.function, inline_candidates)
            optimized_arg = self._inline_expr(expr.argument, inline_candidates)
            
            # Check if we're applying an inline candidate
            if isinstance(optimized_fun, Var) and optimized_fun.name.value in inline_candidates:
                # Inline the function application
                inlined_body = copy.deepcopy(inline_candidates[optimized_fun.name.value])
                
                # If the inlined body is a lambda, perform beta reduction
                if isinstance(inlined_body, Lambda):
                    # Substitute the argument for the parameter
                    return self._substitute(inlined_body.body, inlined_body.param.value, optimized_arg)
                else:
                    # Just apply normally
                    return App(inlined_body, optimized_arg, expr.implicit)
            
            return replace(expr, function=optimized_fun, argument=optimized_arg)
        
        elif isinstance(expr, Lambda):
            optimized_body = self._inline_expr(expr.body, inline_candidates)
            return replace(expr, body=optimized_body)
        
        elif isinstance(expr, Let):
            optimized_value = self._inline_expr(expr.value, inline_candidates)
            optimized_body = self._inline_expr(expr.body, inline_candidates)
            return replace(expr, value=optimized_value, body=optimized_body)
        
        elif isinstance(expr, Case):
            optimized_scrutinee = self._inline_expr(expr.scrutinee, inline_candidates)
            optimized_branches = []
            for branch in expr.branches:
                optimized_branch_body = self._inline_expr(branch.body, inline_candidates)
                optimized_branch = replace(branch, body=optimized_branch_body)
                optimized_branches.append(optimized_branch)
            return replace(expr, scrutinee=optimized_scrutinee, branches=optimized_branches)
        
        else:
            return expr
    
    def _substitute(self, expr: Expr, var_name: str, value: Expr) -> Expr:
        """Substitute a value for a variable in an expression."""
        if isinstance(expr, Var):
            if expr.name.value == var_name:
                return copy.deepcopy(value)
            return expr
        
        elif isinstance(expr, Lambda):
            if expr.param.value == var_name:
                # Variable is shadowed, don't substitute in body
                return expr
            # Need to be careful about variable capture
            if self._appears_free(expr.param.value, value):
                # Would cause variable capture, need to rename
                # For simplicity, we'll skip this case
                return expr
            optimized_body = self._substitute(expr.body, var_name, value)
            return replace(expr, body=optimized_body)
        
        elif isinstance(expr, App):
            optimized_fun = self._substitute(expr.function, var_name, value)
            optimized_arg = self._substitute(expr.argument, var_name, value)
            return replace(expr, function=optimized_fun, argument=optimized_arg)
        
        elif isinstance(expr, Let):
            optimized_value = self._substitute(expr.value, var_name, value)
            if expr.name.value == var_name:
                # Variable is shadowed, don't substitute in body
                return replace(expr, value=optimized_value)
            optimized_body = self._substitute(expr.body, var_name, value)
            return replace(expr, value=optimized_value, body=optimized_body)
        
        elif isinstance(expr, Case):
            optimized_scrutinee = self._substitute(expr.scrutinee, var_name, value)
            optimized_branches = []
            for branch in expr.branches:
                # Check if pattern binds the variable
                if self._pattern_binds(var_name, branch.pattern):
                    # Variable is shadowed
                    optimized_branches.append(branch)
                else:
                    optimized_branch_body = self._substitute(branch.body, var_name, value)
                    optimized_branch = replace(branch, body=optimized_branch_body)
                    optimized_branches.append(optimized_branch)
            return replace(expr, scrutinee=optimized_scrutinee, branches=optimized_branches)
        
        else:
            return expr
    
    def _appears_free(self, var_name: str, expr: Expr) -> bool:
        """Check if a variable appears free in an expression."""
        if isinstance(expr, Var):
            return expr.name.value == var_name
        
        elif isinstance(expr, Lambda):
            if expr.param.value == var_name:
                return False
            return self._appears_free(var_name, expr.body)
        
        elif isinstance(expr, App):
            return (self._appears_free(var_name, expr.function) or 
                   self._appears_free(var_name, expr.argument))
        
        elif isinstance(expr, Let):
            if expr.name.value == var_name:
                return self._appears_free(var_name, expr.value)
            return (self._appears_free(var_name, expr.value) or
                   self._appears_free(var_name, expr.body))
        
        elif isinstance(expr, Case):
            if self._appears_free(var_name, expr.scrutinee):
                return True
            
            for branch in expr.branches:
                if self._pattern_binds(var_name, branch.pattern):
                    continue
                if self._appears_free(var_name, branch.body):
                    return True
            
            return False
        
        else:
            return False
    
    def _pattern_binds(self, var_name: str, pattern: Pattern) -> bool:
        """Check if a pattern binds a variable."""
        if isinstance(pattern, PatternVar):
            return pattern.name.value == var_name
        elif isinstance(pattern, PatternConstructor):
            return any(self._pattern_binds(var_name, arg) for arg in pattern.args)
        else:
            return False


@dataclass
class Optimizer:
    """Main optimizer that runs optimization passes."""
    passes: List[OptimizationPass]
    enabled: bool = True
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.passes = []
        
        # Default optimization passes
        if enabled:
            self.passes = [
                EtaReduction(),
                DeadCodeElimination(),
                Inlining(max_inline_size=10)
            ]
    
    def add_pass(self, pass_: OptimizationPass) -> None:
        """Add an optimization pass."""
        self.passes.append(pass_)
    
    def optimize(self, module: Module, type_checker: TypeChecker) -> Module:
        """Run all optimization passes on a module."""
        if not self.enabled:
            return module
        
        optimized = module
        for pass_ in self.passes:
            print(f"Running optimization pass: {pass_.name()}")
            optimized = pass_.optimize_module(optimized, type_checker)
        
        return optimized
    
    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable optimization."""
        self.enabled = enabled
    
    def set_passes(self, pass_names: List[str]) -> None:
        """Set which optimization passes to run."""
        available_passes = {
            "eta-reduction": EtaReduction(),
            "dead-code-elimination": DeadCodeElimination(),
            "inlining": Inlining(max_inline_size=10)
        }
        
        self.passes = []
        for name in pass_names:
            if name in available_passes:
                self.passes.append(available_passes[name])
            else:
                raise OptimizationError(f"Unknown optimization pass: {name}")


def optimize_module(module: Module, passes: Optional[List[str]] = None, 
                   type_checker: Optional[TypeChecker] = None) -> Module:
    """Optimize a module with the specified passes.
    
    Args:
        module: The module to optimize
        passes: List of pass names to run (e.g., ["eta", "inline", "dce"])
                If None, runs all default passes
        type_checker: Optional type checker instance. If None, creates a new one
        
    Returns:
        Optimized module
    """
    if type_checker is None:
        from .typechecker import type_check_module
        type_checker = type_check_module(module, return_checker=True)
    
    optimizer = Optimizer(enabled=True)
    
    if passes is not None:
        # Map short names to full names
        pass_mapping = {
            "eta": "eta-reduction",
            "dce": "dead-code-elimination", 
            "inline": "inlining"
        }
        
        full_names = []
        for p in passes:
            if p in pass_mapping:
                full_names.append(pass_mapping[p])
            else:
                full_names.append(p)
        
        optimizer.set_passes(full_names)
    
    return optimizer.optimize(module, type_checker)