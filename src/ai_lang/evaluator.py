"""Evaluator for ai-lang.

This module implements the interpreter that executes ai-lang programs
after they have been type checked.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Tuple
import sys

from .syntax import *
from .core import *
from .typechecker import TypeChecker, Context
from .errors import EvalError


@dataclass
class PatternMatch:
    """Result of pattern matching."""
    bindings: Dict[str, Value]
    matched: bool


@dataclass
class Evaluator:
    """Evaluator state."""
    type_checker: TypeChecker
    global_env: Dict[str, Value]
    trace: bool = False
    
    def __init__(self, type_checker: TypeChecker, trace: bool = False):
        self.type_checker = type_checker
        self.global_env = {}
        self.trace = trace
        
        # Initialize built-in constructors
        self._init_builtins()
    
    def _init_builtins(self):
        """Initialize built-in values."""
        # Add data constructors as values
        for name, info in self.type_checker.constructors.items():
            # Create constructor value
            if isinstance(info.type, VPi):
                # Constructor is a function
                self.global_env[name] = self._make_constructor_function(name, info.type)
            else:
                # Constructor is a simple value
                self.global_env[name] = VConstructor(name, [])
    
    def _make_constructor_function(self, name: str, type: Value) -> Value:
        """Create a constructor function value."""
        # Count the number of arguments
        arg_count = 0
        current_type = type
        while isinstance(current_type, VPi):
            arg_count += 1
            # We need a dummy value to apply
            dummy = VNeutral(NVar(0))
            current_type = current_type.codomain_closure.apply(dummy)
        
        # Create a curried constructor function
        def make_constructor_closure(args_so_far: List[Value]) -> Value:
            remaining = arg_count - len(args_so_far)
            if remaining == 0:
                return VConstructor(name, args_so_far)
            else:
                # Return a lambda that takes the next argument
                return VLambda(
                    f"arg{len(args_so_far)}", 
                    ClosureFunc(lambda v: make_constructor_closure(args_so_far + [v])),
                    False
                )
        
        return make_constructor_closure([])
    
    def eval_module(self, module: Module) -> None:
        """Evaluate a module, adding definitions to the global environment."""
        for decl in module.declarations:
            self.eval_declaration(decl)
    
    def eval_declaration(self, decl: Declaration) -> None:
        """Evaluate a declaration."""
        if isinstance(decl, FunctionDef):
            # Create a function value from the clauses
            if len(decl.clauses) == 1 and not decl.clauses[0].patterns:
                # Simple constant definition - convert and evaluate
                body_term = self._convert_expr_to_term(decl.clauses[0].body)
                value = self._eval_term(body_term)
                self.global_env[decl.name.value] = value
            else:
                # Function with pattern matching
                self.global_env[decl.name.value] = self._make_function(decl)
        
        # Data declarations and type signatures don't need evaluation
    
    def _make_function(self, func_def: FunctionDef) -> Value:
        """Create a function value from a function definition."""
        # Get the function's type
        func_type = self.type_checker.global_types.get(func_def.name.value)
        if not func_type:
            raise EvalError(f"No type found for function {func_def.name.value}")
        
        # Convert all clauses to terms ahead of time
        converted_clauses = []
        for clause in func_def.clauses:
            # Create a context with pattern variables
            ctx = Context()
            
            # First, add implicit parameters from the function type
            current_type = func_type
            implicit_param_names = []
            while isinstance(current_type, VPi) and current_type.implicit:
                implicit_param_names.append(current_type.name)
                ctx = ctx.extend(current_type.name, current_type.domain)
                # Create a dummy value to apply
                var_val = VNeutral(NVar(len(ctx.environment.values) - 1))
                current_type = current_type.codomain_closure.apply(var_val)
            
            # Then add pattern variables
            pattern_vars = []
            for pattern in clause.patterns:
                pattern_vars.extend(self._collect_pattern_vars(pattern))
            
            # Add pattern variables to context
            for var in pattern_vars:
                # Dummy type for now - we'd need proper pattern types
                ctx = ctx.extend(var, VType(Level(0)))
            
            # Convert body to term
            body_term = self.type_checker.convert_expr(clause.body, ctx)
            converted_clauses.append((clause.patterns, body_term, pattern_vars))
        
        # Create a function that pattern matches on its arguments
        def apply_clauses(args: List[Value]) -> Value:
            # Separate implicit and explicit arguments
            # The first len(implicit_param_names) args are implicit
            implicit_args = args[:len(implicit_param_names)]
            explicit_args = args[len(implicit_param_names):]
            
            # Try each clause
            for patterns, body_term, pattern_vars in converted_clauses:
                if len(explicit_args) < len(patterns):
                    # Not enough arguments yet, return a partial application
                    return self._make_partial_function(func_def, args)
                
                # Try to match the patterns against explicit arguments only
                match = self._match_patterns(patterns, explicit_args[:len(patterns)])
                if match.matched:
                    # Create environment with pattern bindings
                    env = Environment()
                    
                    # First add implicit parameters (use the actual implicit arguments)
                    for i, _ in enumerate(implicit_param_names):
                        if i < len(implicit_args):
                            env = env.extend(implicit_args[i])
                        else:
                            # Shouldn't happen if type checking is correct
                            env = env.extend(VNeutral(NVar(1000 + i)))
                    
                    # Then add bindings in the same order as pattern vars
                    # (they were collected in the right order)
                    for var in pattern_vars:
                        if var in match.bindings:
                            env = env.extend(match.bindings[var])
                    
                    # Evaluate the body term
                    remaining_args = explicit_args[len(patterns):]
                    # Debug output disabled
                    # print(f"DEBUG EVAL: Evaluating body {body_term} with env size {len(env.values)}")
                    # print(f"DEBUG EVAL: Implicit params: {implicit_param_names}, pattern vars: {pattern_vars}")
                    # print(f"DEBUG EVAL: Pattern bindings: {match.bindings}")
                    # print(f"DEBUG EVAL: Implicit args: {implicit_args}")
                    # print(f"DEBUG EVAL: Explicit args: {explicit_args}")
                    # for i, v in enumerate(env.values):
                    #     print(f"DEBUG EVAL: env[{i}] = {v}")
                    result = self._eval_term(body_term, env)
                    
                    # Apply any remaining arguments
                    for arg in remaining_args:
                        result = apply(result, arg)
                    
                    return result
            
            raise EvalError(f"No matching clause for {func_def.name.value}")
        
        # Return a function value that applies the clauses
        return self._curry_function(apply_clauses, func_type)
    
    def _curry_function(self, f: callable, type: Value) -> Value:
        """Create a curried function from a multi-argument function."""
        if isinstance(type, VPi):
            # Create a lambda that takes one argument
            return VLambda(
                type.name,
                ClosureFunc(lambda v: self._curry_function_helper(f, [v], type.codomain_closure.apply(v))),
                type.implicit
            )
        else:
            # No arguments, just call the function
            return f([])
    
    def _curry_function_helper(self, f: callable, args: List[Value], remaining_type: Value) -> Value:
        """Helper for creating curried functions."""
        if isinstance(remaining_type, VPi):
            # More arguments to go
            return VLambda(
                remaining_type.name,
                ClosureFunc(lambda v: self._curry_function_helper(
                    f, args + [v], remaining_type.codomain_closure.apply(v)
                )),
                remaining_type.implicit
            )
        else:
            # All arguments collected, call the function
            return f(args)
    
    def _make_partial_function(self, func_def: FunctionDef, args: List[Value]) -> Value:
        """Create a partial application of a function."""
        # Get the remaining type after applying the given arguments
        func_type = self.type_checker.global_types.get(func_def.name.value)
        if not func_type:
            raise EvalError(f"No type found for function {func_def.name.value}")
        
        # Apply the type to the arguments to get the remaining type
        remaining_type = func_type
        for arg in args:
            if isinstance(remaining_type, VPi):
                remaining_type = remaining_type.codomain_closure.apply(arg)
            else:
                raise EvalError(f"Too many arguments to {func_def.name.value}")
        
        # Create a lambda that takes the remaining arguments
        def apply_remaining(remaining_args: List[Value]) -> Value:
            all_args = args + remaining_args
            return self._make_function(func_def).apply(all_args)
        
        return self._curry_function(apply_remaining, remaining_type)
    
    def _match_patterns(self, patterns: List[Pattern], values: List[Value]) -> PatternMatch:
        """Match a list of patterns against values."""
        if len(patterns) != len(values):
            return PatternMatch({}, False)
        
        bindings = {}
        for pattern, value in zip(patterns, values):
            match = self._match_pattern(pattern, value)
            if not match.matched:
                return PatternMatch({}, False)
            
            # Check for conflicting bindings
            for name, val in match.bindings.items():
                if name in bindings:
                    # TODO: Check that values are equal
                    pass
                else:
                    bindings[name] = val
        
        return PatternMatch(bindings, True)
    
    def _collect_pattern_vars(self, pattern: Pattern) -> List[str]:
        """Collect all variable names from a pattern."""
        if isinstance(pattern, PatternVar):
            return [pattern.name.value]
        elif isinstance(pattern, PatternConstructor):
            vars = []
            for arg in pattern.args:
                vars.extend(self._collect_pattern_vars(arg))
            return vars
        else:
            return []
    
    def _match_pattern(self, pattern: Pattern, value: Value) -> PatternMatch:
        """Match a single pattern against a value."""
        if isinstance(pattern, PatternVar):
            # Variable pattern always matches
            return PatternMatch({pattern.name.value: value}, True)
        
        elif isinstance(pattern, PatternConstructor):
            # Constructor pattern
            if isinstance(value, VConstructor):
                if value.name == pattern.constructor.value:
                    # Match constructor arguments
                    if len(pattern.args) != len(value.args):
                        return PatternMatch({}, False)
                    
                    return self._match_patterns(pattern.args, value.args)
            
            return PatternMatch({}, False)
        
        elif isinstance(pattern, PatternLiteral):
            # Literal pattern
            if isinstance(value, VNat) and isinstance(pattern.value, int):
                return PatternMatch({}, value.value == pattern.value)
            elif isinstance(value, VBool) and isinstance(pattern.value, bool):
                return PatternMatch({}, value.value == pattern.value)
            elif isinstance(value, VString) and isinstance(pattern.value, str):
                return PatternMatch({}, value.value == pattern.value)
            
            return PatternMatch({}, False)
        
        elif isinstance(pattern, PatternWildcard):
            # Wildcard always matches
            return PatternMatch({}, True)
        
        else:
            raise EvalError(f"Unknown pattern type: {type(pattern)}")
    
    def eval_expr(self, expr: Expr, local_env: Dict[str, Value]) -> Value:
        """Evaluate an expression to a value."""
        if self.trace:
            print(f"Evaluating: {expr}")
        
        if isinstance(expr, Var):
            # Look up variable
            if expr.name.value in local_env:
                return local_env[expr.name.value]
            elif expr.name.value in self.global_env:
                return self.global_env[expr.name.value]
            elif expr.name.value in self.type_checker.constructors:
                # It's a constructor - convert and evaluate
                term = TConstructor(expr.name.value, [])
                return self._eval_term(term)
            elif expr.name.value in self.type_checker.global_types:
                # It's a global function - get from global_env
                if expr.name.value in self.global_env:
                    return self.global_env[expr.name.value]
                else:
                    raise EvalError(f"Function {expr.name.value} not yet defined")
            else:
                raise EvalError(f"Undefined variable: {expr.name.value}")
        
        elif isinstance(expr, Literal):
            # Evaluate literal
            if isinstance(expr.value, int):
                return VNat(expr.value)
            elif isinstance(expr.value, bool):
                return VBool(expr.value)
            elif isinstance(expr.value, str):
                return VString(expr.value)
            else:
                raise EvalError(f"Unknown literal type: {type(expr.value)}")
        
        elif isinstance(expr, Lambda):
            # Create closure
            return VLambda(
                expr.param.value,
                ClosureExpr(expr.body, expr.param.value, local_env, self),
                expr.implicit
            )
        
        elif isinstance(expr, App):
            # Evaluate function and argument
            fun = self.eval_expr(expr.function, local_env)
            arg = self.eval_expr(expr.argument, local_env)
            
            # Apply function to argument
            result = apply(fun, arg)
            
            if self.trace:
                print(f"Applied {fun} to {arg} = {result}")
            
            return result
        
        elif isinstance(expr, Let):
            # Evaluate let binding
            value = self.eval_expr(expr.value, local_env)
            new_env = local_env.copy()
            new_env[expr.name.value] = value
            return self.eval_expr(expr.body, new_env)
        
        elif isinstance(expr, Case):
            # Evaluate case expression
            scrutinee = self.eval_expr(expr.scrutinee, local_env)
            
            # Try each branch
            for branch in expr.branches:
                match = self._match_pattern(branch.pattern, scrutinee)
                if match.matched:
                    # Evaluate branch body with pattern bindings
                    new_env = local_env.copy()
                    new_env.update(match.bindings)
                    return self.eval_expr(branch.body, new_env)
            
            raise EvalError(f"No matching case for value: {scrutinee}")
        
        else:
            raise EvalError(f"Cannot evaluate expression: {type(expr)}")
    
    def _convert_expr_to_term(self, expr: Expr) -> Term:
        """Convert an AST expression to a core term with elaboration."""
        # Use the type checker's conversion
        ctx = Context()
        term = self.type_checker.convert_expr(expr, ctx)
        
        # Now perform type inference to get elaborated term
        try:
            # If this is an application, try to use delayed inference
            if isinstance(term, TApp) and not term.implicit:
                from .delayed_inference import infer_with_delayed_inference
                # Check if the base function has implicit parameters
                base_fun, args = self.type_checker._get_base_function(term)
                if isinstance(base_fun, TGlobal) and base_fun.name in self.type_checker.global_types:
                    base_type = self.type_checker.global_types[base_fun.name]
                    if isinstance(base_type, VPi) and base_type.implicit:
                        # Use delayed inference to get elaborated term
                        elaborated = infer_with_delayed_inference(term, ctx, self.type_checker)
                        return elaborated
        except:
            # If elaboration fails, just return the original term
            pass
        
        return term
    
    def _eval_term(self, term: Term, env: Optional[Environment] = None) -> Value:
        """Evaluate a core term with access to globals."""
        if env is None:
            env = Environment()
            
        if isinstance(term, TGlobal):
            # Look up global value
            if term.name in self.global_env:
                return self.global_env[term.name]
            else:
                raise EvalError(f"Undefined global: {term.name}")
        elif isinstance(term, TApp):
            # Handle application specially to support globals
            fun = self._eval_term(term.function, env)
            arg = self._eval_term(term.argument, env)
            return apply(fun, arg)
        elif isinstance(term, TLet):
            # Handle let with proper environment
            val = self._eval_term(term.value, env)
            new_env = env.extend(val)
            return self._eval_term(term.body, new_env)
        elif isinstance(term, TLambda):
            # Create a closure that uses _eval_term
            return VLambda(
                term.name,
                ClosureEvaluator(term.body, env, self),
                term.implicit
            )
        elif isinstance(term, TVar):
            return env.lookup(term.index)
        elif isinstance(term, TConstructor):
            arg_vals = [self._eval_term(arg, env) for arg in term.args]
            return VConstructor(term.name, arg_vals)
        elif isinstance(term, TLiteral):
            if isinstance(term.value, int):
                return VNat(term.value)
            elif isinstance(term.value, bool):
                return VBool(term.value)
            elif isinstance(term.value, str):
                return VString(term.value)
            else:
                raise TypeError(f"Unknown literal type: {type(term.value)}")
        elif isinstance(term, TType):
            return VType(term.level)
        elif isinstance(term, TPi):
            domain_val = self._eval_term(term.domain, env)
            codomain_closure = ClosureEvaluator(term.codomain, env, self)
            return VPi(term.name, domain_val, codomain_closure, term.implicit)
        else:
            # For other terms, use regular evaluation
            return term.eval(env)
    
    def eval_program(self, source: str) -> Optional[Value]:
        """Evaluate a complete program, returning the last value."""
        from .parser import parse
        from .typechecker import type_check_module
        
        # Parse
        module = parse(source)
        
        # Type check
        checker = type_check_module(module)
        
        # Create evaluator with type checker
        evaluator = Evaluator(checker, trace=self.trace)
        
        # Evaluate
        evaluator.eval_module(module)
        
        # Return the last expression value if any
        last_value = None
        for decl in module.declarations:
            if isinstance(decl, FunctionDef) and decl.name.value == "_main":
                # Evaluate main expression
                last_value = evaluator.eval_expr(decl.clauses[0].body, {})
        
        return last_value


@dataclass
class ClosureFunc:
    """Closure with a Python function."""
    func: callable
    
    def apply(self, value: Value) -> Value:
        """Apply closure to a value."""
        return self.func(value)


@dataclass
class ClosureExpr:
    """Closure with an expression and environment."""
    body: Expr
    param_name: str
    env: Dict[str, Value]
    evaluator: Evaluator
    
    def apply(self, value: Value) -> Value:
        """Apply closure to a value."""
        new_env = self.env.copy()
        new_env[self.param_name] = value
        return self.evaluator.eval_expr(self.body, new_env)


@dataclass  
class ClosureEvaluator:
    """Closure that evaluates a term with the evaluator."""
    body: Term
    env: Environment
    evaluator: Evaluator
    
    def apply(self, value: Value) -> Value:
        """Apply closure to a value."""
        new_env = self.env.extend(value)
        return self.evaluator._eval_term(self.body, new_env)


def pretty_print_value(value: Value) -> str:
    """Pretty print a value."""
    if isinstance(value, VNat):
        return str(value.value)
    
    elif isinstance(value, VBool):
        return "true" if value.value else "false"
    
    elif isinstance(value, VString):
        return f'"{value.value}"'
    
    elif isinstance(value, VConstructor):
        if value.name == "Z" and not value.args:
            return "0"
        elif value.name == "S" and len(value.args) == 1:
            # Pretty print successor
            inner = value.args[0]
            if isinstance(inner, VNat):
                return str(inner.value + 1)
            else:
                return f"S({pretty_print_value(inner)})"
        else:
            # Generic constructor
            if value.args:
                args_str = " ".join(pretty_print_value(arg) for arg in value.args)
                return f"({value.name} {args_str})"
            else:
                return value.name
    
    elif isinstance(value, VLambda):
        return f"<function>"
    
    elif isinstance(value, VPi):
        return f"<type>"
    
    elif isinstance(value, VType):
        return f"Type{value.level.n}" if value.level.n > 0 else "Type"
    
    else:
        return str(value)