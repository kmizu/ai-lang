"""Totality checking for ai-lang.

This module implements:
1. Termination checking for recursive functions
2. Coverage checking for pattern matching
3. Positivity checking for data type definitions
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Set, Optional, Tuple, Union
from enum import Enum, auto

from .syntax import *
from .core import *
from .typechecker import TypeChecker, Context, DataConstructorInfo
from .errors import TypeCheckError


class TotalityError(TypeCheckError):
    """Error raised when totality checking fails."""
    pass


class TerminationError(TotalityError):
    """Error raised when termination checking fails."""
    pass


class CoverageError(TotalityError):
    """Error raised when pattern coverage checking fails."""
    pass


class PositivityError(TotalityError):
    """Error raised when positivity checking fails."""
    pass


# Termination Checking
# ====================

@dataclass
class CallGraph:
    """Represents the call graph for recursive functions."""
    # Map from function name to functions it calls
    calls: Dict[str, Set[str]]
    # Map from function name to recursive calls with their argument info
    recursive_calls: Dict[str, List['RecursiveCall']]
    

@dataclass
class RecursiveCall:
    """Information about a recursive function call."""
    function: str
    # For each argument position, track if it's structurally smaller
    arg_info: List['ArgumentInfo']
    location: Optional[SourceLocation] = None


class ArgumentInfo(Enum):
    """Information about how an argument changes in a recursive call."""
    SMALLER = auto()      # Structurally smaller (e.g., n in S n)
    EQUAL = auto()        # Same as original
    UNKNOWN = auto()      # Cannot determine
    NOT_COMPARABLE = auto() # Different type or not comparable


@dataclass
class TerminationChecker:
    """Checks termination of recursive functions."""
    type_checker: TypeChecker
    
    def check_module(self, module: Module) -> None:
        """Check termination for all functions in a module."""
        # Build call graph
        call_graph = self._build_call_graph(module)
        
        # Find strongly connected components (recursive function groups)
        sccs = self._find_sccs(call_graph)
        
        # Check each SCC for termination
        for scc in sccs:
            if len(scc) > 1 or (len(scc) == 1 and scc[0] in call_graph.calls.get(scc[0], set())):
                # This is a recursive function or mutually recursive group
                self._check_termination_scc(scc, call_graph, module)
    
    def _build_call_graph(self, module: Module) -> CallGraph:
        """Build the call graph for the module."""
        calls = {}
        recursive_calls = {}
        
        for decl in module.declarations:
            if isinstance(decl, FunctionDef):
                func_name = decl.name.value
                calls[func_name] = set()
                recursive_calls[func_name] = []
                
                # Analyze each clause
                for clause in decl.clauses:
                    self._analyze_calls(
                        func_name,
                        clause.body,
                        clause.patterns,
                        calls[func_name],
                        recursive_calls[func_name]
                    )
        
        return CallGraph(calls, recursive_calls)
    
    def _analyze_calls(self, current_func: str, expr: Expr, patterns: List[Pattern],
                      calls_set: Set[str], rec_calls: List[RecursiveCall]) -> None:
        """Analyze function calls in an expression."""
        if isinstance(expr, Var):
            # Check if this is a function call
            if expr.name.value in self.type_checker.global_types:
                calls_set.add(expr.name.value)
                
        elif isinstance(expr, App):
            # Get the function being called
            base_func, args = self._get_applied_function(expr)
            
            if isinstance(base_func, Var) and base_func.name.value == current_func:
                # This is a recursive call
                arg_info = self._analyze_recursive_args(patterns, args)
                rec_calls.append(RecursiveCall(current_func, arg_info))
            
            # Continue analyzing sub-expressions
            self._analyze_calls(current_func, expr.function, patterns, calls_set, rec_calls)
            self._analyze_calls(current_func, expr.argument, patterns, calls_set, rec_calls)
            
        elif isinstance(expr, Lambda):
            self._analyze_calls(current_func, expr.body, patterns, calls_set, rec_calls)
            
        elif isinstance(expr, Let):
            self._analyze_calls(current_func, expr.value, patterns, calls_set, rec_calls)
            self._analyze_calls(current_func, expr.body, patterns, calls_set, rec_calls)
            
        elif isinstance(expr, Case):
            self._analyze_calls(current_func, expr.scrutinee, patterns, calls_set, rec_calls)
            for branch in expr.branches:
                # For case branches, we need to track pattern refinements
                # TODO: Implement more sophisticated tracking of pattern refinements
                self._analyze_calls(current_func, branch.body, patterns, calls_set, rec_calls)
                
        # Add other expression types as needed
    
    def _get_applied_function(self, expr: Expr) -> Tuple[Expr, List[Expr]]:
        """Extract the base function and its arguments from a (possibly nested) application."""
        args = []
        current = expr
        
        while isinstance(current, App):
            args.insert(0, current.argument)
            current = current.function
            
        return current, args
    
    def _analyze_recursive_args(self, patterns: List[Pattern], args: List[Expr]) -> List[ArgumentInfo]:
        """Analyze how arguments change in a recursive call."""
        arg_info = []
        
        for i, (pattern, arg) in enumerate(zip(patterns, args)):
            info = self._compare_pattern_arg(pattern, arg)
            arg_info.append(info)
            
        # Pad with UNKNOWN if there are more arguments than patterns
        while len(arg_info) < len(args):
            arg_info.append(ArgumentInfo.UNKNOWN)
            
        return arg_info
    
    def _compare_pattern_arg(self, pattern: Pattern, arg: Expr) -> ArgumentInfo:
        """Compare a pattern with an argument to determine size relationship."""
        if isinstance(pattern, PatternVar):
            # Check if arg is a subterm of the pattern variable
            if isinstance(arg, Var) and arg.name.value == pattern.name.value:
                return ArgumentInfo.EQUAL
            elif self._is_subterm_of_pattern_var(arg, pattern.name.value):
                return ArgumentInfo.SMALLER
            elif self._is_larger_than_pattern_var(arg, pattern.name.value):
                # Check if argument is larger (e.g., S n when pattern is n)
                return ArgumentInfo.NOT_COMPARABLE
            else:
                return ArgumentInfo.UNKNOWN
                
        elif isinstance(pattern, PatternConstructor):
            # For constructor patterns, check if arg is a subterm
            if pattern.constructor.value == "S" and len(pattern.args) == 1:
                # Special case for successor pattern
                sub_pattern = pattern.args[0]
                if isinstance(sub_pattern, PatternVar) and isinstance(arg, Var):
                    if arg.name.value == sub_pattern.name.value:
                        return ArgumentInfo.SMALLER
                        
            # TODO: Handle other constructor patterns
            
        return ArgumentInfo.UNKNOWN
    
    def _is_larger_than_pattern_var(self, expr: Expr, var_name: str) -> bool:
        """Check if expr is structurally larger than the pattern variable."""
        # Check for constructor applications that contain the variable
        if isinstance(expr, App):
            base, args = self._get_applied_function(expr)
            if isinstance(base, Var) and base.name.value in self.type_checker.constructors:
                # This is a constructor application
                for arg in args:
                    if self._contains_var(arg, var_name):
                        return True
        elif isinstance(expr, Var) and expr.name.value in self.type_checker.constructors:
            # Single constructor with the variable would have been an App
            pass
        return False
    
    def _contains_var(self, expr: Expr, var_name: str) -> bool:
        """Check if an expression contains a variable."""
        if isinstance(expr, Var):
            return expr.name.value == var_name
        elif isinstance(expr, App):
            return self._contains_var(expr.function, var_name) or self._contains_var(expr.argument, var_name)
        elif isinstance(expr, Lambda):
            # Be careful with variable shadowing
            if expr.param.value == var_name:
                return False  # Shadowed
            return self._contains_var(expr.body, var_name)
        # Add other cases as needed
        return False
    
    def _is_subterm_of_pattern_var(self, expr: Expr, var_name: str) -> bool:
        """Check if expr is a structural subterm of the pattern variable."""
        # This is a simplified check - in practice we'd need more sophisticated analysis
        if isinstance(expr, Var):
            return expr.name.value == var_name
        elif isinstance(expr, App):
            # For now, we don't support extracting subterms from constructors
            # A proper implementation would handle pattern matching extractions
            # like getting 'n' from 'S n' when pattern matching
            pass
        return False
    
    def _find_sccs(self, call_graph: CallGraph) -> List[List[str]]:
        """Find strongly connected components using Tarjan's algorithm."""
        index_counter = [0]
        stack = []
        indices = {}
        lowlinks = {}
        on_stack = set()
        sccs = []
        
        def strongconnect(node):
            indices[node] = index_counter[0]
            lowlinks[node] = index_counter[0]
            index_counter[0] += 1
            stack.append(node)
            on_stack.add(node)
            
            for successor in call_graph.calls.get(node, set()):
                if successor not in indices:
                    strongconnect(successor)
                    lowlinks[node] = min(lowlinks[node], lowlinks[successor])
                elif successor in on_stack:
                    lowlinks[node] = min(lowlinks[node], indices[successor])
            
            if lowlinks[node] == indices[node]:
                scc = []
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    scc.append(w)
                    if w == node:
                        break
                sccs.append(scc)
        
        for node in call_graph.calls:
            if node not in indices:
                strongconnect(node)
                
        return sccs
    
    def _check_termination_scc(self, scc: List[str], call_graph: CallGraph, module: Module) -> None:
        """Check termination for a strongly connected component."""
        # For each function in the SCC, check that at least one argument
        # decreases in all recursive calls
        for func_name in scc:
            if func_name not in call_graph.recursive_calls:
                continue
                
            rec_calls = call_graph.recursive_calls[func_name]
            if not rec_calls:
                continue
                
            # Find which argument positions could be used for termination
            num_args = max(len(call.arg_info) for call in rec_calls) if rec_calls else 0
            decreasing_positions = []
            
            for pos in range(num_args):
                # Check if this position decreases in at least one call
                # and never increases
                has_decrease = False
                all_valid = True
                
                for call in rec_calls:
                    if pos < len(call.arg_info):
                        info = call.arg_info[pos]
                        if info == ArgumentInfo.SMALLER:
                            has_decrease = True
                        elif info not in (ArgumentInfo.SMALLER, ArgumentInfo.EQUAL):
                            all_valid = False
                            break
                
                if has_decrease and all_valid:
                    decreasing_positions.append(pos)
            
            if not decreasing_positions:
                # No decreasing argument found
                raise TerminationError(
                    f"Function '{func_name}' may not terminate. "
                    f"No structurally decreasing argument found in recursive calls."
                )


# Coverage Checking
# =================

@dataclass
class PatternMatrix:
    """Matrix of patterns for coverage checking."""
    rows: List[List[Pattern]]  # Each row is a list of patterns from a clause
    

@dataclass
class CoverageChecker:
    """Checks exhaustiveness and redundancy of pattern matching."""
    type_checker: TypeChecker
    
    def check_function(self, func_def: FunctionDef) -> None:
        """Check pattern coverage for a function definition."""
        if not func_def.clauses:
            return
            
        # Get the function type to know the expected patterns
        func_type = self.type_checker.global_types.get(func_def.name.value)
        if not func_type:
            return
            
        # Extract pattern matrix
        pattern_matrix = PatternMatrix([clause.patterns for clause in func_def.clauses])
        
        # Skip implicit parameters when counting expected patterns
        expected_patterns = 0
        current_type = func_type
        while isinstance(current_type, VPi):
            if not current_type.implicit:
                expected_patterns += 1
            dummy = VNeutral(NVar(0))
            current_type = current_type.codomain_closure.apply(dummy)
            if expected_patterns >= len(func_def.clauses[0].patterns):
                break
        
        # Get the types of each argument position
        arg_types = []
        current_type = func_type
        while isinstance(current_type, VPi) and len(arg_types) < expected_patterns:
            if not current_type.implicit:
                arg_types.append(current_type.domain)
            dummy = VNeutral(NVar(0))
            current_type = current_type.codomain_closure.apply(dummy)
        
        # Check exhaustiveness for each argument position
        if func_def.clauses and func_def.clauses[0].patterns:
            # For simplicity, check the first argument position
            first_arg_patterns = [clause.patterns[0] for clause in func_def.clauses if clause.patterns]
            if arg_types:
                missing = self._find_missing_patterns_for_type(first_arg_patterns, arg_types[0])
                if missing:
                    examples = ", ".join(self._pattern_to_string(p) for p in missing[:3])
                    raise CoverageError(
                        f"Non-exhaustive patterns in function '{func_def.name.value}'. "
                        f"Missing case: {examples}"
                    )
        
        # Check for redundant patterns
        for i, clause in enumerate(func_def.clauses[1:], 1):
            if self._is_redundant(pattern_matrix.rows[:i], clause.patterns):
                raise CoverageError(
                    f"Redundant pattern in function '{func_def.name.value}' at clause {i+1}"
                )
    
    def check_case(self, case_expr: Case, scrutinee_type: Value) -> None:
        """Check pattern coverage for a case expression."""
        # Extract pattern matrix (single column for case expressions)
        patterns = [[branch.pattern] for branch in case_expr.branches]
        pattern_matrix = PatternMatrix(patterns)
        
        # Check exhaustiveness
        missing = self._find_missing_patterns_for_type(
            [p[0] for p in patterns], 
            scrutinee_type
        )
        if missing:
            examples = ", ".join(self._pattern_to_string(p) for p in missing[:3])
            if len(missing) > 3:
                examples += ", ..."
            raise CoverageError(
                f"Non-exhaustive patterns in case expression. "
                f"Missing cases: {examples}"
            )
    
    def _find_missing_patterns(self, matrix: PatternMatrix, num_cols: int, 
                              func_type: Value) -> List[List[Pattern]]:
        """Find patterns that are not covered by the matrix."""
        # This is a simplified implementation
        # A full implementation would use the algorithm from
        # "Warnings for pattern matching" by Maranget
        
        if not matrix.rows:
            # Empty matrix - everything is missing
            # Generate a wildcard pattern for each column
            return [[PatternWildcard() for _ in range(num_cols)]]
        
        if num_cols == 0:
            # No columns left - if we got here, it's exhaustive
            return []
        
        # For now, we'll do a simple check for common cases
        first_col_patterns = [row[0] if row else PatternWildcard() for row in matrix.rows]
        
        # Check if we have all constructors for the first column
        # This is oversimplified - we'd need to know the type of each column
        return []
    
    def _find_missing_patterns_for_type(self, patterns: List[Pattern], 
                                       scrutinee_type: Value) -> List[Pattern]:
        """Find missing patterns for a specific type."""
        covered_constructors = set()
        has_wildcard = False
        has_var = False
        
        for pattern in patterns:
            if isinstance(pattern, PatternConstructor):
                covered_constructors.add(pattern.constructor.value)
            elif isinstance(pattern, PatternWildcard) or isinstance(pattern, PatternVar):
                has_wildcard = True
                has_var = isinstance(pattern, PatternVar)
        
        if has_wildcard:
            # Wildcard covers everything
            return []
        
        # Check what constructors are available for this type
        missing = []
        
        # Special handling for known types
        if isinstance(scrutinee_type, VConstructor):
            if scrutinee_type.name == "Bool":
                for ctor in ["True", "False"]:
                    if ctor not in covered_constructors:
                        missing.append(PatternConstructor(Name(ctor), []))
            elif scrutinee_type.name == "Nat":
                if "Z" not in covered_constructors:
                    missing.append(PatternConstructor(Name("Z"), []))
                if "S" not in covered_constructors:
                    # S takes an argument
                    missing.append(PatternConstructor(Name("S"), [PatternWildcard()]))
        
        return missing
    
    def _is_redundant(self, previous_patterns: List[List[Pattern]], 
                     new_pattern: List[Pattern]) -> bool:
        """Check if a pattern is redundant given previous patterns."""
        # Simplified check - a pattern is redundant if it's subsumed by previous patterns
        # This would need a proper implementation using pattern matching algorithms
        return False
    
    def _pattern_to_string(self, pattern: Union[Pattern, List[Pattern]]) -> str:
        """Convert a pattern to a string representation."""
        if isinstance(pattern, list):
            return " ".join(self._pattern_to_string(p) for p in pattern)
        elif isinstance(pattern, PatternVar):
            return pattern.name.value
        elif isinstance(pattern, PatternConstructor):
            if pattern.args:
                args_str = " ".join(self._pattern_to_string(arg) for arg in pattern.args)
                return f"({pattern.constructor.value} {args_str})"
            return pattern.constructor.value
        elif isinstance(pattern, PatternWildcard):
            return "_"
        elif isinstance(pattern, PatternLiteral):
            return repr(pattern.value)
        else:
            return str(pattern)


# Positivity Checking
# ===================

@dataclass
class PositivityChecker:
    """Checks strict positivity of data type definitions."""
    type_checker: TypeChecker
    
    def check_data_type(self, data_decl: DataDecl) -> None:
        """Check that a data type definition is strictly positive."""
        data_name = data_decl.name.value
        
        for ctor in data_decl.constructors:
            self._check_constructor_positivity(
                data_name, 
                ctor, 
                data_decl.type_params
            )
    
    def _check_constructor_positivity(self, data_name: str, ctor: Constructor,
                                    type_params: List[Name]) -> None:
        """Check that a constructor is strictly positive."""
        # Create context with type parameters
        ctx = Context()
        for param in type_params:
            # Add type parameter to context as a type variable
            ctx = ctx.extend(param.value, VType(Level(0)))
        
        # Convert constructor type to core representation for analysis
        ctor_type = self.type_checker.convert_type(ctor.type, ctx)
        ctor_val = ctor_type.eval(ctx.environment)
        
        # For constructors, we start in a special "constructor" context
        self._check_constructor_type(data_name, ctor_val)
    
    def _check_constructor_type(self, data_name: str, type_val: Value) -> None:
        """Check a constructor type for strict positivity."""
        if isinstance(type_val, VPi):
            # In constructor arguments, the data type can appear directly
            # but not in negative positions of nested function types
            self._check_constructor_arg(data_name, type_val.domain)
            
            # Check the rest of the constructor type
            dummy = VNeutral(NVar(0))
            codomain = type_val.codomain_closure.apply(dummy)
            self._check_constructor_type(data_name, codomain)
        elif isinstance(type_val, VConstructor):
            # The constructor should return the data type being defined
            if type_val.name != data_name:
                # This shouldn't happen in well-formed constructors
                pass
        # Other cases are fine
    
    def _check_constructor_arg(self, data_name: str, type_val: Value) -> None:
        """Check a constructor argument for strict positivity."""
        if isinstance(type_val, VPi):
            # Function type in constructor argument
            # The data type cannot appear in the domain (negative position)
            if self._contains_data_type(type_val.domain, data_name):
                raise PositivityError(
                    f"Data type '{data_name}' occurs negatively in constructor argument"
                )
            # Check codomain recursively
            dummy = VNeutral(NVar(0))
            codomain = type_val.codomain_closure.apply(dummy)
            self._check_constructor_arg(data_name, codomain)
        elif isinstance(type_val, VConstructor):
            # Direct occurrence of data type is OK in constructor arguments
            if type_val.name == data_name:
                pass  # This is fine (e.g., S : Nat -> Nat)
            # Check arguments of other type constructors
            for arg in type_val.args:
                self._check_constructor_arg(data_name, arg)
        # Other cases are fine
    
    def _check_type_positivity(self, data_name: str, type_val: Value,
                             positive: bool, seen: Set[str]) -> None:
        """Check positivity of a type value."""
        if isinstance(type_val, VPi):
            # For constructor types at the top level, we're checking if the
            # constructor returns the right type, so the codomain should be positive
            # The domain is in negative position
            self._check_type_positivity(data_name, type_val.domain, False, seen)
            
            # Check codomain (remains in positive position for constructors)
            dummy = VNeutral(NVar(0))
            codomain = type_val.codomain_closure.apply(dummy)
            self._check_type_positivity(data_name, codomain, positive, seen)
            
        elif isinstance(type_val, VConstructor):
            if type_val.name == data_name:
                # Occurrence of the data type being defined
                if not positive:
                    raise PositivityError(
                        f"Data type '{data_name}' occurs negatively in constructor"
                    )
            # Check arguments
            for arg in type_val.args:
                self._check_type_positivity(data_name, arg, positive, seen)
                
        elif isinstance(type_val, VNeutral):
            # Neutral values are fine - they don't contain our data type directly
            pass
            
        # Add other cases as needed
    
    def _contains_data_type(self, type_val: Value, data_name: str) -> bool:
        """Check if a type value contains references to the given data type."""
        if isinstance(type_val, VConstructor):
            if type_val.name == data_name:
                return True
            return any(self._contains_data_type(arg, data_name) for arg in type_val.args)
        elif isinstance(type_val, VPi):
            if self._contains_data_type(type_val.domain, data_name):
                return True
            # Check codomain with dummy value
            dummy = VNeutral(NVar(0))
            codomain = type_val.codomain_closure.apply(dummy)
            return self._contains_data_type(codomain, data_name)
        elif isinstance(type_val, VNeutral):
            # Could potentially contain the type, so be conservative
            return True
        else:
            return False


# Main Totality Checker
# ====================

@dataclass
class TotalityOptions:
    """Options for totality checking."""
    check_termination: bool = True
    check_coverage: bool = True
    check_positivity: bool = True


@dataclass 
class TotalityChecker:
    """Main totality checker that coordinates all checks."""
    type_checker: TypeChecker
    options: TotalityOptions
    
    def __init__(self, type_checker: TypeChecker, options: Optional[TotalityOptions] = None):
        self.type_checker = type_checker
        self.options = options or TotalityOptions()
        
        self.termination_checker = TerminationChecker(type_checker)
        self.coverage_checker = CoverageChecker(type_checker)
        self.positivity_checker = PositivityChecker(type_checker)
    
    def check_module(self, module: Module) -> List[str]:
        """Check totality for a module. Returns list of warnings."""
        warnings = []
        
        # Check positivity of data types
        if self.options.check_positivity:
            for decl in module.declarations:
                if isinstance(decl, DataDecl):
                    try:
                        self.positivity_checker.check_data_type(decl)
                    except PositivityError as e:
                        if self.options.check_positivity:
                            raise
                        warnings.append(f"Warning: {e}")
        
        # Check coverage of pattern matching
        if self.options.check_coverage:
            for decl in module.declarations:
                if isinstance(decl, FunctionDef):
                    try:
                        self.coverage_checker.check_function(decl)
                    except CoverageError as e:
                        if self.options.check_coverage:
                            raise
                        warnings.append(f"Warning: {e}")
        
        # Check termination of recursive functions
        if self.options.check_termination:
            try:
                self.termination_checker.check_module(module)
            except TerminationError as e:
                if self.options.check_termination:
                    raise
                warnings.append(f"Warning: {e}")
        
        return warnings
    
    def check_declaration(self, decl: Declaration) -> List[str]:
        """Check totality for a single declaration."""
        warnings = []
        
        if isinstance(decl, DataDecl) and self.options.check_positivity:
            self.positivity_checker.check_data_type(decl)
            
        elif isinstance(decl, FunctionDef) and self.options.check_coverage:
            self.coverage_checker.check_function(decl)
            
        return warnings


# Utility class for closure functions
@dataclass
class ClosureFunc:
    """Wrapper for Python functions to use as closures."""
    func: callable
    
    def apply(self, value: Value) -> Value:
        return self.func(value)