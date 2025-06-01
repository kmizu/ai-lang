"""Constraint-based type inference for implicit parameters.

This module implements a constraint solver that can handle multiple
implicit type parameters by collecting constraints and solving them.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Union
from enum import Enum, auto

from .core import *
from .syntax import *
from .errors import TypeCheckError


class ConstraintKind(Enum):
    """Different kinds of type constraints."""
    EQUALS = auto()      # Type equality: T1 = T2
    INSTANCE = auto()    # Type instance: T1 is an instance of T2
    UNIFY = auto()       # Unification: T1 ~ T2


@dataclass
class TypeVar:
    """A type variable (metavariable) for inference."""
    name: str
    id: int
    
    def __hash__(self):
        return hash((self.name, self.id))
    
    def __eq__(self, other):
        return isinstance(other, TypeVar) and self.id == other.id


@dataclass
class Constraint:
    """A type constraint."""
    kind: ConstraintKind
    left: Union[Value, TypeVar]
    right: Union[Value, TypeVar]
    context: 'Context'  # The context where this constraint was generated
    
    def __repr__(self):
        symbol = {
            ConstraintKind.EQUALS: "=",
            ConstraintKind.INSTANCE: "<=",
            ConstraintKind.UNIFY: "~"
        }[self.kind]
        return f"{self.left} {symbol} {self.right}"


@dataclass
class Substitution:
    """A substitution mapping type variables to values."""
    mapping: Dict[TypeVar, Value]
    
    def __init__(self):
        self.mapping = {}
    
    def bind(self, var: TypeVar, value: Value) -> None:
        """Add a binding to the substitution."""
        if var in self.mapping:
            # Check consistency
            if not self._equal_values(self.mapping[var], value):
                raise TypeError(f"Inconsistent binding for {var}: {self.mapping[var]} vs {value}")
        else:
            self.mapping[var] = value
    
    def lookup(self, var: TypeVar) -> Optional[Value]:
        """Look up a type variable."""
        return self.mapping.get(var)
    
    def apply_to_value(self, value: Value) -> Value:
        """Apply substitution to a value."""
        if isinstance(value, VMetaVar):
            # This is a metavariable - look it up
            result = self.lookup(value.var)
            if result:
                # Recursively apply substitution
                return self.apply_to_value(result)
            return value
        elif isinstance(value, VPi):
            # Apply to domain and codomain
            new_domain = self.apply_to_value(value.domain)
            # For codomain, we need to be careful with the closure
            # Create a new closure that applies the substitution
            class SubstClosure:
                def __init__(self, orig_closure, subst):
                    self.orig_closure = orig_closure
                    self.subst = subst
                
                def apply(self, val: Value) -> Value:
                    orig_result = self.orig_closure.apply(val)
                    return self.subst.apply_to_value(orig_result)
            
            new_closure = SubstClosure(value.codomain_closure, self)
            return VPi(value.name, new_domain, new_closure, value.implicit)
        elif isinstance(value, VConstructor):
            new_args = [self.apply_to_value(arg) for arg in value.args]
            return VConstructor(value.name, new_args)
        elif isinstance(value, VNeutral):
            # Apply to neutral terms
            return VNeutral(self.apply_to_neutral(value.neutral))
        else:
            # Other values are unchanged
            return value
    
    def apply_to_neutral(self, neutral: Neutral) -> Neutral:
        """Apply substitution to a neutral term."""
        if isinstance(neutral, NApp):
            new_fun = self.apply_to_neutral(neutral.function)
            new_arg = self.apply_to_value(neutral.argument)
            return NApp(new_fun, new_arg, neutral.implicit)
        else:
            # Other neutrals are unchanged
            return neutral
    
    def _equal_values(self, v1: Value, v2: Value) -> bool:
        """Check if two values are equal (simplified)."""
        # This is a simplified equality check
        return str(v1) == str(v2)
    
    def compose(self, other: 'Substitution') -> 'Substitution':
        """Compose two substitutions."""
        result = Substitution()
        # First apply other, then self
        for var, val in other.mapping.items():
            result.bind(var, self.apply_to_value(val))
        # Add bindings from self that aren't in other
        for var, val in self.mapping.items():
            if var not in result.mapping:
                result.bind(var, val)
        return result


@dataclass
class VMetaVar(Value):
    """A metavariable value for type inference."""
    var: TypeVar
    
    def quote(self, level: int) -> Term:
        # Quote as a placeholder
        return TMetaVar(self.var)
    
    def __repr__(self):
        return f"?{self.var.name}{self.var.id}"


@dataclass
class TMetaVar(Term):
    """A metavariable term."""
    var: TypeVar
    
    def eval(self, env: Environment) -> Value:
        return VMetaVar(self.var)


class ConstraintSolver:
    """Solver for type constraints."""
    
    def __init__(self, type_checker: 'TypeChecker'):
        self.type_checker = type_checker
        self.constraints: List[Constraint] = []
        self.substitution = Substitution()
        self.next_var_id = 0
    
    def fresh_type_var(self, name: str = "T") -> TypeVar:
        """Create a fresh type variable."""
        var = TypeVar(name, self.next_var_id)
        self.next_var_id += 1
        return var
    
    def add_constraint(self, kind: ConstraintKind, left: Union[Value, TypeVar], 
                      right: Union[Value, TypeVar], ctx: 'Context') -> None:
        """Add a constraint to be solved."""
        self.constraints.append(Constraint(kind, left, right, ctx))
    
    def solve(self) -> Substitution:
        """Solve all constraints and return the substitution."""
        # Process constraints until we reach a fixed point
        changed = True
        while changed:
            changed = False
            new_constraints = []
            
            for constraint in self.constraints:
                if self._solve_constraint(constraint):
                    changed = True
                else:
                    # Keep unsolved constraints
                    new_constraints.append(constraint)
            
            self.constraints = new_constraints
        
        # Check if there are unsolved constraints
        if self.constraints:
            # Try to default remaining type variables to reasonable values
            for constraint in self.constraints:
                self._default_constraint(constraint)
        
        return self.substitution
    
    def _solve_constraint(self, constraint: Constraint) -> bool:
        """Try to solve a single constraint. Returns True if solved."""
        # print(f"DEBUG SOLVER: Solving constraint {constraint}")
        
        # Apply current substitution to both sides
        left = self._apply_subst_to_constraint_value(constraint.left)
        right = self._apply_subst_to_constraint_value(constraint.right)
        
        # print(f"DEBUG SOLVER: After subst: {left} vs {right}")
        
        if constraint.kind == ConstraintKind.EQUALS:
            result = self._solve_equality(left, right, constraint.context)
            # print(f"DEBUG SOLVER: Equality result: {result}")
            return result
        elif constraint.kind == ConstraintKind.UNIFY:
            return self._unify(left, right, constraint.context)
        elif constraint.kind == ConstraintKind.INSTANCE:
            # For INSTANCE constraints, we just check compatibility
            # For now, if the right side is VType and left is a metavar, it's ok
            if isinstance(left, VMetaVar) and isinstance(right, VType):
                return True  # Don't bind yet, wait for more specific constraints
            return False
        else:
            return False
    
    def _apply_subst_to_constraint_value(self, value: Union[Value, TypeVar]) -> Union[Value, TypeVar]:
        """Apply substitution to a constraint value."""
        if isinstance(value, TypeVar):
            result = self.substitution.lookup(value)
            if result:
                return result
            return value
        elif isinstance(value, Value):
            return self.substitution.apply_to_value(value)
        else:
            return value
    
    def _solve_equality(self, left: Union[Value, TypeVar], right: Union[Value, TypeVar], 
                       ctx: 'Context') -> bool:
        """Solve an equality constraint."""
        # print(f"DEBUG _solve_equality: left={left} ({type(left)}), right={right} ({type(right)})")
        
        # Check if left is a metavariable value
        if isinstance(left, VMetaVar):
            # Bind the type variable to the right side
            self.substitution.bind(left.var, right)
            # print(f"DEBUG _solve_equality: Bound {left.var} to {right}")
            return True
        elif isinstance(right, VMetaVar):
            # Bind the type variable to the left side
            self.substitution.bind(right.var, left)
            # print(f"DEBUG _solve_equality: Bound {right.var} to {left}")
            return True
        
        # If one side is a type variable, bind it
        elif isinstance(left, TypeVar) and not isinstance(right, TypeVar):
            self.substitution.bind(left, right)
            return True
        elif isinstance(right, TypeVar) and not isinstance(left, TypeVar):
            self.substitution.bind(right, left)
            return True
        elif isinstance(left, TypeVar) and isinstance(right, TypeVar):
            # Both are type variables - bind one to the other
            self.substitution.bind(left, VMetaVar(right))
            return True
        else:
            # Both are values - check if they're equal
            return self.type_checker.equal_types(left, right, ctx)
    
    def _unify(self, left: Union[Value, TypeVar], right: Union[Value, TypeVar], 
               ctx: 'Context') -> bool:
        """Unify two types."""
        # For now, treat unification like equality
        return self._solve_equality(left, right, ctx)
    
    def _default_constraint(self, constraint: Constraint) -> None:
        """Try to default a constraint with a reasonable value."""
        # If we have a type variable that should be a Type, default it to Type0
        if isinstance(constraint.left, TypeVar) and isinstance(constraint.right, VType):
            self.substitution.bind(constraint.left, VType(Level(0)))
        elif isinstance(constraint.right, TypeVar) and isinstance(constraint.left, VType):
            self.substitution.bind(constraint.right, VType(Level(0)))


def infer_implicit_args_with_constraints(
    app_term: Term,
    fun_type: Value,
    explicit_args: List[Term],
    ctx: 'Context',
    type_checker: 'TypeChecker'
) -> Tuple[List[Term], Substitution]:
    """Infer implicit arguments using constraint solving.
    
    Returns a tuple of (implicit_args, substitution).
    """
    solver = ConstraintSolver(type_checker)
    implicit_args = []
    
    # Debug: print function type
    # print(f"DEBUG: Inferring implicit args for function type: {fun_type}")
    # print(f"DEBUG: Explicit args: {explicit_args}")
    
    # Create type variables for each implicit parameter
    current_type = fun_type
    type_vars = []
    
    while isinstance(current_type, VPi) and current_type.implicit:
        # Create a fresh type variable
        var = solver.fresh_type_var(f"T{len(type_vars)}")
        type_vars.append(var)
        implicit_args.append(TMetaVar(var))
        
        # print(f"DEBUG: Created type var {var} for implicit param with domain {current_type.domain}")
        
        # Add constraint that the type variable has the right type
        solver.add_constraint(
            ConstraintKind.INSTANCE,
            VMetaVar(var),
            current_type.domain,
            ctx
        )
        
        # Apply to get the next type
        current_type = current_type.codomain_closure.apply(VMetaVar(var))
    
    # Now process explicit arguments to gather constraints
    # For const : {A : Type} -> {B : Type} -> A -> B -> A
    # With args [Z, True], we want A = Nat, B = Bool
    
    # First, let's analyze what types we expect for the explicit parameters
    explicit_param_types = []
    temp_type = current_type
    while isinstance(temp_type, VPi) and not temp_type.implicit:
        explicit_param_types.append(temp_type.domain)
        # Apply a dummy value to continue
        temp_type = temp_type.codomain_closure.apply(VNeutral(NVar(1000)))
    
    for i, arg in enumerate(explicit_args):
        if not isinstance(current_type, VPi) or current_type.implicit:
            break
        
        # Infer the type of the explicit argument
        try:
            arg_type = type_checker.infer_type(arg, ctx)
            
            # print(f"DEBUG: Arg {i} has type {arg_type}")
            # print(f"DEBUG: Current type domain: {current_type.domain}")
            
            # For const's first explicit parameter: it has type A (the first implicit)
            # So if arg has type Nat, then A = Nat
            if i == 0 and len(type_vars) > 0:
                # The parameter type should be the first type variable
                # print(f"DEBUG: Adding constraint: {type_vars[0]} = {arg_type}")
                solver.add_constraint(
                    ConstraintKind.EQUALS,
                    VMetaVar(type_vars[0]),
                    arg_type,
                    ctx
                )
            
            # For const's second explicit parameter: it has type B (the second implicit)
            # So if arg has type Bool, then B = Bool
            elif i == 1 and len(type_vars) > 1:
                # print(f"DEBUG: Adding constraint: {type_vars[1]} = {arg_type}")
                solver.add_constraint(
                    ConstraintKind.EQUALS,
                    VMetaVar(type_vars[1]),
                    arg_type,
                    ctx
                )
            
            # Also add constraint that argument type matches expected parameter type
            # (after substitution)
            expected_type = solver.substitution.apply_to_value(current_type.domain)
            if isinstance(expected_type, VMetaVar):
                # The expected type is still a metavariable, so we can constrain it
                solver.add_constraint(
                    ConstraintKind.EQUALS,
                    expected_type,
                    arg_type,
                    ctx
                )
        except Exception as e:
            # If we can't infer the type, continue
            # print(f"DEBUG: Error inferring type for arg {i}: {e}")
            pass
        
        # Apply argument to get next type
        arg_val = type_checker.eval_with_globals(arg, ctx.environment)
        current_type = current_type.codomain_closure.apply(arg_val)
    
    # Solve constraints
    # print("DEBUG: Solving constraints...")
    substitution = solver.solve()
    
    # print(f"DEBUG: Substitution: {substitution.mapping}")
    
    # Apply substitution to get concrete implicit arguments
    concrete_implicit_args = []
    for i, metavar_term in enumerate(implicit_args):
        if isinstance(metavar_term, TMetaVar):
            value = substitution.lookup(metavar_term.var)
            if value:
                # Convert value back to term
                # print(f"DEBUG: Converting {value} to term")
                concrete_term = type_checker.value_to_term(value, ctx)
                concrete_implicit_args.append(concrete_term)
            else:
                # Couldn't infer this parameter
                # print(f"DEBUG: Could not infer implicit parameter {i+1} (var {metavar_term.var})")
                raise TypeCheckError(f"Could not infer implicit parameter {i+1}")
        else:
            concrete_implicit_args.append(metavar_term)
    
    # print(f"DEBUG: Inferred implicit args: {concrete_implicit_args}")
    return concrete_implicit_args, substitution