"""Core type system definitions for ai-lang.

This module defines the core values and types used during type checking
and evaluation. These are separate from the AST to allow for normalization
and computation during type checking.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Any, Tuple, Callable
from abc import ABC, abstractmethod
from enum import Enum, auto


class Level:
    """Universe levels for predicativity."""
    def __init__(self, n: int = 0):
        self.n = n
    
    def __repr__(self) -> str:
        return f"Level({self.n})"
    
    def __eq__(self, other: object) -> bool:
        return isinstance(other, Level) and self.n == other.n
    
    def __lt__(self, other: Level) -> bool:
        return self.n < other.n
    
    def succ(self) -> Level:
        """Successor level."""
        return Level(self.n + 1)
    
    def max(self, other: Level) -> Level:
        """Maximum of two levels."""
        return Level(max(self.n, other.n))


class Value(ABC):
    """Base class for values during evaluation."""
    
    @abstractmethod
    def quote(self, level: int = 0) -> 'Term':
        """Quote a value back to a term."""
        pass


class Term(ABC):
    """Base class for terms (normalized expressions)."""
    
    @abstractmethod
    def eval(self, env: 'Environment') -> Value:
        """Evaluate a term to a value."""
        pass


@dataclass(frozen=True)
class VType(Value):
    """Type universe value."""
    level: Level
    
    def quote(self, level: int = 0) -> Term:
        return TType(self.level)


@dataclass(frozen=True)
class VNat(Value):
    """Natural number value."""
    value: int
    
    def quote(self, level: int = 0) -> Term:
        # Build up the term representation
        if self.value == 0:
            return TConstructor("Z", [])
        else:
            result = TConstructor("Z", [])
            for _ in range(self.value):
                result = TConstructor("S", [result])
            return result


@dataclass(frozen=True)
class VBool(Value):
    """Boolean value."""
    value: bool
    
    def quote(self, level: int = 0) -> Term:
        return TConstructor("True" if self.value else "False", [])


@dataclass(frozen=True)
class VString(Value):
    """String value."""
    value: str
    
    def quote(self, level: int = 0) -> Term:
        return TLiteral(self.value)


@dataclass(frozen=True)
class VPi(Value):
    """Pi type (dependent function type) value."""
    name: str
    domain: Value
    codomain_closure: 'Closure'
    implicit: bool = False
    
    def quote(self, level: int = 0) -> Term:
        domain_term = self.domain.quote(level)
        # Create a fresh variable for the codomain
        var = VNeutral(NVar(level))
        codomain_val = self.codomain_closure.apply(var)
        codomain_term = codomain_val.quote(level + 1)
        return TPi(self.name, domain_term, codomain_term, self.implicit)


@dataclass(frozen=True)
class VLambda(Value):
    """Lambda (function) value."""
    name: str
    closure: 'Closure'
    implicit: bool = False
    
    def quote(self, level: int = 0) -> Term:
        # Create a fresh variable
        var = VNeutral(NVar(level))
        body_val = self.closure.apply(var)
        body_term = body_val.quote(level + 1)
        return TLambda(self.name, body_term, self.implicit)


@dataclass(frozen=True)
class VConstructor(Value):
    """Data constructor value."""
    name: str
    args: List[Value]
    
    def quote(self, level: int = 0) -> Term:
        return TConstructor(self.name, [arg.quote(level) for arg in self.args])


@dataclass(frozen=True)
class VNeutral(Value):
    """Neutral value (blocked computation)."""
    neutral: 'Neutral'
    
    def quote(self, level: int = 0) -> Term:
        return self.neutral.quote(level)


@dataclass(frozen=True)
class VConstraint(Value):
    """Type class constraint value (e.g., Eq A)."""
    class_name: str
    type_arg: Value
    
    def quote(self, level: int = 0) -> Term:
        return TConstraint(self.class_name, self.type_arg.quote(level))


@dataclass(frozen=True)
class VConstraintPi(Value):
    """Constrained function type value (e.g., Eq A => A -> A -> Bool)."""
    constraints: List[VConstraint]
    body: Value
    
    def quote(self, level: int = 0) -> Term:
        constraint_terms = [(c.class_name, c.type_arg.quote(level)) for c in self.constraints]
        return TConstraintPi(constraint_terms, self.body.quote(level))


@dataclass(frozen=True)
class VInstance(Value):
    """Type class instance value."""
    class_name: str
    type_arg: Value
    methods: Dict[str, Value]  # method name -> implementation
    
    def quote(self, level: int = 0) -> Term:
        method_terms = {name: impl.quote(level) for name, impl in self.methods.items()}
        return TInstance(self.class_name, self.type_arg.quote(level), method_terms)


@dataclass(frozen=True)
class VTypeClassMethod(Value):
    """Type class method that needs instance resolution."""
    class_name: str
    method_name: str
    
    def quote(self, level: int = 0) -> Term:
        # Quote as a global reference
        return TGlobal(self.method_name)


@dataclass(frozen=True)
class VIO(Value):
    """IO type constructor value."""
    result_type: Value
    
    def quote(self, level: int = 0) -> Term:
        return TApp(TGlobal("IO"), self.result_type.quote(level))


@dataclass(frozen=True)
class VUnit(Value):
    """Unit type value."""
    
    def quote(self, level: int = 0) -> Term:
        return TGlobal("Unit")


@dataclass(frozen=True) 
class VUnitValue(Value):
    """Unit value (the single inhabitant of Unit type)."""
    
    def quote(self, level: int = 0) -> Term:
        return TConstructor("unit", [])


@dataclass(frozen=True)
class VIOAction(Value):
    """IO action value (runtime representation)."""
    action: 'IOAction'
    
    def quote(self, level: int = 0) -> Term:
        # IO actions cannot be quoted back to terms
        raise ValueError("Cannot quote IO action")


# Neutral terms (cannot be reduced further)
class Neutral(ABC):
    """Base class for neutral terms."""
    
    @abstractmethod
    def quote(self, level: int = 0) -> Term:
        pass


@dataclass(frozen=True)
class NVar(Neutral):
    """Neutral variable (de Bruijn level)."""
    level: int
    
    def quote(self, level: int = 0) -> Term:
        # Convert de Bruijn level to de Bruijn index
        return TVar(level - self.level - 1)


@dataclass(frozen=True)
class NApp(Neutral):
    """Neutral application."""
    function: Neutral
    argument: Value
    implicit: bool = False
    
    def quote(self, level: int = 0) -> Term:
        return TApp(self.function.quote(level), self.argument.quote(level), self.implicit)


@dataclass(frozen=True)
class NGlobal(Neutral):
    """Neutral global reference."""
    name: str
    
    def quote(self, level: int = 0) -> Term:
        return TGlobal(self.name)


# Terms (syntax after normalization)
@dataclass(frozen=True)
class TType(Term):
    """Type universe term."""
    level: Level
    
    def eval(self, env: Environment) -> Value:
        return VType(self.level)


@dataclass(frozen=True)
class TVar(Term):
    """Variable (de Bruijn index)."""
    index: int
    
    def eval(self, env: Environment) -> Value:
        return env.lookup(self.index)


@dataclass(frozen=True)
class TPi(Term):
    """Pi type term."""
    name: str
    domain: Term
    codomain: Term
    implicit: bool = False
    
    def eval(self, env: Environment) -> Value:
        domain_val = self.domain.eval(env)
        codomain_closure = Closure(env, self.codomain)
        return VPi(self.name, domain_val, codomain_closure, self.implicit)


@dataclass(frozen=True)
class TLambda(Term):
    """Lambda term."""
    name: str
    body: Term
    implicit: bool = False
    
    def eval(self, env: Environment) -> Value:
        closure = Closure(env, self.body)
        return VLambda(self.name, closure, self.implicit)


@dataclass(frozen=True)
class TApp(Term):
    """Application term."""
    function: Term
    argument: Term
    implicit: bool = False
    
    def eval(self, env: Environment) -> Value:
        fun_val = self.function.eval(env)
        arg_val = self.argument.eval(env)
        return apply(fun_val, arg_val)


@dataclass(frozen=True)
class TLiteral(Term):
    """Literal value term."""
    value: Union[int, bool, str]
    
    def eval(self, env: Environment) -> Value:
        if isinstance(self.value, int):
            return VNat(self.value)
        elif isinstance(self.value, bool):
            return VBool(self.value)
        elif isinstance(self.value, str):
            return VString(self.value)
        else:
            raise TypeError(f"Unknown literal type: {type(self.value)}")


@dataclass(frozen=True)
class TConstructor(Term):
    """Constructor term."""
    name: str
    args: List[Term]
    
    def eval(self, env: Environment) -> Value:
        arg_vals = [arg.eval(env) for arg in self.args]
        return VConstructor(self.name, arg_vals)


@dataclass(frozen=True)
class TLet(Term):
    """Let binding term."""
    name: str
    value: Term
    body: Term
    
    def eval(self, env: Environment) -> Value:
        val = self.value.eval(env)
        new_env = env.extend(val)
        return self.body.eval(new_env)


@dataclass(frozen=True)
class TGlobal(Term):
    """Global reference term."""
    name: str
    
    def eval(self, env: Environment) -> Value:
        # During type checking, return a neutral value
        # The actual evaluation will be handled by the evaluator
        return VNeutral(NGlobal(self.name))


@dataclass(frozen=True)
class TIO(Term):
    """IO type constructor term."""
    result_type: Term
    
    def eval(self, env: Environment) -> Value:
        result_type_val = self.result_type.eval(env)
        return VIO(result_type_val)


@dataclass(frozen=True)
class TConstraint(Term):
    """Type class constraint term."""
    class_name: str
    type_arg: Term
    
    def eval(self, env: Environment) -> Value:
        type_arg_val = self.type_arg.eval(env)
        return VConstraint(self.class_name, type_arg_val)


@dataclass(frozen=True)
class TConstraintPi(Term):
    """Constrained function type term."""
    constraints: List[Tuple[str, Term]]  # [(class_name, type_arg)]
    body: Term
    
    def eval(self, env: Environment) -> Value:
        constraint_vals = [VConstraint(cn, ta.eval(env)) for cn, ta in self.constraints]
        body_val = self.body.eval(env)
        return VConstraintPi(constraint_vals, body_val)


@dataclass(frozen=True)
class TInstance(Term):
    """Type class instance term."""
    class_name: str
    type_arg: Term
    methods: Dict[str, Term]  # method name -> implementation
    
    def eval(self, env: Environment) -> Value:
        type_arg_val = self.type_arg.eval(env)
        method_vals = {name: impl.eval(env) for name, impl in self.methods.items()}
        return VInstance(self.class_name, type_arg_val, method_vals)


# Environment and closures
@dataclass
class Environment:
    """Environment mapping de Bruijn indices to values."""
    values: List[Value]
    
    def __init__(self, values: List[Value] = None):
        self.values = values or []
    
    def lookup(self, index: int) -> Value:
        """Look up a variable by de Bruijn index."""
        if 0 <= index < len(self.values):
            return self.values[-(index + 1)]
        raise IndexError(f"Variable index {index} out of range")
    
    def extend(self, value: Value) -> 'Environment':
        """Extend environment with a new value."""
        return Environment(self.values + [value])
    
    def empty(self) -> 'Environment':
        """Create an empty environment."""
        return Environment([])


@dataclass
class Closure:
    """Closure capturing an environment and a term."""
    env: Environment
    term: Term
    
    def apply(self, value: Value) -> Value:
        """Apply closure to a value."""
        new_env = self.env.extend(value)
        return self.term.eval(new_env)


@dataclass
class ConstantClosure:
    """Closure that always returns the same value."""
    value: Value
    
    def apply(self, value: Value) -> Value:
        """Apply closure - ignores input and returns constant."""
        return self.value


def apply(function: Value, argument: Value) -> Value:
    """Apply a function value to an argument value."""
    if isinstance(function, VLambda):
        return function.closure.apply(argument)
    elif isinstance(function, VNeutral):
        return VNeutral(NApp(function.neutral, argument))
    elif isinstance(function, VConstructor):
        # Partial constructor application
        return VConstructor(function.name, function.args + [argument])
    elif isinstance(function, VTypeClassMethod):
        # Type class method - we need the evaluator to resolve this
        # For now, return a neutral application
        return VNeutral(NApp(NGlobal(function.method_name), argument))
    elif isinstance(function, VIOAction):
        # IO actions cannot be directly applied - they need special handling
        raise TypeError("Cannot directly apply IO action - use bind instead")
    else:
        raise TypeError(f"Cannot apply {type(function)} to argument")


def normalize(env: Environment, term: Term) -> Term:
    """Normalize a term by evaluation and quotation."""
    value = term.eval(env)
    return value.quote(len(env.values))


# Built-in type constructors
BUILTIN_TYPES = {
    "Type": lambda level: VType(level),
    "Nat": lambda: VType(Level(0)),  # Nat : Type
    "Bool": lambda: VType(Level(0)), # Bool : Type  
    "String": lambda: VType(Level(0)), # String : Type
    "Unit": lambda: VUnit(), # Unit : Type
}


# IO Action classes for runtime execution
class IOAction(ABC):
    """Abstract base class for IO actions."""
    
    @abstractmethod
    def execute(self, input_handler: Optional[Callable[[], str]] = None) -> Tuple[Value, List['IOEffect']]:
        """Execute the IO action, returning a value and list of effects."""
        pass


@dataclass
class IOPure(IOAction):
    """Pure value wrapped in IO."""
    value: Value
    
    def execute(self, input_handler: Optional[Callable[[], str]] = None) -> Tuple[Value, List['IOEffect']]:
        return (self.value, [])


@dataclass
class IOBind(IOAction):
    """Monadic bind operation."""
    first: IOAction
    cont: Callable[[Value], IOAction]
    
    def execute(self, input_handler: Optional[Callable[[], str]] = None) -> Tuple[Value, List['IOEffect']]:
        val1, effects1 = self.first.execute(input_handler)
        action2 = self.cont(val1)
        val2, effects2 = action2.execute(input_handler)
        return (val2, effects1 + effects2)


@dataclass
class IOPrint(IOAction):
    """Print a string to stdout."""
    message: str
    
    def execute(self, input_handler: Optional[Callable[[], str]] = None) -> Tuple[Value, List['IOEffect']]:
        return (VConstructor("unit", []), [PrintEffect(self.message)])


@dataclass  
class IOGetLine(IOAction):
    """Read a line from stdin."""
    
    def execute(self, input_handler: Optional[Callable[[], str]] = None) -> Tuple[Value, List['IOEffect']]:
        if input_handler:
            line = input_handler()
        else:
            line = input()
        return (VString(line), [ReadEffect(line)])


# IO Effects for tracking
@dataclass
class IOEffect:
    """Base class for IO effects."""
    pass


@dataclass
class PrintEffect(IOEffect):
    """Effect of printing to stdout."""
    message: str


@dataclass
class ReadEffect(IOEffect):
    """Effect of reading from stdin."""
    result: str