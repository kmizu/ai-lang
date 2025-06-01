"""Abstract Syntax Tree definitions for ai-lang."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple
from abc import ABC, abstractmethod


@dataclass(frozen=True)
class SourceLocation:
    """Source code location information."""
    line: int
    column: int
    filename: Optional[str] = None


class ASTNode(ABC):
    """Base class for all AST nodes."""
    
    @abstractmethod
    def __repr__(self) -> str:
        pass


@dataclass(frozen=True)
class Name(ASTNode):
    """Variable or constructor name."""
    value: str
    location: Optional[SourceLocation] = None
    
    def __repr__(self) -> str:
        return f"Name({self.value})"


# Type expressions
class Type(ASTNode):
    """Base class for type expressions."""
    pass


@dataclass(frozen=True)
class TypeVar(Type):
    """Type variable."""
    name: Name
    
    def __repr__(self) -> str:
        return f"TypeVar({self.name.value})"


@dataclass(frozen=True)
class TupleType(Type):
    """Tuple type (A, B, C)."""
    elements: List[Type]
    
    def __repr__(self) -> str:
        elements_str = ", ".join(str(e) for e in self.elements)
        return f"({elements_str})"


@dataclass(frozen=True)
class TypeApp(Type):
    """Type application."""
    constructor: Type
    argument: Type
    
    def __repr__(self) -> str:
        return f"TypeApp({self.constructor}, {self.argument})"


@dataclass(frozen=True)
class TypeConstructor(Type):
    """Built-in or user-defined type constructor."""
    name: Name
    
    def __repr__(self) -> str:
        return f"TypeConstructor({self.name.value})"


@dataclass(frozen=True)
class FunctionType(Type):
    """Function type (arrow type)."""
    param_name: Optional[Name]
    param_type: Type
    return_type: Type
    implicit: bool = False
    
    def __repr__(self) -> str:
        implicit_str = "implicit " if self.implicit else ""
        if self.param_name:
            return f"FunctionType({implicit_str}{self.param_name.value} : {self.param_type} -> {self.return_type})"
        return f"FunctionType({implicit_str}{self.param_type} -> {self.return_type})"


@dataclass(frozen=True)
class UniverseType(Type):
    """Type universe (Type)."""
    level: int = 0
    
    def __repr__(self) -> str:
        return f"Type{self.level}" if self.level > 0 else "Type"


@dataclass(frozen=True)
class ConstraintType(Type):
    """Type with constraints (e.g., Eq A => A -> A -> Bool)."""
    constraints: List[Tuple[Name, Type]]  # [(ClassName, Type)]
    body: Type
    
    def __repr__(self) -> str:
        constraints_str = ", ".join(f"{c.value} {t}" for c, t in self.constraints)
        return f"ConstraintType({constraints_str} => {self.body})"


# Expressions
class Expr(ASTNode):
    """Base class for expressions."""
    pass


@dataclass(frozen=True)
class Var(Expr):
    """Variable reference."""
    name: Name
    qualifier: Optional[Name] = None  # For qualified names like Math.add
    
    def __repr__(self) -> str:
        if self.qualifier:
            return f"Var({self.qualifier.value}.{self.name.value})"
        return f"Var({self.name.value})"


@dataclass(frozen=True)
class Literal(Expr):
    """Literal value."""
    value: Union[int, bool, str]
    
    def __repr__(self) -> str:
        return f"Literal({self.value!r})"


@dataclass(frozen=True)
class Lambda(Expr):
    """Lambda abstraction."""
    param: Name
    param_type: Optional[Type]
    body: Expr
    implicit: bool = False
    
    def __repr__(self) -> str:
        implicit_str = "{" if self.implicit else "("
        implicit_end = "}" if self.implicit else ")"
        type_str = f" : {self.param_type}" if self.param_type else ""
        return f"Lambda({implicit_str}{self.param.value}{type_str}{implicit_end} -> {self.body})"


@dataclass(frozen=True)
class App(Expr):
    """Function application."""
    function: Expr
    argument: Expr
    implicit: bool = False
    
    def __repr__(self) -> str:
        implicit_str = "implicit " if self.implicit else ""
        return f"App({implicit_str}{self.function} {self.argument})"


@dataclass(frozen=True)
class Let(Expr):
    """Let binding."""
    name: Name
    type_annotation: Optional[Type]
    value: Expr
    body: Expr
    
    def __repr__(self) -> str:
        type_str = f" : {self.type_annotation}" if self.type_annotation else ""
        return f"Let({self.name.value}{type_str} = {self.value} in {self.body})"


@dataclass(frozen=True)
class Case(Expr):
    """Pattern matching expression."""
    scrutinee: Expr
    branches: List[CaseBranch]
    
    def __repr__(self) -> str:
        branches_str = ", ".join(str(b) for b in self.branches)
        return f"Case({self.scrutinee} of [{branches_str}])"


@dataclass(frozen=True)
class CaseBranch:
    """A branch in a case expression."""
    pattern: Pattern
    body: Expr
    
    def __repr__(self) -> str:
        return f"{self.pattern} -> {self.body}"


@dataclass(frozen=True)
class ListLiteral(Expr):
    """List literal expression [1, 2, 3]."""
    elements: List[Expr]
    
    def __repr__(self) -> str:
        elements_str = ", ".join(str(e) for e in self.elements)
        return f"[{elements_str}]"


@dataclass(frozen=True)
class RecordLiteral(Expr):
    """Record literal expression {field = value, ...}."""
    fields: List[Tuple[Name, Expr]]
    
    def __repr__(self) -> str:
        fields_str = ", ".join(f"{n.value} = {e}" for n, e in self.fields)
        return f"{{ {fields_str} }}"


@dataclass(frozen=True)
class FieldAccess(Expr):
    """Field access expression record.field."""
    record: Expr
    field: Name
    
    def __repr__(self) -> str:
        return f"{self.record}.{self.field.value}"


@dataclass(frozen=True)
class RecordUpdate(Expr):
    """Record update expression record { field = value, ... }."""
    record: Expr
    updates: List[Tuple[Name, Expr]]
    
    def __repr__(self) -> str:
        updates_str = ", ".join(f"{n.value} = {e}" for n, e in self.updates)
        return f"{self.record} {{ {updates_str} }}"


@dataclass(frozen=True)
class TupleLiteral(Expr):
    """Tuple literal expression (a, b, c)."""
    elements: List[Expr]
    
    def __repr__(self) -> str:
        elements_str = ", ".join(str(e) for e in self.elements)
        return f"({elements_str})"


# Patterns
class Pattern(ASTNode):
    """Base class for patterns."""
    pass


@dataclass(frozen=True)
class PatternVar(Pattern):
    """Variable pattern."""
    name: Name
    
    def __repr__(self) -> str:
        return f"PatternVar({self.name.value})"


@dataclass(frozen=True)
class PatternConstructor(Pattern):
    """Constructor pattern."""
    constructor: Name
    args: List[Pattern]
    
    def __repr__(self) -> str:
        args_str = " ".join(str(arg) for arg in self.args)
        return f"PatternConstructor({self.constructor.value} {args_str})"


@dataclass(frozen=True)
class PatternLiteral(Pattern):
    """Literal pattern."""
    value: Union[int, bool, str]
    
    def __repr__(self) -> str:
        return f"PatternLiteral({self.value!r})"


@dataclass(frozen=True)
class PatternWildcard(Pattern):
    """Wildcard pattern (_)."""
    
    def __repr__(self) -> str:
        return "PatternWildcard"


@dataclass(frozen=True)
class PatternRecord(Pattern):
    """Record pattern { field = pattern, ... }."""
    fields: List[Tuple[Name, Pattern]]
    
    def __repr__(self) -> str:
        fields_str = ", ".join(f"{n.value} = {p}" for n, p in self.fields)
        return f"PatternRecord({ {fields_str} })"


# Declarations
class Declaration(ASTNode):
    """Base class for declarations."""
    pass


@dataclass(frozen=True)
class TypeSignature(Declaration):
    """Type signature declaration."""
    name: Name
    type: Type
    
    def __repr__(self) -> str:
        return f"TypeSignature({self.name.value} : {self.type})"


@dataclass(frozen=True)
class FunctionDef(Declaration):
    """Function definition with pattern matching."""
    name: Name
    clauses: List[FunctionClause]
    
    def __repr__(self) -> str:
        clauses_str = "; ".join(str(c) for c in self.clauses)
        return f"FunctionDef({self.name.value}, [{clauses_str}])"


@dataclass(frozen=True)
class FunctionClause:
    """A clause in a function definition."""
    patterns: List[Pattern]
    body: Expr
    
    def __repr__(self) -> str:
        patterns_str = " ".join(str(p) for p in self.patterns)
        return f"{patterns_str} = {self.body}"


@dataclass(frozen=True)
class DataDecl(Declaration):
    """Inductive data type declaration."""
    name: Name
    type_params: List[Name]
    indices: List[tuple[Name, Type]]  # For indexed types like Vec
    constructors: List[Constructor]
    
    def __repr__(self) -> str:
        params_str = " ".join(p.value for p in self.type_params)
        indices_str = " ".join(f"({n.value} : {t})" for n, t in self.indices)
        ctors_str = "; ".join(str(c) for c in self.constructors)
        return f"DataDecl({self.name.value} {params_str} {indices_str} where [{ctors_str}])"


@dataclass(frozen=True)
class Constructor:
    """Data constructor."""
    name: Name
    type: Type
    
    def __repr__(self) -> str:
        return f"Constructor({self.name.value} : {self.type})"


@dataclass(frozen=True)
class RecordDecl(Declaration):
    """Record type declaration."""
    name: Name
    fields: List[Tuple[Name, Type]]
    
    def __repr__(self) -> str:
        fields_str = ", ".join(f"{n.value} : {t}" for n, t in self.fields)
        return f"RecordDecl({self.name.value} : Type where {{ {fields_str} }})"


@dataclass(frozen=True)
class TypeAlias(Declaration):
    """Type alias declaration."""
    name: Name
    params: List[Name]
    body: Type
    
    def __repr__(self) -> str:
        params_str = " ".join(p.value for p in self.params)
        if params_str:
            return f"TypeAlias(type {self.name.value} {params_str} = {self.body})"
        else:
            return f"TypeAlias(type {self.name.value} = {self.body})"


@dataclass(frozen=True)
class ImportItem:
    """A single imported name."""
    name: Name
    alias: Optional[Name] = None
    
    def __repr__(self) -> str:
        if self.alias:
            return f"{self.name.value} as {self.alias.value}"
        return self.name.value


@dataclass(frozen=True)
class Import(Declaration):
    """Import declaration."""
    module: Name
    alias: Optional[Name] = None
    items: Optional[List[ImportItem]] = None  # None means import all
    
    def __repr__(self) -> str:
        if self.alias:
            return f"Import({self.module.value} as {self.alias.value})"
        elif self.items is not None:
            items_str = ", ".join(str(item) for item in self.items)
            return f"Import({self.module.value} ({items_str}))"
        else:
            return f"Import({self.module.value})"


@dataclass(frozen=True)
class Export(Declaration):
    """Export declaration."""
    names: List[Name]
    
    def __repr__(self) -> str:
        names_str = ", ".join(n.value for n in self.names)
        return f"Export({names_str})"


@dataclass(frozen=True)
class ClassDecl(Declaration):
    """Type class declaration."""
    name: Name
    type_param: Name
    superclasses: List[Tuple[Name, Type]]  # [(ClassName, Type)]
    methods: List[Tuple[Name, Type]]  # [(method_name, method_type)]
    
    def __repr__(self) -> str:
        super_str = ", ".join(f"{c.value} {t}" for c, t in self.superclasses) if self.superclasses else ""
        methods_str = "; ".join(f"{n.value} : {t}" for n, t in self.methods)
        if super_str:
            return f"ClassDecl(class {super_str} => {self.name.value} {self.type_param.value} where [{methods_str}])"
        else:
            return f"ClassDecl(class {self.name.value} {self.type_param.value} where [{methods_str}])"


@dataclass(frozen=True)
class InstanceDecl(Declaration):
    """Type class instance declaration."""
    class_name: Name
    type: Type
    constraints: List[Tuple[Name, Type]]  # [(ClassName, Type)]
    methods: List[Tuple[Name, Expr]]  # [(method_name, implementation)]
    
    def __repr__(self) -> str:
        constraints_str = ", ".join(f"{c.value} {t}" for c, t in self.constraints) if self.constraints else ""
        methods_str = "; ".join(f"{n.value} = {e}" for n, e in self.methods)
        if constraints_str:
            return f"InstanceDecl(instance {constraints_str} => {self.class_name.value} {self.type} where [{methods_str}])"
        else:
            return f"InstanceDecl(instance {self.class_name.value} {self.type} where [{methods_str}])"


@dataclass(frozen=True)
class Module(ASTNode):
    """Module containing declarations."""
    name: Optional[Name]
    imports: List[Import]
    exports: List[Export]
    declarations: List[Declaration]
    
    def __repr__(self) -> str:
        name_str = f"{self.name.value}" if self.name else "<main>"
        decls_str = "\n".join(str(d) for d in self.declarations)
        return f"Module({name_str}):\n{decls_str}"