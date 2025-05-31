"""Type checker for ai-lang using bidirectional type checking.

This implements a bidirectional type checker that can handle dependent types,
implicit arguments, and pattern matching.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union
from enum import Enum, auto

from .syntax import *
from .core import *


class TypeCheckError(Exception):
    """Type checking error."""
    def __init__(self, message: str, location: Optional[SourceLocation] = None):
        super().__init__(message)
        self.location = location


# Context for type checking
@dataclass
class Binding:
    """A variable binding in the context."""
    name: str
    type: Value
    value: Optional[Value] = None  # For let-bound variables


@dataclass
class Context:
    """Type checking context."""
    bindings: List[Binding]
    environment: Environment
    
    def __init__(self):
        self.bindings = []
        self.environment = Environment([])
    
    def extend(self, name: str, type_val: Value, value: Optional[Value] = None) -> 'Context':
        """Extend context with a new binding."""
        new_ctx = Context()
        new_ctx.bindings = self.bindings + [Binding(name, type_val, value)]
        if value is not None:
            new_ctx.environment = self.environment.extend(value)
        else:
            # For bound variables without values, use a neutral variable
            level = len(self.environment.values)
            new_ctx.environment = self.environment.extend(VNeutral(NVar(level)))
        return new_ctx
    
    def lookup(self, name: str) -> Optional[Tuple[int, Value, Optional[Value]]]:
        """Look up a variable by name, returning (index, type, value)."""
        for i, binding in enumerate(reversed(self.bindings)):
            if binding.name == name:
                return (i, binding.type, binding.value)
        return None
    
    def get_type(self, index: int) -> Value:
        """Get type of variable by de Bruijn index."""
        if 0 <= index < len(self.bindings):
            return self.bindings[-(index + 1)].type
        raise IndexError(f"Variable index {index} out of range")


@dataclass
class DataConstructorInfo:
    """Information about a data constructor."""
    name: str
    type: Value
    data_type: str
    parameters: List[str]
    indices: List[Tuple[str, Value]]


@dataclass
class TypeChecker:
    """Type checker state."""
    data_types: Dict[str, DataDecl]
    constructors: Dict[str, DataConstructorInfo]
    global_types: Dict[str, Value]
    global_values: Dict[str, Value]
    
    def __init__(self):
        self.data_types = {}
        self.constructors = {}
        self.global_types = {}
        self.global_values = {}
    
    def add_data_type(self, decl: DataDecl, ctx: Context) -> None:
        """Add a data type declaration."""
        self.data_types[decl.name.value] = decl
        
        # First, add the data type itself to the context
        # For parameterized types, create a function type
        if decl.type_params:
            # For now, just treat parameterized types as Type
            # TODO: Properly handle type parameters
            data_type = VType(Level(0))
        else:
            data_type = VType(Level(0))
        
        self.add_global(decl.name.value, data_type)
        
        # Create context with type parameters
        param_ctx = ctx
        for param in decl.type_params:
            param_ctx = param_ctx.extend(param.value, VType(Level(0)))
        
        # Add constructors
        for ctor in decl.constructors:
            # Convert constructor type
            # For now, ignore type parameters in constructors
            # Just convert the type as-is, treating type parameters as type constructors
            ctor_type_term = self.convert_type_ignoring_params(ctor.type, decl.type_params)
            ctor_type_val = ctor_type_term.eval(ctx.environment)
            
            info = DataConstructorInfo(
                name=ctor.name.value,
                type=ctor_type_val,
                data_type=decl.name.value,
                parameters=[p.value for p in decl.type_params],
                indices=[(n.value, self.convert_type(t, ctx).eval(ctx.environment)) 
                        for n, t in decl.indices]
            )
            self.constructors[ctor.name.value] = info
    
    def add_global(self, name: str, type_val: Value, value: Optional[Value] = None) -> None:
        """Add a global definition."""
        self.global_types[name] = type_val
        if value is not None:
            self.global_values[name] = value
    
    def check_module(self, module: Module) -> None:
        """Type check a complete module."""
        ctx = Context()
        
        for decl in module.declarations:
            self.check_declaration(decl, ctx)
    
    def check_declaration(self, decl: Declaration, ctx: Context) -> None:
        """Type check a declaration."""
        if isinstance(decl, DataDecl):
            self.add_data_type(decl, ctx)
        
        elif isinstance(decl, TypeSignature):
            # Convert and check the type
            type_term = self.convert_type(decl.type, ctx)
            type_type = self.infer_type(type_term, ctx)
            # Check that the type has a valid sort
            self.check_is_type(type_type, ctx)
            
            # Store the type
            type_val = type_term.eval(ctx.environment)
            self.add_global(decl.name.value, type_val)
        
        elif isinstance(decl, FunctionDef):
            # Look up the type signature if it exists
            if decl.name.value in self.global_types:
                expected_type = self.global_types[decl.name.value]
            else:
                # Try to infer type from the first clause
                if not decl.clauses:
                    raise TypeCheckError(f"Empty function definition: {decl.name.value}")
                
                # For now, we require a type signature
                raise TypeCheckError(
                    f"Function {decl.name.value} requires a type signature"
                )
            
            # Type check all clauses
            for clause in decl.clauses:
                self.check_function_clause(
                    decl.name.value, clause, expected_type, ctx
                )
            
            # Store the function value for use during type checking
            # This allows polymorphic functions to be used before evaluation
            if decl.name.value not in self.global_values:
                # Create a dummy value that will be replaced during evaluation
                self.global_values[decl.name.value] = VNeutral(NGlobal(decl.name.value))
    
    def check_function_clause(self, name: str, clause: FunctionClause, 
                            expected_type: Value, ctx: Context) -> None:
        """Type check a function clause against expected type."""
        # Extract parameter types from the function type
        current_type = expected_type
        new_ctx = ctx
        
        # First, handle implicit parameters that don't have patterns
        while isinstance(current_type, VPi) and current_type.implicit:
            # Add implicit parameter to context
            new_ctx = new_ctx.extend(current_type.name, current_type.domain)
            var_val = VNeutral(NVar(len(new_ctx.environment.values) - 1))
            current_type = current_type.codomain_closure.apply(var_val)
        
        for pattern in clause.patterns:
            if not isinstance(current_type, VPi):
                raise TypeCheckError(
                    f"Too many patterns in function {name}"
                )
            
            # Skip implicit parameters (they should have been handled above)
            if current_type.implicit:
                raise TypeCheckError(
                    f"Pattern for implicit parameter in function {name}"
                )
            
            # Check pattern and extend context
            pattern_ctx = self.check_pattern(pattern, current_type.domain, new_ctx)
            new_ctx = pattern_ctx
            
            # Apply the function type to continue
            # For simplicity, use a dummy value
            var_val = VNeutral(NVar(len(new_ctx.environment.values) - 1))
            current_type = current_type.codomain_closure.apply(var_val)
        
        # Check the body against the remaining type
        body_term = self.convert_expr(clause.body, new_ctx)
        self.check_type(body_term, current_type, new_ctx)
    
    def check_pattern(self, pattern: Pattern, expected_type: Value, ctx: Context) -> Context:
        """Type check a pattern and return extended context."""
        if isinstance(pattern, PatternVar):
            # Variable pattern - just bind the variable
            return ctx.extend(pattern.name.value, expected_type)
        
        elif isinstance(pattern, PatternConstructor):
            # Constructor pattern
            ctor_name = pattern.constructor.value
            if ctor_name not in self.constructors:
                raise TypeCheckError(f"Unknown constructor in pattern: {ctor_name}")
            
            ctor_info = self.constructors[ctor_name]
            
            # Check that the constructor produces the expected type
            # For now, simplified check
            # TODO: Properly check constructor return type matches expected_type
            
            # Check constructor arguments
            ctor_type = ctor_info.type
            new_ctx = ctx
            
            for arg_pattern in pattern.args:
                if not isinstance(ctor_type, VPi):
                    raise TypeCheckError(f"Too many arguments to constructor {ctor_name}")
                
                # Check argument pattern
                new_ctx = self.check_pattern(arg_pattern, ctor_type.domain, new_ctx)
                
                # Apply to get remaining type
                # Use dummy value for now
                dummy = VNeutral(NVar(len(new_ctx.environment.values) - 1))
                ctor_type = ctor_type.codomain_closure.apply(dummy)
            
            return new_ctx
        
        elif isinstance(pattern, PatternLiteral):
            # Literal pattern - check type matches
            # For now, just return the context unchanged
            return ctx
        
        elif isinstance(pattern, PatternWildcard):
            # Wildcard - no binding
            return ctx
        
        else:
            raise TypeCheckError(f"Unknown pattern type: {type(pattern)}")
    
    def infer_type(self, term: Term, ctx: Context) -> Value:
        """Infer the type of a term."""
        if isinstance(term, TType):
            # Type : Type(n+1)
            return VType(term.level.succ())
        
        elif isinstance(term, TVar):
            return ctx.get_type(term.index)
        
        elif isinstance(term, TPi):
            # Check domain type
            domain_type = self.infer_type(term.domain, ctx)
            self.check_is_type(domain_type, ctx)
            
            # Check codomain type in extended context
            domain_val = term.domain.eval(ctx.environment)
            new_ctx = ctx.extend(term.name, domain_val)
            codomain_type = self.infer_type(term.codomain, new_ctx)
            self.check_is_type(codomain_type, new_ctx)
            
            # Pi type has the type of the maximum universe level
            if isinstance(domain_type, VType) and isinstance(codomain_type, VType):
                return VType(domain_type.level.max(codomain_type.level))
            else:
                # For now, be more permissive - return Type0
                return VType(Level(0))
        
        elif isinstance(term, TLambda):
            raise TypeCheckError("Cannot infer type of lambda without annotation")
        
        elif isinstance(term, TApp):
            # Infer function type
            fun_type = self.infer_type(term.function, ctx)
            
            # If we have implicit parameters and an explicit application,
            # we need to insert implicit arguments
            if isinstance(fun_type, VPi) and fun_type.implicit and not term.implicit:
                # Build the full application with implicit arguments
                result_term = self.insert_implicit_args(term, fun_type, ctx)
                return self.infer_type(result_term, ctx)
            
            # Check it's a Pi type
            if not isinstance(fun_type, VPi):
                raise TypeCheckError(f"Expected function type, got {fun_type}")
            
            # Check argument type matches
            if term.implicit != fun_type.implicit:
                raise TypeCheckError(f"Implicit argument mismatch")
            
            # Check argument type
            self.check_type(term.argument, fun_type.domain, ctx)
            
            # Return the instantiated codomain type
            arg_val = term.argument.eval(ctx.environment)
            return fun_type.codomain_closure.apply(arg_val)
        
        elif isinstance(term, TLiteral):
            if isinstance(term.value, int):
                return VConstructor("Nat", [])  # 42 : Nat
            elif isinstance(term.value, bool):
                return VConstructor("Bool", [])  # True : Bool
            elif isinstance(term.value, str):
                return VConstructor("String", [])  # "hello" : String
            else:
                raise TypeCheckError(f"Unknown literal type: {type(term.value)}")
        
        elif isinstance(term, TConstructor):
            # Check for built-in types first
            if term.name in ["Nat", "Bool", "String"]:
                return VType(Level(0))  # Built-in types have type Type
            
            # Look up constructor type
            if term.name not in self.constructors:
                # Check if it's a data type name
                if term.name in self.data_types:
                    # Data type constructors have type Type (for now)
                    return VType(Level(0))
                raise TypeCheckError(f"Unknown constructor: {term.name}")
            
            ctor_info = self.constructors[term.name]
            ctor_type = ctor_info.type
            
            # Apply constructor to arguments
            for arg in term.args:
                if not isinstance(ctor_type, VPi):
                    raise TypeCheckError(f"Too many arguments to constructor {term.name}")
                
                self.check_type(arg, ctor_type.domain, ctx)
                arg_val = arg.eval(ctx.environment)
                ctor_type = ctor_type.codomain_closure.apply(arg_val)
            
            return ctor_type
        
        elif isinstance(term, TGlobal):
            # Look up global type
            if term.name in self.global_types:
                return self.global_types[term.name]
            else:
                raise TypeCheckError(f"Unknown global: {term.name}")
        
        else:
            raise TypeCheckError(f"Cannot infer type of {type(term)}")
    
    def check_type(self, term: Term, expected_type: Value, ctx: Context) -> None:
        """Check that a term has the expected type."""
        if isinstance(term, TLambda):
            # For lambdas, we can check against a Pi type
            if not isinstance(expected_type, VPi):
                raise TypeCheckError(f"Expected function type, got {expected_type}")
            
            # Check that parameter types match if lambda has annotation
            # For now, assume they match
            
            # Check body in extended context
            new_ctx = ctx.extend(term.name, expected_type.domain)
            # Instantiate the codomain with a variable
            var_val = VNeutral(NVar(len(new_ctx.environment.values) - 1))
            body_type = expected_type.codomain_closure.apply(var_val)
            self.check_type(term.body, body_type, new_ctx)
        
        elif isinstance(term, TApp) and not term.implicit:
            # Special handling for explicit applications that may need implicit arguments
            fun_type = self.infer_type(term.function, ctx)
            
            # If function has implicit parameters, insert them
            if isinstance(fun_type, VPi) and fun_type.implicit:
                # Use the same logic as infer_type
                result_term = self.insert_implicit_args(term, fun_type, ctx)
                self.check_type(result_term, expected_type, ctx)
            else:
                # Normal application - infer and check
                inferred_type = self.infer_type(term, ctx)
                if not self.equal_types(inferred_type, expected_type, ctx):
                    raise TypeCheckError(
                        f"Type mismatch: expected {expected_type}, got {inferred_type}"
                    )
        
        else:
            # For other terms, infer and check equality
            inferred_type = self.infer_type(term, ctx)
            if not self.equal_types(inferred_type, expected_type, ctx):
                raise TypeCheckError(
                    f"Type mismatch: expected {expected_type}, got {inferred_type}"
                )
    
    def check_is_type(self, type_val: Value, ctx: Context) -> None:
        """Check that a value is a valid type (has type Type)."""
        # A value is a valid type if it has type Type_n for some n
        if isinstance(type_val, VType):
            return  # Type_n is a valid type
        
        # Other values can also be types if they have type Type
        # For now, we'll be more permissive
        # TODO: Properly check that type_val : Type_n for some n
        pass
    
    def infer_implicit_arg(self, expected_type: Value, ctx: Context) -> Optional[Value]:
        """Try to infer an implicit argument of the given type."""
        # For now, we only handle the case where the expected type is Type
        if isinstance(expected_type, VType):
            # We can't infer which type to use without more context
            # In a full implementation, this would use unification with the expected result type
            return None
        return None
    
    def insert_implicit_args(self, app: TApp, fun_type: Value, ctx: Context) -> Term:
        """Insert implicit arguments for a function application."""
        current_app = app
        current_type = fun_type
        
        # Build applications from the inside out
        while isinstance(current_type, VPi) and current_type.implicit:
            # We need to infer the implicit argument
            if isinstance(current_type.domain, VType):
                # The implicit argument is a Type
                # We can try to infer it from the explicit argument
                try:
                    # Get the type of the explicit argument
                    arg_type = self.infer_type(app.argument, ctx)
                    implicit_arg = self.value_to_term(arg_type, ctx)
                except Exception as e:
                    # If we can't infer, create a metavariable (for now, just fail)
                    raise TypeCheckError(f"Cannot infer implicit type argument: {e}")
            else:
                # For non-type implicit arguments, we need more context
                raise TypeCheckError(f"Cannot infer implicit argument of type {current_type.domain}")
            
            # Create new application with implicit argument
            current_app = TApp(
                TApp(current_app.function, implicit_arg, implicit=True),
                current_app.argument,
                implicit=False
            )
            
            
            # Update the function type
            arg_val = implicit_arg.eval(ctx.environment)
            current_type = current_type.codomain_closure.apply(arg_val)
        
        return current_app
    
    def infer_implicit_from_expected(self, fun_type: VPi, expected_type: Value, 
                                    app_term: TApp, ctx: Context) -> Optional[Term]:
        """Try to infer an implicit argument from the expected result type."""
        # This is a simplified approach
        # In a full implementation, we would use unification
        
        # For the special case of id : {A : Type} -> A -> A
        # If we're applying it to something and expecting a specific type,
        # we can use that type as the implicit argument
        if isinstance(fun_type.domain, VType) and isinstance(expected_type, VType):
            # The implicit argument should be a type
            # Use the expected type directly
            if expected_type.level.n == 0:
                # It's Type, so we can use the constructor for the expected type
                # This is a hack - we should properly track what type we expect
                # For now, infer from the actual argument
                arg_type = self.infer_type(app_term.argument, ctx)
                return self.value_to_term(arg_type, ctx)
        
        return None
    
    def value_to_term(self, val: Value, ctx: Context) -> Term:
        """Convert a value back to a term."""
        # This is a simplified quotation
        if isinstance(val, VType):
            return TType(val.level)
        elif isinstance(val, VConstructor):
            arg_terms = [self.value_to_term(arg, ctx) for arg in val.args]
            return TConstructor(val.name, arg_terms)
        elif isinstance(val, VNeutral):
            return val.neutral.quote(len(ctx.environment.values))
        elif isinstance(val, VPi):
            # Quote a Pi type
            domain_term = self.value_to_term(val.domain, ctx)
            # Create a fresh variable for the codomain
            var_val = VNeutral(NVar(len(ctx.environment.values)))
            new_ctx = ctx.extend(val.name, val.domain, var_val)
            codomain_val = val.codomain_closure.apply(var_val)
            codomain_term = self.value_to_term(codomain_val, new_ctx)
            return TPi(val.name, domain_term, codomain_term, val.implicit)
        else:
            # For other values, we need full quotation
            # This is incomplete
            raise TypeCheckError(f"Cannot quote value {val}")
    
    def equal_types(self, type1: Value, type2: Value, ctx: Context) -> bool:
        """Check if two types are equal."""
        # For now, use a simple structural equality
        # TODO: Implement proper definitional equality
        
        if type(type1) != type(type2):
            return False
        
        if isinstance(type1, VType):
            return type1.level == type2.level
        
        elif isinstance(type1, VPi):
            # Check domains are equal
            if not self.equal_types(type1.domain, type2.domain, ctx):
                return False
            
            # Check codomains are equal (with a fresh variable)
            var_val = VNeutral(NVar(len(ctx.environment.values)))
            cod1 = type1.codomain_closure.apply(var_val)
            cod2 = type2.codomain_closure.apply(var_val)
            
            new_ctx = ctx.extend("_", type1.domain)
            return self.equal_types(cod1, cod2, new_ctx)
        
        elif isinstance(type1, VNeutral) and isinstance(type2, VNeutral):
            # For neutral values, we need to check if they represent the same variable
            # This is a simplified check - proper normalization would be better
            if isinstance(type1.neutral, NVar) and isinstance(type2.neutral, NVar):
                # Two neutral variables are equal if they have the same de Bruijn level
                # relative to the current context size
                ctx_size = len(ctx.environment.values)
                # Convert absolute levels to de Bruijn indices
                index1 = ctx_size - 1 - type1.neutral.level
                index2 = ctx_size - 1 - type2.neutral.level
                return index1 == index2
            # For other neutrals, use structural equality
            return str(type1) == str(type2)
        
        # For other types, use structural equality
        # TODO: Implement proper equality
        return str(type1) == str(type2)
    
    def convert_type_ignoring_params(self, ast_type: Type, type_params: List[Name]) -> Term:
        """Convert a type, treating type parameters as opaque type constructors."""
        if isinstance(ast_type, TypeConstructor):
            name = ast_type.name.value
            
            # Check if it's a type parameter
            if any(p.value == name for p in type_params):
                # Treat as an opaque type constructor
                return TConstructor(name, [])
            
            # Otherwise, use normal conversion
            return self.convert_type(ast_type, Context())
        
        elif isinstance(ast_type, TypeApp):
            # Type application
            ctor = self.convert_type_ignoring_params(ast_type.constructor, type_params)
            arg = self.convert_type_ignoring_params(ast_type.argument, type_params)
            return TApp(ctor, arg)
        
        elif isinstance(ast_type, FunctionType):
            domain = self.convert_type_ignoring_params(ast_type.param_type, type_params)
            codomain = self.convert_type_ignoring_params(ast_type.return_type, type_params)
            param_name = ast_type.param_name.value if ast_type.param_name else "_"
            return TPi(param_name, domain, codomain, ast_type.implicit)
        
        else:
            # For other types, use normal conversion
            return self.convert_type(ast_type, Context())
    
    def convert_type(self, ast_type: Type, ctx: Context) -> Term:
        """Convert AST type to core term."""
        if isinstance(ast_type, TypeConstructor):
            name = ast_type.name.value
            
            # Check for built-in types
            if name == "Type":
                return TType(Level(0))
            elif name in ["Nat", "Bool", "String"]:
                # These are just type constants
                return TConstructor(name, [])
            
            # Check for user-defined types
            if name in self.data_types:
                return TConstructor(name, [])
            
            # Check global types
            if name in self.global_types:
                # For now, return a constructor
                # TODO: Handle global references properly
                return TConstructor(name, [])
            
            # Otherwise, might be a type variable
            result = ctx.lookup(name)
            if result:
                index, _, _ = result
                return TVar(index)
            
            raise TypeCheckError(f"Unknown type: {name}")
        
        elif isinstance(ast_type, FunctionType):
            domain = self.convert_type(ast_type.param_type, ctx)
            
            if ast_type.param_name:
                # Dependent function type
                domain_val = domain.eval(ctx.environment)
                new_ctx = ctx.extend(ast_type.param_name.value, domain_val)
                codomain = self.convert_type(ast_type.return_type, new_ctx)
            else:
                # Non-dependent function type
                # Even for non-dependent types, we need to account for the parameter binding
                domain_val = domain.eval(ctx.environment)
                new_ctx = ctx.extend("_", domain_val)
                codomain = self.convert_type(ast_type.return_type, new_ctx)
            
            param_name = ast_type.param_name.value if ast_type.param_name else "_"
            return TPi(param_name, domain, codomain, ast_type.implicit)
        
        elif isinstance(ast_type, TypeApp):
            # Type application
            ctor = self.convert_type(ast_type.constructor, ctx)
            arg = self.convert_type(ast_type.argument, ctx)
            return TApp(ctor, arg)
        
        elif isinstance(ast_type, UniverseType):
            return TType(Level(ast_type.level))
        
        elif isinstance(ast_type, TypeVar):
            # Look up type variable
            result = ctx.lookup(ast_type.name.value)
            if result:
                index, _, _ = result
                return TVar(index)
            raise TypeCheckError(f"Unknown type variable: {ast_type.name.value}")
        
        else:
            raise TypeCheckError(f"Cannot convert type: {type(ast_type)}")
    
    def convert_expr(self, expr: Expr, ctx: Context) -> Term:
        """Convert AST expression to core term."""
        if isinstance(expr, Var):
            result = ctx.lookup(expr.name.value)
            if result:
                index, _, _ = result
                return TVar(index)
            
            # Check if it's a constructor
            if expr.name.value in self.constructors:
                # Return the constructor with no arguments
                return TConstructor(expr.name.value, [])
            
            # Check if it's a data type name
            if expr.name.value in self.data_types:
                # Data types used as expressions should be type constructors
                return TConstructor(expr.name.value, [])
            
            # Check global definitions (functions)
            if expr.name.value in self.global_types:
                # Create a global reference term
                return TGlobal(expr.name.value)
            
            raise TypeCheckError(f"Unknown variable: {expr.name.value}")
        
        elif isinstance(expr, Literal):
            return TLiteral(expr.value)
        
        elif isinstance(expr, Lambda):
            # Convert parameter type if present
            if expr.param_type:
                param_type_term = self.convert_type(expr.param_type, ctx)
                param_type_val = param_type_term.eval(ctx.environment)
            else:
                # We'll need to infer this during type checking
                raise TypeCheckError("Lambda requires type annotation")
            
            # Convert body in extended context
            new_ctx = ctx.extend(expr.param.value, param_type_val)
            body = self.convert_expr(expr.body, new_ctx)
            
            return TLambda(expr.param.value, body, expr.implicit)
        
        elif isinstance(expr, App):
            fun = self.convert_expr(expr.function, ctx)
            arg = self.convert_expr(expr.argument, ctx)
            return TApp(fun, arg, expr.implicit)
        
        elif isinstance(expr, Let):
            value = self.convert_expr(expr.value, ctx)
            
            # Infer type of value if not annotated
            if expr.type_annotation:
                type_term = self.convert_type(expr.type_annotation, ctx)
                type_val = type_term.eval(ctx.environment)
            else:
                type_val = self.infer_type(value, ctx)
            
            # Extend context with let binding
            value_val = value.eval(ctx.environment)
            new_ctx = ctx.extend(expr.name.value, type_val, value_val)
            body = self.convert_expr(expr.body, new_ctx)
            
            return TLet(expr.name.value, value, body)
        
        elif isinstance(expr, ListLiteral):
            # Desugar list literal to Cons/Nil
            # [] becomes Nil
            # [x, y, z] becomes Cons x (Cons y (Cons z Nil))
            result = TConstructor("Nil", [])
            for elem in reversed(expr.elements):
                elem_term = self.convert_expr(elem, ctx)
                # Build Cons elem result
                result = TConstructor("Cons", [elem_term, result])
            return result
        
        elif isinstance(expr, Case):
            # Convert case expression
            scrutinee = self.convert_expr(expr.scrutinee, ctx)
            scrutinee_type = self.infer_type(scrutinee, ctx)
            
            # Convert branches
            # For now, we don't properly handle the result type
            # TODO: Infer result type from branches
            branches_converted = []
            result_type = None
            
            for branch in expr.branches:
                # Check pattern and get extended context
                branch_ctx = self.check_pattern(branch.pattern, scrutinee_type, ctx)
                
                # Convert body in extended context
                body = self.convert_expr(branch.body, branch_ctx)
                
                # TODO: Check all branches have same type
                if result_type is None:
                    result_type = self.infer_type(body, branch_ctx)
                
                branches_converted.append((branch.pattern, body))
            
            # For now, return the scrutinee (case not implemented in core)
            # TODO: Add case to core language
            return scrutinee
        
        else:
            raise TypeCheckError(f"Cannot convert expression: {type(expr)}")


def type_check_module(module: Module) -> TypeChecker:
    """Type check a module and return the type checker with definitions."""
    checker = TypeChecker()
    checker.check_module(module)
    return checker