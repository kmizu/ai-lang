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
from .errors import TypeCheckError


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
class ModuleInfo:
    """Information about a loaded module."""
    name: str
    exports: Dict[str, Union[Value, DataDecl, DataConstructorInfo]]
    path: str


@dataclass
class TypeChecker:
    """Type checker state."""
    data_types: Dict[str, DataDecl]
    constructors: Dict[str, DataConstructorInfo]
    global_types: Dict[str, Value]
    global_values: Dict[str, Value]
    modules: Dict[str, 'ModuleInfo']  # Loaded modules
    module_paths: List[str]  # Search paths for modules
    elaborated_terms: Dict[str, Term]  # Store elaborated terms for evaluation
    
    def __init__(self):
        self.data_types = {}
        self.constructors = {}
        self.global_types = {}
        self.global_values = {}
        self.modules = {}
        self.module_paths = ['.', 'lib']  # Default search paths
        self.elaborated_terms = {}
    
    def eval_with_globals(self, term: Term, env: Environment) -> Value:
        """Evaluate a term with global lookups."""
        if isinstance(term, TGlobal):
            # Look up global value
            if term.name in self.global_values:
                return self.global_values[term.name]
            elif term.name in self.global_types:
                return self.global_types[term.name]
            else:
                # Return neutral for unknown globals
                return VNeutral(NGlobal(term.name))
        elif isinstance(term, TApp):
            # Handle application with globals
            fun_val = self.eval_with_globals(term.function, env)
            arg_val = self.eval_with_globals(term.argument, env)
            
            # Special handling for type-level applications
            if isinstance(fun_val, VPi):
                # Apply Pi type by instantiating the codomain
                return fun_val.codomain_closure.apply(arg_val)
            else:
                return apply(fun_val, arg_val)
        else:
            # Use standard evaluation
            return term.eval(env)
    
    def add_data_type(self, decl: DataDecl, ctx: Context) -> None:
        """Add a data type declaration."""
        self.data_types[decl.name.value] = decl
        
        # First, add the data type itself to the context
        # For parameterized types, create a function type
        if decl.type_params:
            # Create a Pi type for each parameter
            # Start with the result type as a term
            result_term = TType(Level(0))
            
            # Build the type backwards (right-to-left) for proper Pi nesting
            for param in reversed(decl.type_params):
                # Each parameter is a Type -> ... -> Type function
                result_term = TPi(param.value, TType(Level(0)), result_term, False)
            
            # Evaluate the complete type
            data_type = self.eval_with_globals(result_term, ctx.environment)
        else:
            data_type = VType(Level(0))
        
        self.add_global(decl.name.value, data_type)
        
        # Create context with type parameters
        param_ctx = ctx
        for param in decl.type_params:
            param_ctx = param_ctx.extend(param.value, VType(Level(0)))
        
        # Add constructors
        for ctor in decl.constructors:
            # Convert constructor type in the context with type parameters
            ctor_type_term = self.convert_type(ctor.type, param_ctx)
            
            # For parameterized constructors, we need to add implicit parameters
            if decl.type_params:
                # Build the constructor type with implicit type parameters
                for param in reversed(decl.type_params):
                    ctor_type_term = TPi(
                        param.value,
                        TType(Level(0)),
                        ctor_type_term,
                        implicit=True  # Type parameters are implicit
                    )
            
            ctor_type_val = self.eval_with_globals(ctor_type_term, ctx.environment)
            
            info = DataConstructorInfo(
                name=ctor.name.value,
                type=ctor_type_val,
                data_type=decl.name.value,
                parameters=[p.value for p in decl.type_params],
                indices=[(n.value, self.convert_type(t, param_ctx).eval(param_ctx.environment)) 
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
        
        # Process imports first
        for imp in module.imports:
            self.process_import(imp, ctx)
        
        # Then check declarations
        for decl in module.declarations:
            self.check_declaration(decl, ctx)
        
        # Process exports if this is a named module
        if module.name:
            self.process_exports(module)
    
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
            # Don't fully evaluate - preserve the structure for type checking
            if isinstance(type_term, TApp) and isinstance(type_term.function, TGlobal):
                # Special case for applied data types - store as neutral
                type_val = VNeutral(self.term_to_neutral(type_term))
            else:
                type_val = self.eval_with_globals(type_term, ctx.environment)
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
        
        # Keep track of implicit parameters for later use
        implicit_params = []
        
        # First, handle implicit parameters that don't have patterns
        while isinstance(current_type, VPi) and current_type.implicit:
            # Add implicit parameter to context
            param_name = current_type.name
            implicit_params.append((param_name, current_type.domain))
            new_ctx = new_ctx.extend(param_name, current_type.domain)
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
            domain_val = self.eval_with_globals(term.domain, ctx.environment)
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
            # Special handling for applications that may need delayed inference
            # Check if this is a top-level application of a function with implicit params
            from .delayed_inference import infer_with_delayed_inference
            
            # Try delayed inference first for better results
            if not term.implicit:
                try:
                    # Check if the base function has implicit parameters
                    base_fun, args = self._get_base_function(term)
                    if isinstance(base_fun, TGlobal) and base_fun.name in self.global_types:
                        base_type = self.global_types[base_fun.name]
                        if isinstance(base_type, VPi) and base_type.implicit:
                            # Use delayed inference
                            result_term = infer_with_delayed_inference(term, ctx, self)
                            return self.infer_type(result_term, ctx)
                except Exception as e:
                    # print(f"DEBUG: Delayed inference failed: {e}")
                    # Fall through to regular inference
                    pass
            
            # Regular inference
            # Infer function type
            fun_type = self.infer_type(term.function, ctx)
            
            # Debug output disabled
            # print(f"DEBUG infer_type TApp: function={term.function}, arg={term.argument}")
            # print(f"DEBUG infer_type TApp: fun_type={fun_type}")
            
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
            arg_val = self.eval_with_globals(term.argument, ctx.environment)
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
        
        # Neutral terms representing type applications are also valid types
        if isinstance(type_val, VNeutral) and self.is_type_application(type_val.neutral):
            return  # Data type applications are valid types
        
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
        # First try the new constraint-based inference
        try:
            from .constraints import infer_implicit_args_with_constraints
            
            # Collect all explicit arguments from nested applications
            explicit_args = []
            current = app
            base_fun = None
            
            # Unwrap nested applications to get all arguments
            while isinstance(current, TApp) and not current.implicit:
                explicit_args.insert(0, current.argument)
                if isinstance(current.function, TApp):
                    current = current.function
                else:
                    base_fun = current.function
                    break
            
            if base_fun is None:
                base_fun = app.function
                explicit_args = [app.argument]
            
            # print(f"DEBUG: Collected {len(explicit_args)} explicit args for inference")
            
            # Infer implicit arguments using constraints
            implicit_args, substitution = infer_implicit_args_with_constraints(
                app, fun_type, explicit_args, ctx, self
            )
            
            # Build the application with implicit arguments
            result = base_fun  # The base function
            for implicit_arg in implicit_args:
                result = TApp(result, implicit_arg, implicit=True)
            
            # Add the explicit arguments back
            for explicit_arg in explicit_args:
                result = TApp(result, explicit_arg, implicit=False)
            
            return result
            
        except Exception as e:
            # Fall back to the old approach for single implicit parameters
            current_app = app
            current_type = fun_type
            
            # Count implicit parameters
            implicit_count = 0
            temp_type = fun_type
            while isinstance(temp_type, VPi) and temp_type.implicit:
                implicit_count += 1
                dummy = VNeutral(NVar(1000 + implicit_count))
                temp_type = temp_type.codomain_closure.apply(dummy)
            
            if implicit_count > 1:
                # Multiple implicit parameters - use constraint solver failed
                # Suppress warning for now
                # print(f"Warning: Function has {implicit_count} implicit parameters, constraint solving failed: {e}")
                pass
            
            # Simple approach: infer one implicit at a time
            # This works for single implicit parameters but not multiple
            inferred_count = 0
            while isinstance(current_type, VPi) and current_type.implicit and inferred_count < 1:
                # We need to infer the implicit argument
                if isinstance(current_type.domain, VType):
                    # The implicit argument is a Type
                    # Try to infer it from the explicit argument
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
                arg_val = self.eval_with_globals(implicit_arg, ctx.environment)
                current_type = current_type.codomain_closure.apply(arg_val)
                inferred_count += 1
            
            return current_app
    
    def infer_implicit_from_args(self, fun_type: Value, placeholders: List[Optional[Term]], 
                                 explicit_args: List[Term], ctx: Context) -> List[Optional[Term]]:
        """Infer implicit arguments from explicit arguments."""
        # This is a simplified inference algorithm
        # A full implementation would use constraint solving
        
        current_type = fun_type
        result = []
        placeholder_idx = 0
        
        # Skip implicit parameters (we're inferring these)
        while isinstance(current_type, VPi) and current_type.implicit:
            result.append(None)  # Will fill in later
            # Use dummy value to proceed
            dummy_val = VNeutral(NVar(1000 + placeholder_idx))
            current_type = current_type.codomain_closure.apply(dummy_val)
            placeholder_idx += 1
        
        # Now match explicit arguments
        if explicit_args and isinstance(current_type, VPi):
            # Infer from first explicit argument
            try:
                arg_type = self.infer_type(explicit_args[0], ctx)
                # For const example: if first arg has type Nat, then A = Nat
                if result:
                    result[0] = self.value_to_term(arg_type, ctx)
                
                # Apply the first inferred type and continue
                if result[0]:
                    current_type = fun_type
                    # Apply first implicit
                    if isinstance(current_type, VPi) and current_type.implicit:
                        arg_val = self.eval_with_globals(result[0], ctx.environment)
                        current_type = current_type.codomain_closure.apply(arg_val)
                        
                        # Try to infer second implicit from second explicit arg
                        if len(result) > 1 and len(explicit_args) > 1:
                            # Skip to next explicit parameter position
                            while isinstance(current_type, VPi) and current_type.implicit:
                                dummy_val = VNeutral(NVar(1001))
                                current_type = current_type.codomain_closure.apply(dummy_val)
                            
                            # Skip first explicit parameter
                            if isinstance(current_type, VPi):
                                dummy_val = VNeutral(NVar(1002))
                                current_type = current_type.codomain_closure.apply(dummy_val)
                            
                            # Now we should be at the second explicit parameter
                            # Infer from second argument
                            try:
                                arg2_type = self.infer_type(explicit_args[1], ctx)
                                result[1] = self.value_to_term(arg2_type, ctx)
                            except:
                                pass
            except:
                pass
        
        return result
    
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
            # For neutral values, we need to check if they represent the same computation
            return self.equal_neutrals(type1.neutral, type2.neutral)
        
        elif isinstance(type1, VNeutral) and isinstance(type2, VType):
            # Special case: a neutral term might represent a type
            # This happens with parameterized data types like (List Nat)
            # Check if the neutral is a type application
            if self.is_type_application(type1.neutral):
                return True  # Assume it's a valid type for now
            return False
        
        elif isinstance(type1, VType) and isinstance(type2, VNeutral):
            # Symmetric case
            return self.equal_types(type2, type1, ctx)
        
        # For other types, use structural equality
        # TODO: Implement proper equality
        return str(type1) == str(type2)
    
    def equal_neutrals(self, n1: Neutral, n2: Neutral) -> bool:
        """Check if two neutral terms are equal."""
        if type(n1) != type(n2):
            return False
        
        if isinstance(n1, NVar) and isinstance(n2, NVar):
            return n1.level == n2.level
        elif isinstance(n1, NGlobal) and isinstance(n2, NGlobal):
            return n1.name == n2.name
        elif isinstance(n1, NApp) and isinstance(n2, NApp):
            return (self.equal_neutrals(n1.function, n2.function) and
                    self.equal_values(n1.argument, n2.argument))
        else:
            return False
    
    def equal_values(self, v1: Value, v2: Value) -> bool:
        """Check if two values are equal."""
        # For now, use string comparison
        return str(v1) == str(v2)
    
    def is_type_application(self, neutral: Neutral) -> bool:
        """Check if a neutral term is a type application to a data type."""
        if isinstance(neutral, NApp):
            # Check if the function is a data type
            if isinstance(neutral.function, NGlobal):
                return neutral.function.name in self.data_types
        return False
    
    def term_to_neutral(self, term: Term) -> Neutral:
        """Convert a term to a neutral value (for preserving structure)."""
        if isinstance(term, TGlobal):
            return NGlobal(term.name)
        elif isinstance(term, TApp):
            fun_neutral = self.term_to_neutral(term.function)
            # We need to evaluate the argument to a value
            arg_val = self.eval_with_globals(term.argument, Environment([]))
            return NApp(fun_neutral, arg_val, term.implicit)
        else:
            raise TypeCheckError(f"Cannot convert {term} to neutral")
    
    def value_to_term(self, val: Value, ctx: Context) -> Term:
        """Convert a value to a term (reification)."""
        # This is a simplified version - full implementation would handle all cases
        if isinstance(val, VType):
            return TType(val.level)
        elif isinstance(val, VConstructor):
            if val.args:
                # Build constructor application
                result = TConstructor(val.name, [])
                for arg in val.args:
                    arg_term = self.value_to_term(arg, ctx)
                    result = TApp(result, arg_term)
                return result
            else:
                return TConstructor(val.name, [])
        elif isinstance(val, VNeutral):
            return self.neutral_to_term(val.neutral, ctx)
        else:
            # For other values, use quote
            return val.quote(len(ctx.environment.values))
    
    def neutral_to_term(self, neutral: Neutral, ctx: Context) -> Term:
        """Convert a neutral to a term."""
        if isinstance(neutral, NVar):
            return TVar(neutral.level)
        elif isinstance(neutral, NGlobal):
            return TGlobal(neutral.name)
        elif isinstance(neutral, NApp):
            fun_term = self.neutral_to_term(neutral.function, ctx)
            arg_term = self.value_to_term(neutral.argument, ctx)
            return TApp(fun_term, arg_term, neutral.implicit)
        else:
            raise TypeCheckError(f"Cannot convert neutral {neutral} to term")
    
    def process_import(self, imp: Import, ctx: Context) -> None:
        """Process an import statement."""
        module_name = imp.module.value
        
        # Check if module is already loaded
        if module_name not in self.modules:
            # Load the module
            self.load_module(module_name)
        
        module_info = self.modules[module_name]
        
        # Import specific names or all exports
        if imp.items:
            # Import specific names
            for item in imp.items:
                if item.name.value not in module_info.exports:
                    raise TypeCheckError(f"Module {module_name} does not export {item.name.value}")
                self.import_name(item.name.value, module_info.exports[item.name.value], item.alias)
        else:
            # Import all exports
            prefix = imp.alias.value if imp.alias else ""
            for export_name, export_value in module_info.exports.items():
                import_name = f"{prefix}.{export_name}" if prefix else export_name
                self.import_name(import_name, export_value, None)
    
    def import_name(self, name: str, value: Union[Value, DataDecl, DataConstructorInfo], alias: Optional[Name]) -> None:
        """Import a single name."""
        import_name = alias.value if alias else name
        
        if isinstance(value, Value):
            # It's a type or value
            self.global_types[import_name] = value
        elif isinstance(value, DataDecl):
            # It's a data type
            self.data_types[import_name] = value
        elif isinstance(value, DataConstructorInfo):
            # It's a constructor
            self.constructors[import_name] = value
    
    def load_module(self, module_name: str) -> None:
        """Load a module from disk."""
        import os
        from .parser import Parser
        from .lexer import Lexer
        
        # Find module file
        module_file = None
        for path in self.module_paths:
            candidate = os.path.join(path, f"{module_name}.ai")
            if os.path.exists(candidate):
                module_file = candidate
                break
        
        if not module_file:
            raise TypeCheckError(f"Cannot find module {module_name}")
        
        # Parse the module
        with open(module_file, 'r') as f:
            source = f.read()
        
        lexer = Lexer(source)
        tokens = list(lexer.tokenize())
        parser = Parser(tokens)
        module = parser.parse_module()
        
        # Type check the module
        sub_checker = TypeChecker()
        sub_checker.module_paths = self.module_paths
        sub_checker.check_module(module)
        
        # Collect exports
        exports = {}
        if module.exports:
            # Explicit exports
            for export in module.exports:
                for name in export.names:
                    export_name = name.value
                    if export_name in sub_checker.global_types:
                        exports[export_name] = sub_checker.global_types[export_name]
                    elif export_name in sub_checker.data_types:
                        exports[export_name] = sub_checker.data_types[export_name]
                    elif export_name in sub_checker.constructors:
                        exports[export_name] = sub_checker.constructors[export_name]
                    else:
                        raise TypeCheckError(f"Cannot export undefined name {export_name}")
        else:
            # No explicit exports - export everything
            exports.update({name: val for name, val in sub_checker.global_types.items()})
            exports.update({name: val for name, val in sub_checker.data_types.items()})
            exports.update({name: val for name, val in sub_checker.constructors.items()})
        
        # Store module info
        self.modules[module_name] = ModuleInfo(
            name=module_name,
            exports=exports,
            path=module_file
        )
    
    def process_exports(self, module: Module) -> None:
        """Process module exports."""
        # This is handled during module loading
        pass
    
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
                # For data types, we should return a global reference
                # so they can be applied to arguments
                return TGlobal(name)
            
            # Check global types
            if name in self.global_types:
                # Return global reference for proper type application
                return TGlobal(name)
            
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
                index, type_val, _ = result
                # Special handling: if this is a Type being used as an expression,
                # we might need to handle it differently
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
    
    def _get_base_function(self, term: Term) -> Tuple[Term, List[Term]]:
        """Get the base function and arguments from an application."""
        args = []
        current = term
        
        while isinstance(current, TApp) and not current.implicit:
            args.insert(0, current.argument)
            current = current.function
        
        return current, args


def type_check_module(module: Module, module_paths: Optional[List[str]] = None) -> TypeChecker:
    """Type check a module and return the type checker with definitions."""
    checker = TypeChecker()
    if module_paths:
        checker.module_paths = module_paths
    checker.check_module(module)
    return checker