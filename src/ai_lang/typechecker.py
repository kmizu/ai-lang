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
from .errors import TypeCheckError, ErrorContext, ErrorKind, get_trace
from .error_reporting import format_type_for_display


def format_type_for_error(type_val: Value) -> str:
    """Format a type value for user-friendly error messages."""
    if isinstance(type_val, VType):
        return "Type"
    elif isinstance(type_val, VConstructor):
        if type_val.args:
            args_str = " ".join(format_type_for_error(arg) for arg in type_val.args)
            return f"{type_val.name} {args_str}"
        return type_val.name
    elif isinstance(type_val, VPi):
        domain_str = format_type_for_error(type_val.domain)
        # For simple function types, use arrow notation
        if not type_val.implicit and type_val.name == "_":
            # Try to peek at codomain for simple cases
            dummy_val = VNeutral(NVar(9999))
            codomain = type_val.codomain_closure.apply(dummy_val)
            codomain_str = format_type_for_error(codomain)
            return f"{domain_str} -> {codomain_str}"
        elif type_val.implicit:
            return f"{{{type_val.name} : {domain_str}}} -> ..."
        else:
            return f"({type_val.name} : {domain_str}) -> ..."
    elif isinstance(type_val, VNeutral):
        # Handle neutral terms (variables, applications)
        if isinstance(type_val.neutral, NVar):
            return f"<type variable>"
        else:
            return f"<neutral type>"
    else:
        # Fallback to string representation
        return str(type_val)


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
class ClassInfo:
    """Information about a type class."""
    name: str
    type_param: str
    superclasses: List[Tuple[str, Value]]  # [(class_name, type)]
    methods: Dict[str, Value]  # method_name -> method_type
    

@dataclass
class InstanceInfo:
    """Information about a type class instance."""
    class_name: str
    type_arg: Value
    constraints: List[Tuple[str, Value]]  # prerequisite constraints
    methods: Dict[str, Term]  # method_name -> implementation term


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
    type_classes: Dict[str, 'ClassInfo']  # Type class definitions
    instances: Dict[str, List['InstanceInfo']]  # class_name -> list of instances
    source_code: Optional[str]  # Source code for error reporting
    filename: Optional[str]  # Filename for error reporting
    current_location: Optional[SourceLocation]  # Current location being checked
    
    def __init__(self, source_code: Optional[str] = None, filename: Optional[str] = None):
        self.data_types = {}
        self.constructors = {}
        self.global_types = {}
        self.global_values = {}
        self.modules = {}
        self.module_paths = ['.', 'lib']  # Default search paths
        self.elaborated_terms = {}
        self.source_code = source_code
        self.filename = filename
        self.current_location = None
        self.type_classes = {}
        self.instances = {}
        
        # Initialize built-in types
        self._init_builtins()
    
    def _init_builtins(self):
        """Initialize built-in types and constructors."""
        # Add built-in types: Nat, Bool, String
        self.global_types["Nat"] = VType(Level(0))
        self.global_types["Bool"] = VType(Level(0))
        self.global_types["String"] = VType(Level(0))
        
        # Add Bool constructors
        true_info = DataConstructorInfo("True", VConstructor("Bool", []), "Bool", [], [])
        false_info = DataConstructorInfo("False", VConstructor("Bool", []), "Bool", [], [])
        self.constructors["True"] = true_info
        self.constructors["False"] = false_info
    
    def raise_error(self, message: str, kind: Optional[ErrorKind] = None, 
                   location: Optional[SourceLocation] = None, **kwargs) -> None:
        """Raise a type check error with proper context."""
        loc = location or self.current_location
        context = ErrorContext(
            source_code=self.source_code,
            filename=self.filename,
            location=loc,
            kind=kind,
            **kwargs
        )
        raise TypeCheckError(message, location=loc, context=context)
    
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
    
    def add_type_class(self, decl: ClassDecl, ctx: Context) -> None:
        """Add a type class declaration."""
        # Create context with type parameter
        param_ctx = ctx.extend(decl.type_param.value, VType(Level(0)))
        
        # Convert superclass constraints
        superclasses = []
        for class_name, type_arg in decl.superclasses:
            type_term = self.convert_type(type_arg, param_ctx)
            type_val = self.eval_with_globals(type_term, param_ctx.environment)
            superclasses.append((class_name.value, type_val))
        
        # Convert method types
        methods = {}
        for method_name, method_type in decl.methods:
            method_type_term = self.convert_type(method_type, param_ctx)
            method_type_val = self.eval_with_globals(method_type_term, param_ctx.environment)
            methods[method_name.value] = method_type_val
        
        # Store class info
        class_info = ClassInfo(
            name=decl.name.value,
            type_param=decl.type_param.value,
            superclasses=superclasses,
            methods=methods
        )
        self.type_classes[decl.name.value] = class_info
        
        # Add class type to globals (as a Type -> Type function)
        # class Eq A means Eq : Type -> Type
        class_type = VPi(decl.type_param.value, VType(Level(0)), 
                        Closure(param_ctx.environment, TType(Level(0))), False)
        self.add_global(decl.name.value, class_type)
    
    def add_instance(self, decl: InstanceDecl, ctx: Context) -> None:
        """Add a type class instance."""
        # Check that the class exists
        if decl.class_name.value not in self.type_classes:
            raise TypeCheckError(f"Unknown type class: {decl.class_name.value}")
        
        class_info = self.type_classes[decl.class_name.value]
        
        # Convert the instance type
        type_term = self.convert_type(decl.type, ctx)
        type_val = self.eval_with_globals(type_term, ctx.environment)
        
        # Convert constraints
        constraints = []
        for class_name, constraint_type in decl.constraints:
            constraint_term = self.convert_type(constraint_type, ctx)
            constraint_val = self.eval_with_globals(constraint_term, ctx.environment)
            constraints.append((class_name.value, constraint_val))
        
        # Type check method implementations
        method_impls = {}
        for method_name, impl_expr in decl.methods:
            if method_name.value not in class_info.methods:
                raise TypeCheckError(f"Method {method_name.value} not in class {decl.class_name.value}")
            
            # Get expected method type by substituting the instance type
            expected_type = self.substitute_in_type(class_info.methods[method_name.value], 
                                                  class_info.type_param, type_val)
            
            # Convert and check implementation with type guidance
            impl_term = self.convert_expr_with_type(impl_expr, expected_type, ctx)
            method_impls[method_name.value] = impl_term
        
        # Check that all methods are implemented
        for method_name in class_info.methods:
            if method_name not in method_impls:
                raise TypeCheckError(f"Missing implementation for method {method_name}")
        
        # Store instance info
        instance_info = InstanceInfo(
            class_name=decl.class_name.value,
            type_arg=type_val,
            constraints=constraints,
            methods=method_impls
        )
        
        if decl.class_name.value not in self.instances:
            self.instances[decl.class_name.value] = []
        self.instances[decl.class_name.value].append(instance_info)
    
    def substitute_in_type(self, type_val: Value, param_name: str, arg_val: Value) -> Value:
        """Substitute a type argument in a type value."""
        # For type class methods, we need to substitute the type parameter
        # The type parameter 'A' is represented as de Bruijn index 0 in the method type
        # We'll do a simple substitution for now
        
        if isinstance(type_val, VPi):
            # For a function type like A -> A -> Bool, we need to substitute A with the concrete type
            # First, substitute in the domain
            new_domain = self.substitute_in_type(type_val.domain, param_name, arg_val)
            
            # For the codomain, we need to be careful
            # If this Pi binds a variable, we need to preserve the binding
            # Create a dummy variable to evaluate the codomain
            dummy_var = VNeutral(NVar(1000))  # Use a high level to avoid conflicts
            codomain_val = type_val.codomain_closure.apply(dummy_var)
            new_codomain = self.substitute_in_type(codomain_val, param_name, arg_val)
            
            # Create a new closure that returns the substituted codomain
            return VPi(type_val.name, new_domain, ConstantClosure(new_codomain), type_val.implicit)
            
        elif isinstance(type_val, VNeutral):
            # Check if this is a type variable (de Bruijn index 0)
            if isinstance(type_val.neutral, NVar) and type_val.neutral.level == 0:
                # This is our type parameter, substitute it
                return arg_val
            return type_val
            
        elif isinstance(type_val, VConstructor):
            # Substitute in constructor arguments if any
            new_args = [self.substitute_in_type(arg, param_name, arg_val) for arg in type_val.args]
            return VConstructor(type_val.name, new_args)
            
        # For other value types, return as-is
        return type_val
    
    def resolve_instance(self, class_name: str, type_arg: Value, ctx: Context) -> Optional[InstanceInfo]:
        """Resolve a type class instance."""
        if class_name not in self.instances:
            return None
        
        # Look for matching instance
        for instance in self.instances[class_name]:
            if self.equal_types(instance.type_arg, type_arg, ctx):
                # Check that constraints are satisfied
                # For now, we'll assume they are - full implementation would recursively resolve
                return instance
        
        return None
    
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
        
        elif isinstance(decl, ClassDecl):
            self.add_type_class(decl, ctx)
        
        elif isinstance(decl, InstanceDecl):
            self.add_instance(decl, ctx)
        
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
        
        # Handle constraint types (e.g., Eq A => A -> A -> Bool)
        if isinstance(current_type, VConstraintPi):
            # Add type class methods to context
            for constraint in current_type.constraints:
                class_name = constraint.class_name
                if class_name in self.type_classes:
                    class_info = self.type_classes[class_name]
                    # Add each method of the type class to the context
                    for method_name, method_type in class_info.methods.items():
                        # The method type needs to be instantiated with the constraint's type argument
                        instantiated_type = self.substitute_in_type(method_type, class_info.type_param, constraint.type_arg)
                        new_ctx = new_ctx.extend(method_name, instantiated_type)
            
            # Unwrap to the body type
            current_type = current_type.body
        
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
        body_term = self.convert_expr_with_type(clause.body, current_type, new_ctx)
    
    def check_pattern(self, pattern: Pattern, expected_type: Value, ctx: Context) -> Context:
        """Type check a pattern and return extended context."""
        if isinstance(pattern, PatternVar):
            # Variable pattern - just bind the variable
            return ctx.extend(pattern.name.value, expected_type)
        
        elif isinstance(pattern, PatternConstructor):
            # Constructor pattern
            ctor_name = pattern.constructor.value
            if ctor_name not in self.constructors:
                self.raise_error(f"Unknown constructor in pattern: {ctor_name}",
                           kind=ErrorKind.UNKNOWN_CONSTRUCTOR,
                           location=pattern.constructor.location if hasattr(pattern.constructor, 'location') else None,
                           available_names=list(self.constructors.keys()))
            
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
        # Add type derivation trace
        trace = get_trace()
        if trace.enabled:
            ctx_dict = {b.name: format_type_for_display(b.type) for b in ctx.bindings[-3:]}  # Show last 3 bindings
            trace.add_step(f"Inferring type of {type(term).__name__}", 
                          location=self.current_location,
                          context=ctx_dict)
        
        if isinstance(term, TType):
            # Type : Type(n+1)
            result = VType(term.level.succ())
            if trace.enabled:
                trace.add_step(f"Type has type Type{term.level.succ().n}", result=format_type_for_display(result))
            return result
        
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
        
        elif isinstance(term, TConstraintPi):
            # Check that all constraints are well-formed
            for class_name, type_arg in term.constraints:
                type_arg_type = self.infer_type(type_arg, ctx)
                self.check_is_type(type_arg_type, ctx)
            
            # Check body type
            body_type = self.infer_type(term.body, ctx)
            self.check_is_type(body_type, ctx)
            
            # Constraint types have type Type
            return VType(Level(0))
        
        elif isinstance(term, TLambda):
            self.raise_error("Cannot infer type of lambda without annotation",
                           kind=ErrorKind.MISSING_TYPE_ANNOTATION)
        
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
            
            # Handle constraint types
            if isinstance(fun_type, VConstraintPi):
                # We need to resolve instances for the constraints
                # For now, just unwrap to the body type
                # TODO: Implement full instance resolution
                fun_type = fun_type.body
            
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
                    self.raise_error(
                        f"Type mismatch: expected {format_type_for_error(expected_type)}, got {format_type_for_error(inferred_type)}",
                        kind=ErrorKind.TYPE_MISMATCH,
                        expected=format_type_for_error(expected_type),
                        actual=format_type_for_error(inferred_type)
                    )
        
        else:
            # For other terms, first try polymorphic subsumption
            inferred_type = self.infer_type(term, ctx)
            
            # Check if we can use polymorphic subsumption
            if self.can_subsume(inferred_type, expected_type, ctx):
                return  # Subsumption successful
            
            # Otherwise, check for exact equality
            if not self.equal_types(inferred_type, expected_type, ctx):
                raise TypeCheckError(
                    f"Type mismatch: expected {format_type_for_error(expected_type)}, got {format_type_for_error(inferred_type)}"
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
            
        except TypeCheckError:
            # Re-raise TypeCheckError with our improved messages
            raise
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
                        raise TypeCheckError(f"Cannot infer implicit type argument\nReason: {e}")
                else:
                    # For non-type implicit arguments, we need more context
                    raise TypeCheckError(f"Cannot infer implicit argument of type {format_type_for_error(current_type.domain)}")
                
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
        """Check if two types are equal (including alpha-equivalence)."""
        # Implement proper definitional equality with alpha-equivalence
        
        if type(type1) != type(type2):
            return False
        
        if isinstance(type1, VType):
            return type1.level == type2.level
        
        elif isinstance(type1, VPi):
            # For Pi types, we need to check:
            # 1. The implicit flags match
            # 2. The domains are equal
            # 3. The codomains are equal under alpha-equivalence
            
            # Check implicit flags match
            if type1.implicit != type2.implicit:
                return False
            
            # Check domains are equal
            if not self.equal_types(type1.domain, type2.domain, ctx):
                return False
            
            # Check codomains are equal (with a fresh variable)
            # This implements alpha-equivalence: we use the same fresh variable
            # for both codomains, so parameter names don't matter
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
    
    def can_subsume(self, actual_type: Value, expected_type: Value, ctx: Context) -> bool:
        """Check if actual_type can be subsumed to expected_type (polymorphic subsumption).
        
        This handles cases like:
        - A polymorphic type can be specialized to a monomorphic type
        - {A : Type} -> A -> A  can be used where  Nat -> Nat  is expected
        """
        # First check if types are already equal
        if self.equal_types(actual_type, expected_type, ctx):
            return True
        
        # Handle polymorphic subsumption for Pi types
        if isinstance(actual_type, VPi) and actual_type.implicit:
            # If actual type has implicit parameters, we can instantiate them
            # Try to find an instantiation that makes the types equal
            
            # For now, we'll handle the simple case where we can instantiate
            # the implicit parameter and continue checking
            if isinstance(actual_type.domain, VType):
                # The implicit parameter is a Type
                # We need to figure out what to instantiate it with
                
                # Try instantiating with different types based on expected_type
                if isinstance(expected_type, VPi):
                    # Try to instantiate to make domains match
                    # This is a simplified approach - full implementation would use unification
                    
                    # Instantiate the implicit parameter with the expected domain
                    instantiated = actual_type.codomain_closure.apply(expected_type.domain)
                    return self.can_subsume(instantiated, expected_type, ctx)
                elif isinstance(expected_type, VConstructor):
                    # Expected type is a concrete type like Nat
                    # Try instantiating with that type
                    instantiated = actual_type.codomain_closure.apply(expected_type)
                    return self.equal_types(instantiated, expected_type, ctx)
        
        return False
    
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
        
        elif isinstance(ast_type, ConstraintType):
            # Convert constraints
            constraint_terms = []
            for class_name, type_arg in ast_type.constraints:
                type_arg_term = self.convert_type(type_arg, ctx)
                constraint_terms.append((class_name.value, type_arg_term))
            
            # Convert body
            body_term = self.convert_type(ast_type.body, ctx)
            
            return TConstraintPi(constraint_terms, body_term)
        
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
        # Update current location if available
        if hasattr(expr, 'name') and hasattr(expr.name, 'location'):
            self.current_location = expr.name.location
        elif hasattr(expr, 'location'):
            self.current_location = expr.location
            
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
            
            # Get all available names for suggestions
            available_names = []
            # Add local variables
            for binding in ctx.bindings:
                available_names.append(binding.name)
            # Add constructors
            available_names.extend(self.constructors.keys())
            # Add global names
            available_names.extend(self.global_types.keys())
            
            # Find similar names
            from .error_reporting import suggest_similar_names
            similar = suggest_similar_names(expr.name.value, available_names)
            
            self.raise_error(f"Unknown variable: {expr.name.value}",
                           kind=ErrorKind.UNKNOWN_VARIABLE,
                           location=expr.name.location if hasattr(expr.name, 'location') else None,
                           actual=expr.name.value,
                           available_names=available_names,
                           similar_names=similar)
        
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
    
    def convert_expr_with_type(self, expr: Expr, expected_type: Value, ctx: Context) -> Term:
        """Convert an expression to a term with type guidance."""
        # Special handling for lambdas without type annotations
        if isinstance(expr, Lambda) and not expr.param_type:
            if not isinstance(expected_type, VPi):
                raise TypeCheckError(f"Expected function type for lambda, got {expected_type}")
            
            # Use the domain type from the expected Pi type
            param_type_val = expected_type.domain
            
            # Convert body in extended context
            new_ctx = ctx.extend(expr.param.value, param_type_val)
            # Apply the parameter to get the codomain type
            param_var = VNeutral(NVar(len(ctx.bindings)))
            codomain_type = expected_type.codomain_closure.apply(param_var)
            body = self.convert_expr_with_type(expr.body, codomain_type, new_ctx)
            
            return TLambda(expr.param.value, body, expr.implicit)
        
        # For other expressions, use regular conversion and check
        term = self.convert_expr(expr, ctx)
        self.check_type(term, expected_type, ctx)
        return term


def type_check_module(module: Module, module_paths: Optional[List[str]] = None,
                     return_checker: bool = False, source_code: Optional[str] = None,
                     filename: Optional[str] = None) -> Union[TypeChecker, None]:
    """Type check a module and optionally return the type checker with definitions.
    
    Args:
        module: The module to type check
        module_paths: Optional list of paths to search for modules
        return_checker: If True, returns the TypeChecker instance. If False, returns None.
        source_code: Optional source code for error reporting
        filename: Optional filename for error reporting
        
    Returns:
        TypeChecker instance if return_checker is True, None otherwise
    """
    checker = TypeChecker(source_code=source_code, filename=filename)
    if module_paths:
        checker.module_paths = module_paths
    checker.check_module(module)
    
    if return_checker:
        return checker
    else:
        return None