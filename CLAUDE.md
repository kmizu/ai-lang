# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ai-lang is a dependently-typed programming language implemented in Python. The language supports:
- Dependent types (types can depend on values)
- Basic data types (integers, booleans, strings)
- Collection types with size information (Vec n a)
- Pattern matching
- Type inference with bidirectional type checking

## Development Commands

```bash
# Install dependencies
poetry install

# Run tests
poetry run pytest

# Run specific test
poetry run pytest tests/test_lexer.py::test_function_name

# Type check with mypy
poetry run mypy src

# Format code
poetry run black src tests

# Lint code
poetry run ruff src tests

# Run the REPL
poetry run ai-lang
# Or directly:
PYTHONPATH=src python3 src/ai_lang/cli.py

# Run a file
poetry run ai-lang examples/minimal.ai
# Or directly:
PYTHONPATH=src python3 src/ai_lang/cli.py examples/minimal.ai
```

## Project Structure

```
ai-lang/
├── src/ai_lang/
│   ├── __init__.py       # Package initialization
│   ├── ast.py            # AST node definitions
│   ├── lexer.py          # Lexical analyzer
│   ├── parser.py         # Parser (to be implemented)
│   ├── typechecker.py    # Type checker (to be implemented)
│   ├── evaluator.py      # Interpreter (to be implemented)
│   └── cli.py            # Command-line interface
├── tests/                # Test suite
├── examples/             # Example programs
└── docs/
    └── syntax.md         # Language syntax specification
```

## Architecture

### Type System
- Based on dependent type theory with universe hierarchy
- Bidirectional type checking for better inference
- Support for implicit arguments
- Normalization by evaluation (NbE) for type equality

### Key Components
1. **Lexer**: Tokenizes source code, handles comments and layout
2. **Parser**: Builds AST using recursive descent or Lark grammar
3. **Type Checker**: Implements bidirectional type checking with:
   - Type synthesis (inferring types from terms)
   - Type checking (checking terms against types)
   - Unification for implicit arguments
4. **Evaluator**: Call-by-value interpreter with environments

### AST Design
- Immutable dataclasses with frozen=True
- Separate types for Type and Expr nodes
- Pattern matching support through Pattern hierarchy
- Source location tracking for error messages

## Implementation Status

### Completed Features
- ✅ Lexer with full token support
- ✅ Parser with comprehensive AST generation
- ✅ Core type system with dependent types
- ✅ Bidirectional type checker
- ✅ Evaluator/interpreter with pattern matching
- ✅ REPL with color support
- ✅ Basic data types (Nat, Bool, String)
- ✅ List literals with syntactic sugar
- ✅ Function definitions and calls
- ✅ Polymorphic types with implicit parameters
- ✅ Basic implicit type argument inference
- ✅ Explicit type application syntax `{Type}`
- ✅ Parameterized data types (e.g., `List A`)
- ✅ Module system with imports and exports
- ✅ **Constraint-based type inference for multiple implicit parameters**
- ✅ **Delayed inference for handling partial applications**
- ✅ **Implicit parameters accessible in function bodies**
- ✅ **Optimization passes: eta-reduction, dead code elimination, inlining**

### Known Issues and Limitations

1. **Implicit Type Inference** ✅ FULLY RESOLVED
   - ✅ Can now infer multiple implicit type parameters during type checking
   - ✅ Constraint-based solver successfully handles functions like `const : {A : Type} -> {B : Type} -> A -> B -> A`
   - ✅ Type checking correctly infers `const Z True` as `const {Nat} {Bool} Z True`
   - ✅ Direct evaluation now works correctly with inferred implicit arguments
   - ✅ The evaluator automatically elaborates terms during evaluation

2. **De Bruijn Indices** ✅ RESOLVED
   - ✅ Fixed issue with non-dependent function types in closures
   - ✅ Proper context extension for all function types
   - ✅ Fixed evaluator environment construction to match type checking context

3. **Type Conversion**
   - Data type names correctly convert to constructors in expression contexts
   - Added VPi quotation support for function types

## All TODO Items Completed ✅

All planned features and improvements have been successfully implemented:

### Completed Features

1. **Implicit Type Inference** ✅ 
   - Constraint-based inference for multiple implicit parameters
   - Improved error messages for inference failures

2. **Higher-Rank Polymorphism** ✅
   - Functions can take and return polymorphic functions
   - Alpha-equivalence and polymorphic subsumption

3. **Type Classes/Interfaces** ✅
   - Full type class system with automatic instance resolution
   - Support for superclasses and multiple constraints

4. **Totality Checking** ✅
   - Termination, coverage, and positivity checking
   - Command-line options for fine-grained control

5. **Optimizations** ✅
   - Eta-reduction, dead code elimination, and inlining
   - Configurable optimization framework

6. **Enhanced Error Messages** ✅
   - Source locations, visual context, and smart suggestions
   - Type derivation traces in verbose mode

### Medium Priority

2. **Advanced Implicit Type Inference** ✅ FULLY COMPLETE
   - ✅ Implemented constraint-based inference for multiple parameters
   - ✅ Support inference from multiple arguments
   - ✅ Created comprehensive constraint solver with type variables
   - ✅ Evaluator now performs elaboration on-the-fly for proper evaluation
   - ✅ Better error messages for inference failures

3. **Higher-Rank Polymorphism** ✅ COMPLETE
   - ✅ Support functions taking polymorphic functions as arguments
   - ✅ Alpha-equivalence for polymorphic types
   - ✅ Polymorphic subsumption (using polymorphic functions at less polymorphic types)
   - ✅ Example that now works:
   ```ai-lang
   apply_poly : ({A : Type} -> A -> A) -> Nat -> Nat
   apply_poly f n = f {Nat} n
   ```

### Low Priority

4. **Type Classes / Interfaces** ✅ COMPLETE
   - ✅ Designed and implemented a trait/interface system
   - ✅ Automatic instance resolution during type checking
   - ✅ Basic type class declarations with methods
   - ✅ Instance definitions for specific types
   - ✅ Type class constraints in function signatures
   - ✅ Support for superclasses and multiple constraints
   - ⚠️  Coherence checking (overlapping instances) not yet implemented

5. **Totality Checking** ✅ COMPLETE
   - ✅ Termination checking for recursive functions using structural recursion
   - ✅ Coverage checking for exhaustive pattern matching
   - ✅ Positivity checking for well-founded data types
   - ✅ Command-line options to control checking (--totality, --no-termination, etc.)
   - ✅ Clear error messages for totality violations

6. **Optimization** ✅ COMPLETE
   - ✅ Implemented eta-reduction - converts `\x -> f x` to `f` when x doesn't appear free
   - ✅ Dead code elimination - removes unused functions and let bindings
   - ✅ Inlining of simple functions - inlines small, non-recursive functions
   - ✅ Optimization framework with configurable passes
   - ✅ Command-line options to control optimizations

7. **Better Error Messages** ✅ COMPLETE
   - ✅ All errors now include source locations (file, line, column)
   - ✅ Visual error display with source code context
   - ✅ Smart suggestions for common mistakes (typos, missing imports)
   - ✅ Type derivation traces in verbose mode (-v flag)
   - ✅ Color-coded error output with clear indicators
   - ✅ Contextual hints based on error type

## Testing Strategy

When implementing new features:
1. Start with minimal test cases in `examples/`
2. Add unit tests for individual components
3. Create integration tests for full programs
4. Document limitations in example files

## Code Style

- Use type annotations for all function signatures
- Prefer immutable data structures
- Use descriptive variable names
- Add docstrings for public APIs
- Keep functions focused and small