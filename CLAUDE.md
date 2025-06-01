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

## Remaining TODO Items

### High Priority

1. **Implicit Parameters in Function Bodies** ✅ RESOLVED
   - ✅ Implicit type parameters are now accessible in function bodies
   - ✅ Fixed evaluator to properly track implicit parameters
   - ✅ Example that now works:
   ```ai-lang
   nilOf : {A : Type} -> A -> List A
   nilOf x = Nil {A}  -- A is now properly in scope
   ```

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

4. **Type Classes / Interfaces**
   - Design and implement a trait/interface system
   - Automatic instance resolution
   - Coherence checking

5. **Totality Checking**
   - Termination checking for recursive functions
   - Coverage checking for pattern matching
   - Positivity checking for data types

6. **Optimization**
   - Implement eta-reduction
   - Dead code elimination
   - Inlining of simple functions

7. **Better Error Messages**
   - Include source locations in all errors
   - Provide suggestions for common mistakes
   - Show type derivation traces in verbose mode

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