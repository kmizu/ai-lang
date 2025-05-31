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
- ✅ Module declarations and imports (parsing only)

### Known Issues and Limitations

1. **Implicit Type Inference**
   - Can only infer single implicit type parameters
   - Cannot infer multiple implicit parameters (e.g., `const 1 True` needs `const {Nat} {Bool} 1 True`)
   - Cannot infer from function arguments (e.g., `twice succ` needs `twice {Nat} succ`)

2. **De Bruijn Indices**
   - Fixed issue with non-dependent function types in closures
   - Proper context extension for all function types

3. **Type Conversion**
   - Data type names correctly convert to constructors in expression contexts
   - Added VPi quotation support for function types

## Remaining TODO Items

### High Priority

1. **Fully Implement Parameterized Data Types**
   - Currently syntax is parsed but type checking is incomplete
   - Need to handle type parameters in data constructors
   - Implement proper type parameter substitution
   - Example that should work:
   ```ai-lang
   data List (A : Type) : Type where
     Nil : List A
     Cons : A -> List A -> List A
   ```

2. **Fully Implement Module Loading and Resolution**
   - Module syntax is parsed but not executed
   - Need to implement:
     - Module dependency resolution
     - Import mechanism
     - Namespace management
     - Cyclic dependency detection
   - File system integration for multi-file projects

### Medium Priority

3. **Advanced Implicit Type Inference**
   - Implement constraint-based inference for multiple parameters
   - Support inference from multiple arguments
   - Better error messages for inference failures
   - Possible approach: collect constraints and solve

4. **Higher-Rank Polymorphism**
   - Support functions taking polymorphic functions as arguments
   - Example that should work:
   ```ai-lang
   apply_poly : ({A : Type} -> A -> A) -> Nat -> Nat
   apply_poly f n = f {Nat} n
   ```

### Low Priority

5. **Type Classes / Interfaces**
   - Design and implement a trait/interface system
   - Automatic instance resolution
   - Coherence checking

6. **Totality Checking**
   - Termination checking for recursive functions
   - Coverage checking for pattern matching
   - Positivity checking for data types

7. **Optimization**
   - Implement eta-reduction
   - Dead code elimination
   - Inlining of simple functions

8. **Better Error Messages**
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