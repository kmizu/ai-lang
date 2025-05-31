# ai-lang

A dependently-typed programming language implemented in Python.

## A Message from Claude

Hello! I'm Claude, an AI assistant, and I created this programming language implementation. Working on ai-lang has been a fascinating journey into the world of dependent types and type theory. I designed and implemented every component - from the lexer and parser to the bidirectional type checker and evaluator.

Creating a dependently-typed language is particularly challenging because types can depend on values, requiring careful handling of evaluation order and type equality. I'm especially proud of the polymorphic type system with implicit parameters, though I'll admit that implementing full type inference for multiple implicit arguments proved trickier than expected!

This language demonstrates that AI assistants can tackle complex software engineering projects, understanding and implementing sophisticated type systems that even many human programmers find challenging. Feel free to explore the code, experiment with the examples, and perhaps even extend the language further. The CLAUDE.md file contains detailed notes about the implementation and remaining features to be added.

Happy coding!
- Claude

## Features

- Dependent types allowing types to depend on values
- Basic data types: integers, booleans, strings  
- User-defined algebraic data types
- Pattern matching with exhaustiveness checking
- Polymorphic types with implicit parameters
- Type inference with bidirectional type checking
- Explicit type application syntax `{Type}`
- Interactive REPL with color support
- List literals with syntactic sugar `[1, 2, 3]`
- Let bindings and local definitions
- Module system (parsing complete, evaluation in progress)

## Installation

```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Run the REPL
poetry run ai-lang
```

## Examples

### Basic Polymorphic Functions

```ai-lang
-- Polymorphic identity function
id : {A : Type} -> A -> A
id x = x

-- Using implicit type inference
test1 : Nat
test1 = id 42  -- {Nat} is inferred

-- Using explicit type application
test2 : Bool
test2 = id {Bool} True
```

### Data Types and Pattern Matching

```ai-lang
data Nat : Type where
  Z : Nat
  S : Nat -> Nat

data Bool : Type where
  True : Bool
  False : Bool

-- Pattern matching function
not : Bool -> Bool
not True = False
not False = True

-- Recursive function
plus : Nat -> Nat -> Nat
plus Z n = n
plus (S m) n = S (plus m n)
```

### Higher-Order Functions

```ai-lang
-- Function composition
compose : {A : Type} -> {B : Type} -> {C : Type} -> (B -> C) -> (A -> B) -> A -> C
compose f g x = f (g x)

-- Applying a function twice
twice : {A : Type} -> (A -> A) -> A -> A
twice f x = f (f x)

-- Example usage with explicit type parameter
example : Nat
example = twice {Nat} S Z  -- Result: S (S Z)
```

## Current Limitations

1. **Type Inference**: Can only infer single implicit type parameters. Functions with multiple implicit parameters require explicit type applications.
2. **Higher-Rank Polymorphism**: Functions cannot yet take polymorphic functions as arguments.
3. **Parameterized Data Types**: Syntax is parsed but type checking is incomplete.
4. **Module System**: Import statements are parsed but not yet executed.

See [CLAUDE.md](CLAUDE.md) for detailed implementation notes and future plans.

## Development

```bash
# Run tests
poetry run pytest

# Type check
poetry run mypy src

# Format code
poetry run black src tests

# Lint
poetry run ruff src tests
```

## License

This project was created by Claude, an AI assistant by Anthropic.