# Testing ai-lang

This guide covers different ways to test ai-lang, from unit tests to full programs.

## Quick Start

### 1. Run Existing Tests

```bash
# Run all tests with pytest
cd /home/mizushima/repo/ai-lang
PYTHONPATH=src python3 -m pytest tests/

# Run specific test file
PYTHONPATH=src python3 -m pytest tests/test_lexer.py

# Run with coverage
PYTHONPATH=src python3 -m pytest tests/ --cov=ai_lang
```

### 2. Test with Examples

```bash
# Run example files
PYTHONPATH=src python3 src/ai_lang/cli.py examples/minimal.ai
PYTHONPATH=src python3 src/ai_lang/cli.py examples/basic.ai

# Type check only
PYTHONPATH=src python3 src/ai_lang/cli.py examples/minimal.ai --type-check-only

# Show AST
PYTHONPATH=src python3 src/ai_lang/cli.py examples/minimal.ai --ast
```

### 3. Interactive REPL Testing

```bash
# Start REPL
PYTHONPATH=src python3 src/ai_lang/cli.py

# In REPL:
ai-lang> zero : Nat
ai-lang> zero = Z
ai-lang> :type zero
ai-lang> :quit
```

## Testing Components

### Lexer Tests

Test tokenization of source code:

```python
from ai_lang.lexer import lex

# Test basic tokens
tokens = lex("let x = 42 in x")
for token in tokens:
    print(token)

# Test comments
tokens = lex("-- comment\nx")
```

### Parser Tests

Test AST generation:

```python
from ai_lang.parser import parse

# Test expression parsing
module = parse("S (S Z)")
print(module)

# Test function definition
module = parse("""
f : Nat -> Nat
f x = x
""")
```

### Type Checker Tests

Test type checking:

```python
from ai_lang.parser import parse
from ai_lang.typechecker import type_check_module

source = """
data Bool : Type where
  True : Bool
  False : Bool

not : Bool -> Bool
not True = False
not False = True
"""

module = parse(source)
checker = type_check_module(module)
print("Type checking passed!")
```

### Evaluator Tests

Test evaluation:

```python
from ai_lang.evaluator import Evaluator
from ai_lang.parser import parse
from ai_lang.typechecker import type_check_module

source = """
data Nat : Type where
  Z : Nat
  S : Nat -> Nat

two : Nat
two = S (S Z)
"""

module = parse(source)
checker = type_check_module(module)
evaluator = Evaluator(checker)
evaluator.eval_module(module)

result = evaluator.global_env['two']
print(f"two = {result}")
```

## Example Test Programs

### 1. Basic Data Types (test_datatypes.ai)

```ai-lang
-- Test basic data type definitions
data Nat : Type where
  Z : Nat
  S : Nat -> Nat

data Bool : Type where
  True : Bool
  False : Bool

-- Test constructors
zero : Nat
zero = Z

one : Nat
one = S Z

yes : Bool
yes = True

main : Nat
main = one
```

### 2. Pattern Matching (test_patterns.ai)

```ai-lang
data Nat : Type where
  Z : Nat
  S : Nat -> Nat

data Bool : Type where
  True : Bool
  False : Bool

-- Test pattern matching
isZero : Nat -> Bool
isZero Z = True
isZero (S n) = False

pred : Nat -> Nat
pred Z = Z
pred (S n) = n

-- Note: Function calls not fully supported yet
```

### 3. Let Bindings (test_let.ai)

```ai-lang
data Nat : Type where
  Z : Nat
  S : Nat -> Nat

-- Test let expressions
test : Nat
test = let x = S Z in 
       let y = S x in
       y

main : Nat
main = test
```

## Manual Testing Checklist

### REPL Features
- [ ] Start REPL: `PYTHONPATH=src python3 src/ai_lang/cli.py`
- [ ] Define data type
- [ ] Define function with type signature
- [ ] Use `:type` command
- [ ] Use `:list` command
- [ ] Use `:help` command
- [ ] Use `:quit` command

### CLI Options
- [ ] `--version` - Shows version
- [ ] `--ast` - Shows AST
- [ ] `--type-check-only` - Only type checks
- [ ] `--verbose` - Shows detailed output
- [ ] `--timing` - Shows performance metrics
- [ ] `--output file.txt` - Saves output to file

### Language Features
- [ ] Data type declarations
- [ ] Constructor applications
- [ ] Type signatures
- [ ] Function definitions
- [ ] Pattern matching (in type checker)
- [ ] Let bindings
- [ ] Lambda expressions

## Known Limitations

1. **Function Calls**: Global function references not fully implemented
   ```ai-lang
   f : Nat -> Nat
   f x = x
   
   -- This won't work yet:
   test : Nat
   test = f Z
   ```

2. **Type Parameters**: Polymorphic types need more work
   ```ai-lang
   -- This won't work yet:
   data List a : Type where
     Nil : List a
     Cons : a -> List a -> List a
   ```

3. **Case Expressions**: Not implemented in evaluator
   ```ai-lang
   -- This won't parse yet:
   test = case x of
     Z -> True
     S n -> False
   ```

## Creating New Tests

### Unit Test Template

```python
# tests/test_my_feature.py
import pytest
from ai_lang.my_module import my_function

def test_basic_functionality():
    """Test basic feature."""
    result = my_function("input")
    assert result == expected_output

def test_error_handling():
    """Test error cases."""
    with pytest.raises(MyError):
        my_function("bad input")
```

### Integration Test Template

```python
# tests/test_integration.py
def test_full_pipeline():
    """Test parsing, type checking, and evaluation."""
    source = """
    data Nat : Type where
      Z : Nat
      S : Nat -> Nat
    
    main : Nat
    main = S Z
    """
    
    module = parse(source)
    checker = type_check_module(module)
    evaluator = Evaluator(checker)
    evaluator.eval_module(module)
    
    result = evaluator.global_env['main']
    assert str(result) == "S(0)"
```

## Debugging Tips

1. **Use verbose mode**: `--verbose` flag shows evaluation trace
2. **Check AST**: `--ast` flag helps debug parsing issues
3. **Type check only**: `--type-check-only` isolates type errors
4. **REPL exploration**: Use `:type` to check inferred types
5. **Simple examples**: Start with minimal test cases

## Performance Testing

```bash
# Time a program
PYTHONPATH=src python3 src/ai_lang/cli.py big_program.ai --timing

# Profile with Python
PYTHONPATH=src python3 -m cProfile src/ai_lang/cli.py program.ai
```