# Side Effect Handling Design for ai-lang

## Overview

This document describes the design of a side-effect handling system for ai-lang that integrates cleanly with its dependent type system. The approach uses an IO monad similar to Haskell but adapted for dependent types.

## Design Principles

1. **Type Safety**: All side effects must be tracked in the type system
2. **Purity**: Pure functions remain pure - no hidden side effects
3. **Composability**: IO actions can be composed using monadic operations
4. **Dependent Types**: IO types can depend on values (e.g., reading n lines)
5. **Simplicity**: Start with basic IO, extensible to other effects later

## The IO Type

### Basic Definition

```ai-lang
-- IO is a built-in type constructor
-- IO A represents a computation that may perform I/O and returns a value of type A
IO : Type -> Type

-- Basic IO primitives (built-in)
pure : {A : Type} -> A -> IO A
bind : {A : Type} -> {B : Type} -> IO A -> (A -> IO B) -> IO B

-- Convenient operators
(>>=) : {A : Type} -> {B : Type} -> IO A -> (A -> IO B) -> IO B
(>>=) = bind

(>>) : {A : Type} -> {B : Type} -> IO A -> IO B -> IO B
m >> n = m >>= \_ -> n

-- Built-in IO operations
print : String -> IO Unit
putStr : String -> IO Unit
putStrLn : String -> IO Unit
getLine : IO String
readFile : String -> IO String
writeFile : String -> String -> IO Unit

-- Error handling
data IOError : Type where
  FileNotFound : String -> IOError
  PermissionDenied : String -> IOError
  IOError : String -> IOError

catchIO : {A : Type} -> IO A -> (IOError -> IO A) -> IO A
throwIO : {A : Type} -> IOError -> IO A
```

## Dependent IO Types

The IO type can be used with dependent types for more precise specifications:

```ai-lang
-- Read exactly n lines from input
readLines : (n : Nat) -> IO (Vec n String)

-- Write a list of strings, return how many were written
writeLines : List String -> IO Nat

-- Read until a condition is met
readUntil : (String -> Bool) -> IO (List String)
```

## Monadic Operations

### do-notation (syntactic sugar)

```ai-lang
-- Desugaring of do-notation
-- do { x <- m; e } => m >>= \x -> e
-- do { m; e } => m >> e
-- do { let x = v; e } => let x = v in do { e }
-- do { e } => e

example : IO Unit
example = do
  name <- getLine
  putStrLn ("Hello, " ++ name ++ "!")
```

### Lifting Pure Functions

```ai-lang
-- Lift a pure function to work on IO values
liftIO : {A : Type} -> {B : Type} -> (A -> B) -> IO A -> IO B
liftIO f m = m >>= \x -> pure (f x)

-- Lift a binary function
liftIO2 : {A B C : Type} -> (A -> B -> C) -> IO A -> IO B -> IO C
liftIO2 f ma mb = do
  a <- ma
  b <- mb
  pure (f a b)
```

## Implementation Strategy

### 1. Core Type (core.py)

Add a new value type for IO:

```python
@dataclass(frozen=True)
class VIO(Value):
    """IO monad value."""
    result_type: Value
    
    def quote(self, level: int = 0) -> Term:
        return TIO(self.result_type.quote(level))

@dataclass(frozen=True)
class VIOAction(Value):
    """Runtime IO action."""
    action: IOAction
    
    def quote(self, level: int = 0) -> Term:
        raise ValueError("Cannot quote IO action")
```

### 2. Runtime Actions

Define IO actions that can be executed:

```python
@dataclass
class IOAction(ABC):
    """Abstract IO action."""
    @abstractmethod
    def execute(self) -> Tuple[Value, List[IOEffect]]:
        pass

@dataclass
class IOPure(IOAction):
    value: Value
    
    def execute(self) -> Tuple[Value, List[IOEffect]]:
        return (self.value, [])

@dataclass
class IOBind(IOAction):
    first: IOAction
    cont: Callable[[Value], IOAction]
    
    def execute(self) -> Tuple[Value, List[IOEffect]]:
        val1, effects1 = self.first.execute()
        val2, effects2 = self.cont(val1).execute()
        return (val2, effects1 + effects2)

@dataclass
class IOPrint(IOAction):
    message: str
    
    def execute(self) -> Tuple[Value, List[IOEffect]]:
        return (VUnit(), [PrintEffect(self.message)])
```

### 3. Effect Tracking

Track effects separately from evaluation for testing and debugging:

```python
@dataclass
class IOEffect:
    """Base class for IO effects."""
    pass

@dataclass
class PrintEffect(IOEffect):
    message: str

@dataclass
class ReadEffect(IOEffect):
    prompt: Optional[str]
    result: str
```

### 4. Main Function

Programs with IO need a main function of type `IO Unit`:

```ai-lang
-- Program entry point
main : IO Unit
main = do
  putStrLn "What's your name?"
  name <- getLine
  putStrLn ("Hello, " ++ name ++ "!")
```

## Example Programs

### Hello World

```ai-lang
main : IO Unit
main = putStrLn "Hello, World!"
```

### Interactive Program

```ai-lang
-- Number guessing game
guessNumber : Nat -> IO Unit
guessNumber secret = do
  putStr "Guess a number: "
  input <- getLine
  case parseNat input of
    Nothing -> do
      putStrLn "Invalid number, try again"
      guessNumber secret
    Just n ->
      if n == secret then
        putStrLn "Correct!"
      else if n < secret then do
        putStrLn "Too low!"
        guessNumber secret
      else do
        putStrLn "Too high!"
        guessNumber secret

main : IO Unit
main = do
  putStrLn "I'm thinking of a number between 1 and 100"
  guessNumber 42
```

### File I/O

```ai-lang
-- Copy file contents
copyFile : String -> String -> IO Unit
copyFile src dst = do
  contents <- readFile src
  writeFile dst contents
  putStrLn ("Copied " ++ src ++ " to " ++ dst)

-- Process lines of a file
processFile : String -> IO Unit
processFile filename = do
  contents <- readFile filename
  let lines = splitLines contents
  let count = length lines
  putStrLn ("File has " ++ show count ++ " lines")
```

## Future Extensions

1. **Effect Polymorphism**: Generalize to arbitrary effect types
2. **Effect Handlers**: Algebraic effects and handlers
3. **Resource Management**: Bracket-style resource handling
4. **Concurrency**: Async IO operations
5. **STM**: Software transactional memory

## Integration with Existing Features

### Type Classes

```ai-lang
-- Monad type class (once we have kind polymorphism)
class Monad (m : Type -> Type) where
  return : {A : Type} -> A -> m A
  (>>=) : {A B : Type} -> m A -> (A -> m B) -> m B

instance Monad IO where
  return = pure
  (>>=) = bind
```

### Dependent Types

```ai-lang
-- Read a file and prove properties about its contents
readValidated : (filename : String) -> 
                (validate : String -> Bool) ->
                IO (Σ String (\s -> validate s == True))
```

## Testing Strategy

1. **Pure Testing**: Test pure functions normally
2. **IO Testing**: Mock IO operations or use test harness
3. **Effect Verification**: Check that effects match expectations
4. **Property Testing**: QuickCheck-style testing for IO properties