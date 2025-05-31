# ai-lang Syntax Specification

## Basic Syntax

### Comments
```ai-lang
-- Single line comment
{- Multi-line
   comment -}
```

### Identifiers
- Start with letter or underscore
- Contain letters, digits, underscores
- Case sensitive

### Literals
```ai-lang
42          -- Natural number
-42         -- Integer
true        -- Boolean
false       -- Boolean  
"hello"     -- String
```

## Type System

### Base Types
```ai-lang
Type        -- Type of types (universe)
Nat         -- Natural numbers
Int         -- Integers
Bool        -- Booleans
String      -- Strings
```

### Function Types
```ai-lang
-- Simple function type
Int -> Bool

-- Dependent function type (Pi type)
(x : Nat) -> Vec x Int

-- Implicit arguments
{a : Type} -> a -> a
```

### Product Types
```ai-lang
-- Simple pairs
(Int, Bool)

-- Dependent pairs (Sigma type)
(x : Nat, Vec x Int)

-- Records
{ name : String, age : Nat }
```

## Expressions

### Variables and Application
```ai-lang
x           -- Variable
f x         -- Application
f x y       -- Multiple application
```

### Lambda Abstraction
```ai-lang
\x -> x + 1                     -- Simple lambda
\(x : Int) -> x + 1            -- With type annotation
\{a : Type} -> \(x : a) -> x   -- Implicit parameter
```

### Let Bindings
```ai-lang
let x = 42 in x + 1
let f = \x -> x + 1 in f 5
```

### Pattern Matching
```ai-lang
case n of
  0 -> "zero"
  S m -> "successor"

case p of
  (x, y) -> x + y
```

## Declarations

### Type Definitions
```ai-lang
-- Type alias
type Predicate a = a -> Bool

-- Inductive types
data Nat : Type where
  Z : Nat
  S : Nat -> Nat

data List : Type -> Type where
  Nil : {a : Type} -> List a
  Cons : {a : Type} -> a -> List a -> List a

-- Indexed types (GADTs)
data Vec : Nat -> Type -> Type where
  VNil : {a : Type} -> Vec Z a
  VCons : {n : Nat} -> {a : Type} -> a -> Vec n a -> Vec (S n) a
```

### Function Definitions
```ai-lang
-- Simple function
add : Nat -> Nat -> Nat
add Z y = y
add (S x) y = S (add x y)

-- With implicit arguments
id : {a : Type} -> a -> a
id x = x

-- Pattern matching on dependent types
head : {n : Nat} -> {a : Type} -> Vec (S n) a -> a
head (VCons x xs) = x
```

## Module System

```ai-lang
module List where
  -- Definitions...

import List
import List (map, filter)
import List as L
```