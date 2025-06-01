# Type Class Design for ai-lang

## Syntax Design

### Type Class Declaration

```ai-lang
class ClassName TypeParam where
  method1 : Type1
  method2 : Type2
  ...
```

Example:
```ai-lang
class Eq A where
  eq : A -> A -> Bool
  neq : A -> A -> Bool
```

### Instance Declaration

```ai-lang
instance ClassName Type where
  method1 = implementation1
  method2 = implementation2
  ...
```

Example:
```ai-lang
instance Eq Nat where
  eq = natEq
  neq = \x y -> not (natEq x y)
```

### Type Class Constraints

Functions can have type class constraints using the `=>` syntax:

```ai-lang
functionName : {A : Type} -> ClassName A => Type
```

Example:
```ai-lang
equal : {A : Type} -> Eq A => A -> A -> Bool
equal x y = eq x y
```

### Multiple Constraints

Multiple constraints are separated by commas:

```ai-lang
compare : {A : Type} -> Eq A, Ord A => A -> A -> Ordering
```

### Superclasses

Type classes can have superclass constraints:

```ai-lang
class Eq A => Ord A where
  lt : A -> A -> Bool
  lte : A -> A -> Bool
```

## Implementation Notes

1. Type classes will be desugared into records of functions
2. Instances will be treated as implicit arguments that are resolved by the type checker
3. Instance resolution will use a simple search algorithm with coherence checking
4. We'll start with single-parameter type classes and potentially extend to multi-parameter later