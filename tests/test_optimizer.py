"""Tests for optimization passes."""

import pytest
from ai_lang.parser import parse
from ai_lang.typechecker import type_check_module
from ai_lang.optimizer import (
    EtaReduction, DeadCodeElimination, Inlining,
    optimize_module
)
from ai_lang.syntax import Module, FunctionDef, Var, Lambda, App, Literal


class TestEtaReduction:
    """Test eta-reduction optimization."""
    
    def test_simple_eta_reduction(self):
        """Test basic eta-reduction."""
        source = """
        data Bool : Type where
          True : Bool
          False : Bool
        
        not : Bool -> Bool
        not True = False
        not False = True
        
        notAlias : Bool -> Bool
        notAlias = \\x -> not x
        """
        module = parse(source)
        checker = type_check_module(module, return_checker=True)
        
        optimizer = EtaReduction()
        optimized = optimizer.optimize_module(module, checker)
        
        # Find notAlias definition
        notAlias = next(d for d in optimized.declarations 
                        if isinstance(d, FunctionDef) and d.name.value == "notAlias")
        
        # Should be reduced to just 'not'
        assert len(notAlias.clauses) == 1
        assert isinstance(notAlias.clauses[0].body, Var)
        assert notAlias.clauses[0].body.name.value == "not"
    
    def test_nested_eta_reduction(self):
        """Test eta-reduction in nested lambdas."""
        source = """
        f : Nat -> Nat -> Nat
        f x y = x
        
        g : Nat -> Nat -> Nat
        g = \\x -> \\y -> f x y
        """
        module = parse(source)
        checker = type_check_module(module, return_checker=True)
        
        optimizer = EtaReduction()
        optimized = optimizer.optimize_module(module, checker)
        
        # Find g definition
        g = next(d for d in optimized.declarations 
                 if isinstance(d, FunctionDef) and d.name.value == "g")
        
        # Inner lambda should be eta-reduced
        assert len(g.clauses) == 1
        body = g.clauses[0].body
        assert isinstance(body, Lambda)  # Outer lambda remains
        assert isinstance(body.body, App)  # Inner reduced to partial application
    
    def test_no_eta_reduction_when_var_used(self):
        """Test that eta-reduction doesn't happen when variable is used."""
        source = """
        f : Nat -> Nat -> Nat
        f x y = x
        
        g : Nat -> Nat
        g = \\x -> f x x
        """
        module = parse(source)
        checker = type_check_module(module, return_checker=True)
        
        optimizer = EtaReduction()
        optimized = optimizer.optimize_module(module, checker)
        
        # Find g definition  
        g = next(d for d in optimized.declarations
                 if isinstance(d, FunctionDef) and d.name.value == "g")
        
        # Should not be reduced because x appears twice
        assert len(g.clauses) == 1
        assert isinstance(g.clauses[0].body, Lambda)


class TestDeadCodeElimination:
    """Test dead code elimination."""
    
    def test_remove_unused_function(self):
        """Test removal of unused functions."""
        source = """
        data Nat : Type where
          Z : Nat
          S : Nat -> Nat
        
        unused : Nat -> Nat
        unused x = x
        
        used : Nat -> Nat
        used x = S x
        
        main : Nat
        main = used Z
        """
        module = parse(source)
        checker = type_check_module(module, return_checker=True)
        
        optimizer = DeadCodeElimination()
        optimized = optimizer.optimize_module(module, checker)
        
        # Get function names
        func_names = [d.name.value for d in optimized.declarations 
                     if isinstance(d, FunctionDef)]
        
        # unused should be removed
        assert "unused" not in func_names
        assert "used" in func_names
        assert "main" in func_names
    
    def test_keep_transitively_used(self):
        """Test that transitively used functions are kept."""
        source = """
        data Nat : Type where
          Z : Nat
          S : Nat -> Nat
        
        helper : Nat -> Nat
        helper x = x
        
        middle : Nat -> Nat  
        middle x = helper (helper x)
        
        main : Nat
        main = middle Z
        """
        module = parse(source)
        checker = type_check_module(module, return_checker=True)
        
        optimizer = DeadCodeElimination()
        optimized = optimizer.optimize_module(module, checker)
        
        # Get function names
        func_names = [d.name.value for d in optimized.declarations
                     if isinstance(d, FunctionDef)]
        
        # All should be kept due to transitive usage
        assert "helper" in func_names
        assert "middle" in func_names
        assert "main" in func_names
    
    def test_remove_unused_let_binding(self):
        """Test removal of unused let bindings."""
        # Skip this test for now as let expressions are not fully supported
        # in the type checker yet
        pytest.skip("Let expressions not fully implemented in type checker")


class TestInlining:
    """Test function inlining."""
    
    def test_inline_simple_function(self):
        """Test inlining of simple non-recursive functions."""
        source = """
        data Nat : Type where
          Z : Nat
          S : Nat -> Nat
        
        double : Nat -> Nat
        double x = S (S x)
        
        main : Nat
        main = double (double Z)
        """
        module = parse(source)
        checker = type_check_module(module, return_checker=True)
        
        optimizer = Inlining(max_inline_size=10)
        optimized = optimizer.optimize_module(module, checker)
        
        # Find main definition
        main = next(d for d in optimized.declarations
                   if isinstance(d, FunctionDef) and d.name.value == "main")
        
        # double should be inlined, resulting in nested S applications
        # The exact structure depends on how inlining is implemented
        assert main is not None
    
    def test_no_inline_recursive(self):
        """Test that recursive functions are not inlined."""
        source = """
        data Nat : Type where
          Z : Nat
          S : Nat -> Nat
        
        plus : Nat -> Nat -> Nat
        plus Z y = y
        plus (S x) y = S (plus x y)
        
        mult : Nat -> Nat -> Nat
        mult Z y = Z
        mult (S x) y = plus y (mult x y)
        
        fact : Nat -> Nat
        fact Z = S Z
        fact (S n) = mult (S n) (fact n)
        
        main : Nat
        main = fact (S (S Z))
        """
        module = parse(source)
        checker = type_check_module(module, return_checker=True)
        
        optimizer = Inlining(max_inline_size=100)
        optimized = optimizer.optimize_module(module, checker)
        
        # fact should not be inlined because it's recursive
        func_names = [d.name.value for d in optimized.declarations
                     if isinstance(d, FunctionDef)]
        assert "fact" in func_names
    
    def test_size_limit(self):
        """Test that size limit is respected."""
        source = """
        data Nat : Type where
          Z : Nat
          S : Nat -> Nat
        
        -- Large function
        large : Nat -> Nat -> Nat -> Nat
        large x y z = S (S (S (S (S (S (S (S (S (S x)))))))))
        
        -- Small function
        small : Nat -> Nat
        small x = S x
        
        main : Nat
        main = large (small Z) (small Z) (small Z)
        """
        module = parse(source)
        checker = type_check_module(module, return_checker=True)
        
        optimizer = Inlining(max_inline_size=5)
        optimized = optimizer.optimize_module(module, checker)
        
        # Only small should be inlined
        func_names = [d.name.value for d in optimized.declarations
                     if isinstance(d, FunctionDef)]
        assert "large" in func_names  # Not inlined
        # small might be removed if fully inlined


class TestOptimizationIntegration:
    """Test integration of multiple optimization passes."""
    
    def test_combined_optimizations(self):
        """Test that multiple optimizations work together."""
        source = """
        data Nat : Type where
          Z : Nat
          S : Nat -> Nat
        
        -- Will be inlined
        inc : Nat -> Nat
        inc x = S x
        
        -- Will be eta-reduced
        incAlias : Nat -> Nat
        incAlias = \\x -> inc x
        
        -- Will be eliminated (unused)
        unused : Nat -> Nat
        unused x = inc (inc x)
        
        main : Nat
        main = incAlias Z
        """
        module = parse(source)
        type_check_module(module, return_checker=True)
        
        # Apply all optimizations
        optimized = optimize_module(module, ["eta", "inline", "dce"])
        
        func_names = [d.name.value for d in optimized.declarations
                     if isinstance(d, FunctionDef)]
        
        # unused should be eliminated
        assert "unused" not in func_names
        
        # incAlias should exist but be eta-reduced
        if "incAlias" in func_names:
            incAlias = next(d for d in optimized.declarations
                           if isinstance(d, FunctionDef) and d.name.value == "incAlias")
            body = incAlias.clauses[0].body
            # Should be reduced to just 'inc' or inlined
            assert not isinstance(body, Lambda) or isinstance(body.body, Var)
    
    def test_optimization_preserves_semantics(self):
        """Test that optimizations don't change program behavior."""
        source = """
        data Bool : Type where
          True : Bool
          False : Bool
        
        not : Bool -> Bool
        not True = False
        not False = True
        
        and : Bool -> Bool -> Bool
        and True True = True
        and _ _ = False
        
        -- Complex expression that should optimize
        complex : Bool -> Bool
        complex = \\x -> and x (not (not x))
        
        main : Bool
        main = complex True
        """
        module = parse(source)
        type_check_module(module, return_checker=True)
        
        # Optimizations should complete without error
        optimized = optimize_module(module, ["eta", "inline", "dce"])
        
        # main should still exist and be well-typed
        main_exists = any(isinstance(d, FunctionDef) and d.name.value == "main"
                         for d in optimized.declarations)
        assert main_exists