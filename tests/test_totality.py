"""Tests for totality checking."""

import pytest
from ai_lang.syntax import *
from ai_lang.core import *
from ai_lang.typechecker import TypeChecker, type_check_module
from ai_lang.totality import (
    TotalityChecker, TotalityOptions,
    TerminationError, CoverageError, PositivityError,
    TerminationChecker, CoverageChecker, PositivityChecker
)
from ai_lang.parser import parse


class TestTerminationChecker:
    """Test termination checking."""
    
    def test_non_recursive_function(self):
        """Non-recursive functions should always pass."""
        source = """
        add : Nat -> Nat -> Nat
        add x y = x
        """
        module = parse(source)
        checker = type_check_module(module)
        term_checker = TerminationChecker(checker)
        
        # Should not raise
        term_checker.check_module(module)
    
    def test_structural_recursion_nat(self):
        """Structural recursion on Nat should pass."""
        source = """
        data Nat : Type where
          Z : Nat
          S : Nat -> Nat
        
        plus : Nat -> Nat -> Nat
        plus Z y = y
        plus (S x) y = S (plus x y)
        """
        module = parse(source)
        checker = type_check_module(module)
        term_checker = TerminationChecker(checker)
        
        # Should not raise
        term_checker.check_module(module)
    
    def test_non_terminating_recursion(self):
        """Non-terminating recursion should fail."""
        source = """
        data Nat : Type where
          Z : Nat
          S : Nat -> Nat
        
        bad : Nat -> Nat
        bad n = bad (S n)
        """
        module = parse(source)
        checker = type_check_module(module)
        term_checker = TerminationChecker(checker)
        
        with pytest.raises(TerminationError) as exc_info:
            term_checker.check_module(module)
        assert "bad" in str(exc_info.value)
        assert "may not terminate" in str(exc_info.value)
    
    def test_mutual_recursion(self):
        """Mutual recursion with decreasing argument should pass."""
        source = """
        data Nat : Type where
          Z : Nat
          S : Nat -> Nat
        
        data Bool : Type where
          True : Bool
          False : Bool
        
        even : Nat -> Bool
        odd : Nat -> Bool
        
        even Z = True
        even (S n) = odd n
        
        odd Z = False
        odd (S n) = even n
        """
        module = parse(source)
        checker = type_check_module(module)
        term_checker = TerminationChecker(checker)
        
        # Should not raise
        term_checker.check_module(module)


class TestCoverageChecker:
    """Test coverage checking."""
    
    def test_exhaustive_bool(self):
        """Exhaustive pattern matching on Bool should pass."""
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
        cov_checker = CoverageChecker(checker)
        
        # Should not raise
        for decl in module.declarations:
            if isinstance(decl, FunctionDef):
                cov_checker.check_function(decl)
    
    def test_non_exhaustive_bool(self):
        """Non-exhaustive pattern matching on Bool should fail."""
        source = """
        data Bool : Type where
          True : Bool
          False : Bool
        
        bad : Bool -> Bool
        bad True = False
        """
        module = parse(source)
        checker = type_check_module(module)
        cov_checker = CoverageChecker(checker)
        
        with pytest.raises(CoverageError) as exc_info:
            for decl in module.declarations:
                if isinstance(decl, FunctionDef) and decl.name.value == "bad":
                    cov_checker.check_function(decl)
        
        assert "Non-exhaustive" in str(exc_info.value)
        assert "False" in str(exc_info.value)
    
    def test_exhaustive_nat(self):
        """Exhaustive pattern matching on Nat should pass."""
        source = """
        data Nat : Type where
          Z : Nat
          S : Nat -> Nat
        
        data Bool : Type where
          True : Bool
          False : Bool
        
        isZero : Nat -> Bool
        isZero Z = True
        isZero (S n) = False
        """
        module = parse(source)
        checker = type_check_module(module)
        cov_checker = CoverageChecker(checker)
        
        # Should not raise
        for decl in module.declarations:
            if isinstance(decl, FunctionDef):
                cov_checker.check_function(decl)
    
    def test_wildcard_pattern(self):
        """Wildcard patterns should make matching exhaustive."""
        source = """
        data Nat : Type where
          Z : Nat
          S : Nat -> Nat
        
        data Bool : Type where
          True : Bool
          False : Bool
        
        isZero : Nat -> Bool
        isZero Z = True
        isZero _ = False
        """
        module = parse(source)
        checker = type_check_module(module)
        cov_checker = CoverageChecker(checker)
        
        # Should not raise
        for decl in module.declarations:
            if isinstance(decl, FunctionDef):
                cov_checker.check_function(decl)
    
    def test_case_expression_exhaustive(self):
        """Exhaustive case expressions should pass."""
        source = """
        data Bool : Type where
          True : Bool
          False : Bool
        
        test : Bool -> Bool
        test True = False
        test False = True
        """
        module = parse(source)
        checker = type_check_module(module)
        
        # We need to test case expressions during type checking
        # This is more of an integration test
        # Should not raise
        type_check_module(module)
    
    def test_redundant_pattern(self):
        """Redundant patterns should be detected."""
        source = """
        data Bool : Type where
          True : Bool
          False : Bool
        
        bad : Bool -> Bool
        bad True = False
        bad False = True
        bad True = True
        """
        module = parse(source)
        checker = type_check_module(module)
        cov_checker = CoverageChecker(checker)
        
        # Current implementation doesn't check redundancy strictly
        # This test documents expected behavior
        for decl in module.declarations:
            if isinstance(decl, FunctionDef) and decl.name.value == "bad":
                # Should detect redundancy in future
                cov_checker.check_function(decl)


class TestPositivityChecker:
    """Test positivity checking."""
    
    def test_positive_data_type(self):
        """Positive data types should pass."""
        source = """
        data List A : Type where
          Nil : List A
          Cons : A -> List A -> List A
        """
        module = parse(source)
        checker = type_check_module(module)
        pos_checker = PositivityChecker(checker)
        
        # Should not raise
        for decl in module.declarations:
            if isinstance(decl, DataDecl):
                pos_checker.check_data_type(decl)
    
    def test_negative_occurrence(self):
        """Negative occurrences should fail."""
        source = """
        data Bad : Type where
          MkBad : (Bad -> Nat) -> Bad
        """
        module = parse(source)
        checker = type_check_module(module)
        pos_checker = PositivityChecker(checker)
        
        with pytest.raises(PositivityError) as exc_info:
            for decl in module.declarations:
                if isinstance(decl, DataDecl):
                    pos_checker.check_data_type(decl)
        
        assert "Bad" in str(exc_info.value)
        assert "negatively" in str(exc_info.value)
    
    def test_nested_positive(self):
        """Nested positive occurrences should pass."""
        source = """
        data List A : Type where
          Nil : List A
          Cons : A -> List A -> List A
        
        data Tree A : Type where
          Leaf : A -> Tree A
          Node : Tree A -> Tree A -> Tree A
        
        data Forest A : Type where
          Empty : Forest A
          Trees : List (Tree A) -> Forest A
        """
        module = parse(source)
        checker = type_check_module(module)
        pos_checker = PositivityChecker(checker)
        
        # Should not raise
        for decl in module.declarations:
            if isinstance(decl, DataDecl):
                pos_checker.check_data_type(decl)


class TestTotalityIntegration:
    """Integration tests for totality checking."""
    
    def test_full_totality_check(self):
        """Test complete totality checking."""
        source = """
        data Nat : Type where
          Z : Nat
          S : Nat -> Nat
        
        data List A : Type where
          Nil : List A
          Cons : A -> List A -> List A
        
        length : {A : Type} -> List A -> Nat
        length Nil = Z
        length (Cons x xs) = S (length xs)
        
        map : {A : Type} -> {B : Type} -> (A -> B) -> List A -> List B
        map f Nil = Nil
        map f (Cons x xs) = Cons (f x) (map f xs)
        """
        module = parse(source)
        checker = type_check_module(module)
        
        totality_checker = TotalityChecker(checker)
        warnings = totality_checker.check_module(module)
        
        # Should pass without warnings
        assert warnings == []
    
    def test_totality_with_options(self):
        """Test totality checking with specific options disabled."""
        source = """
        data Bad : Type where
          MkBad : (Bad -> Nat) -> Bad
        
        partial : Nat -> Nat
        partial n = partial n
        """
        module = parse(source)
        checker = type_check_module(module)
        
        # With all checks enabled, should fail
        totality_checker = TotalityChecker(checker)
        with pytest.raises(PositivityError):
            totality_checker.check_module(module)
        
        # With positivity check disabled, should fail on termination
        options = TotalityOptions(
            check_positivity=False,
            check_termination=True,
            check_coverage=True
        )
        totality_checker = TotalityChecker(checker, options)
        with pytest.raises(TerminationError):
            totality_checker.check_module(module)
        
        # With both disabled, should pass
        options = TotalityOptions(
            check_positivity=False,
            check_termination=False,
            check_coverage=True
        )
        totality_checker = TotalityChecker(checker, options)
        warnings = totality_checker.check_module(module)
        # Should complete without raising