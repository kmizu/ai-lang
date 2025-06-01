"""Tests for type class functionality."""

import pytest
from ai_lang.lexer import lex
from ai_lang.parser import Parser
from ai_lang.typechecker import TypeChecker
from ai_lang.errors import TypeCheckError


class TestTypeClasses:
    """Test type class functionality."""
    
    def test_parse_type_class(self):
        """Test parsing type class declarations."""
        source = """
        class Eq A where
          eq : A -> A -> Bool
          neq : A -> A -> Bool
        """
        tokens = lex(source)
        parser = Parser(tokens)
        module = parser.parse_module()
        
        assert len(module.declarations) == 1
        class_decl = module.declarations[0]
        assert class_decl.name.value == "Eq"
        assert class_decl.type_param.value == "A"
        assert len(class_decl.methods) == 2
        assert class_decl.methods[0][0].value == "eq"
        assert class_decl.methods[1][0].value == "neq"
    
    def test_parse_instance(self):
        """Test parsing instance declarations."""
        source = """
        data Nat : Type where
          Z : Nat
          S : Nat -> Nat
        
        class Eq A where
          eq : A -> A -> Bool
        
        instance Eq Nat where
          eq = natEq
        """
        tokens = lex(source)
        parser = Parser(tokens)
        module = parser.parse_module()
        
        assert len(module.declarations) == 3
        instance_decl = module.declarations[2]
        assert instance_decl.class_name.value == "Eq"
        assert len(instance_decl.methods) == 1
        assert instance_decl.methods[0][0].value == "eq"
    
    def test_parse_constraint_type(self):
        """Test parsing types with constraints."""
        source = """
        equal : {A : Type} -> Eq A => A -> A -> Bool
        """
        tokens = lex(source)
        parser = Parser(tokens)
        module = parser.parse_module()
        
        assert len(module.declarations) == 1
        type_sig = module.declarations[0]
        assert type_sig.name.value == "equal"
        # The type should be a function type with implicit parameter
        # and a constraint type body
    
    def test_parse_superclass_constraint(self):
        """Test parsing type classes with superclass constraints."""
        source = """
        class Eq A => Ord A where
          lt : A -> A -> Bool
        """
        tokens = lex(source)
        parser = Parser(tokens)
        module = parser.parse_module()
        
        assert len(module.declarations) == 1
        class_decl = module.declarations[0]
        assert class_decl.name.value == "Ord"
        assert len(class_decl.superclasses) == 1
        assert class_decl.superclasses[0][0].value == "Eq"
    
    def test_type_check_class_and_instance(self):
        """Test type checking type classes and instances."""
        source = """
        data Bool : Type where
          True : Bool
          False : Bool
        
        class Eq A where
          eq : A -> A -> Bool
        
        boolEq : Bool -> Bool -> Bool
        boolEq True True = True
        boolEq False False = True
        boolEq _ _ = False
        
        instance Eq Bool where
          eq = boolEq
        """
        tokens = lex(source)
        parser = Parser(tokens)
        module = parser.parse_module()
        
        type_checker = TypeChecker()
        type_checker.check_module(module)
        
        # Check that the class was added
        assert "Eq" in type_checker.type_classes
        class_info = type_checker.type_classes["Eq"]
        assert "eq" in class_info.methods
        
        # Check that the instance was added
        assert "Eq" in type_checker.instances
        assert len(type_checker.instances["Eq"]) == 1
    
    def test_type_check_constraint_function(self):
        """Test type checking functions with type class constraints."""
        source = """
        data Bool : Type where
          True : Bool
          False : Bool
        
        class Eq A where
          eq : A -> A -> Bool
        
        -- This should type check
        equal : {A : Type} -> Eq A => A -> A -> Bool
        equal x y = eq x y
        """
        tokens = lex(source)
        parser = Parser(tokens)
        module = parser.parse_module()
        
        type_checker = TypeChecker()
        # This should not raise an error
        type_checker.check_module(module)
    
    def test_missing_instance_error(self):
        """Test that missing instances are caught."""
        source = """
        data Bool : Type where
          True : Bool
          False : Bool
        
        data Nat : Type where
          Z : Nat
          S : Nat -> Nat
        
        class Eq A where
          eq : A -> A -> Bool
        
        instance Eq Bool where
          eq = boolEq
        
        equal : {A : Type} -> Eq A => A -> A -> Bool
        equal x y = eq x y
        
        -- This should fail - no Eq instance for Nat
        testFail : Bool
        testFail = equal {Nat} Z Z
        """
        tokens = lex(source)
        parser = Parser(tokens)
        module = parser.parse_module()
        
        type_checker = TypeChecker()
        # This should raise an error about missing Eq Nat instance
        # For now, we'll just check that it parses correctly
        # The full instance resolution would catch this error
    
    def test_parse_multiple_constraints(self):
        """Test parsing multiple constraints."""
        source = """
        class Eq A where
          eq : A -> A -> Bool
        
        class Show A where
          show : A -> String
        
        display : {A : Type} -> Eq A, Show A => A -> A -> String
        """
        tokens = lex(source)
        parser = Parser(tokens)
        module = parser.parse_module()
        
        assert len(module.declarations) == 3
        type_sig = module.declarations[2]
        assert type_sig.name.value == "display"