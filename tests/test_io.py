"""Tests for IO side-effect handling."""

import pytest
from ai_lang.parser import parse
from ai_lang.typechecker import type_check_module
from ai_lang.evaluator import Evaluator
from ai_lang.core import (
    VIO, VUnit, VString, VIOAction, 
    IOPure, IOBind, IOPrint, IOGetLine,
    PrintEffect, ReadEffect
)


class TestIO:
    """Test IO monad functionality."""
    
    def test_io_type_checking(self):
        """Test that IO types are properly checked."""
        source = """
        main : IO Unit
        main = putStrLn "Hello, World!"
        """
        module = parse(source)
        checker = type_check_module(module, return_checker=True)
        
        # Should type check without errors
        assert checker is not None
        
        # main should have type IO Unit
        main_type = checker.global_types.get("main")
        assert isinstance(main_type, VIO)
        assert main_type.result_type == VUnit()
    
    def test_io_pure(self):
        """Test pure lifts values into IO."""
        source = """
        liftedValue : IO String
        liftedValue = pure {String} "Hello"
        """
        module = parse(source)
        checker = type_check_module(module, return_checker=True)
        
        # Should type check
        assert checker is not None
        
        # Evaluate the module
        result = eval_module(module, checker)
        io_action = result.get("liftedValue")
        
        assert isinstance(io_action, VIOAction)
        assert isinstance(io_action.action, IOPure)
        assert io_action.action.value == VString("Hello")
    
    def test_io_bind(self):
        """Test bind composition of IO actions."""
        source = """
        -- Echo input back
        echo : IO Unit  
        echo = bind {String} {Unit} getLine putStrLn
        """
        module = parse(source)
        checker = type_check_module(module, return_checker=True)
        
        # Should type check
        assert checker is not None
        
        # Check that echo has the right type
        echo_type = checker.global_types.get("echo")
        assert isinstance(echo_type, VIO)
        assert echo_type.result_type == VUnit()
    
    def test_io_effects(self):
        """Test that IO actions produce correct effects."""
        source = """
        testPrint : IO Unit
        testPrint = print "Test"
        """
        module = parse(source)
        checker = type_check_module(module, return_checker=True)
        result = eval_module(module, checker)
        
        io_action = result.get("testPrint")
        assert isinstance(io_action, VIOAction)
        
        # Execute and check effects
        from ai_lang.evaluator import Evaluator
        evaluator = Evaluator(checker.global_values)
        value, effects = evaluator.execute_io(io_action.action)
        
        assert len(effects) == 1
        assert isinstance(effects[0], PrintEffect)
        assert effects[0].text == "Test"
    
    def test_io_sequence(self):
        """Test sequencing multiple IO actions."""
        source = """
        seq : {A : Type} -> {B : Type} -> IO A -> IO B -> IO B
        seq {A} {B} first second = bind {A} {B} first (\\_ -> second)
        
        twoLines : IO Unit
        twoLines = seq {Unit} {Unit} 
            (putStrLn "First line")
            (putStrLn "Second line")
        """
        module = parse(source)
        checker = type_check_module(module, return_checker=True)
        result = eval_module(module, checker)
        
        # Should type check and evaluate
        assert "seq" in result
        assert "twoLines" in result
        
        # Check that twoLines produces two print effects
        io_action = result.get("twoLines")
        from ai_lang.evaluator import Evaluator
        evaluator = Evaluator(checker.global_values)
        value, effects = evaluator.execute_io(io_action.action)
        
        assert len(effects) == 2
        assert all(isinstance(e, PrintEffect) for e in effects)
        assert effects[0].text == "First line\n"
        assert effects[1].text == "Second line\n"