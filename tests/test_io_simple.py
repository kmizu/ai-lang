"""Simple tests for IO functionality."""

import pytest
from ai_lang.parser import parse
from ai_lang.typechecker import type_check_module
from ai_lang.core import VIO, VUnit


class TestIOBasic:
    """Basic IO tests."""
    
    def test_io_hello_world_typechecks(self):
        """Test that Hello World program type checks."""
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
    
    def test_io_bind_typechecks(self):
        """Test that bind composition type checks."""
        source = """
        echo : IO Unit
        echo = bind {String} {Unit} getLine putStrLn
        """
        module = parse(source)
        checker = type_check_module(module, return_checker=True)
        
        # Should type check
        assert checker is not None
        
        # echo should have type IO Unit
        echo_type = checker.global_types.get("echo")
        assert isinstance(echo_type, VIO)
        assert echo_type.result_type == VUnit()
    
    def test_io_pure_typechecks(self):
        """Test that pure type checks."""
        source = """
        liftedString : IO String
        liftedString = pure {String} "Hello"
        
        liftedUnit : IO Unit  
        liftedUnit = pure {Unit} unit
        """
        module = parse(source)
        checker = type_check_module(module, return_checker=True)
        
        # Should type check
        assert checker is not None
    
    def test_io_seq_helper(self):
        """Test sequencing helper function."""
        source = """
        -- Sequence two IO actions, discarding first result
        seq : {A : Type} -> {B : Type} -> IO A -> IO B -> IO B
        seq {A} {B} first second = bind {A} {B} first (\\_ -> second)
        
        -- Use it
        twoLines : IO Unit
        twoLines = seq {Unit} {Unit} 
            (putStrLn "First") 
            (putStrLn "Second")
        """
        module = parse(source)
        checker = type_check_module(module, return_checker=True)
        
        # Should type check
        assert checker is not None
        
        # Check types
        seq_type = checker.global_types.get("seq")
        assert seq_type is not None  # Should be a Pi type
        
        two_lines_type = checker.global_types.get("twoLines") 
        assert isinstance(two_lines_type, VIO)
        assert two_lines_type.result_type == VUnit()