"""Tests for the parser."""

import pytest
from ai_lang.parser import Parser, ParseError, parse
from ai_lang.lexer import lex
from ai_lang.syntax import *


def parse_expr(source: str):
    """Helper to parse a bare expression."""
    # Wrap expression in a main function
    wrapped = f"main = {source}"
    ast = parse(wrapped)
    if ast.declarations:
        func_def = ast.declarations[0]
        if isinstance(func_def, FunctionDef) and func_def.clauses:
            return func_def.clauses[0].body
    return None


def test_parse_literals():
    """Test parsing literal expressions."""
    # Integer
    expr = parse_expr("42")
    assert isinstance(expr, Literal)
    assert expr.value == 42
    
    # Boolean
    expr = parse_expr("true")
    assert isinstance(expr, Literal)
    assert expr.value is True
    
    # String
    expr = parse_expr('"hello"')
    assert isinstance(expr, Literal)
    assert expr.value == "hello"


def test_parse_variables():
    """Test parsing variable references."""
    expr = parse_expr("x")
    assert isinstance(expr, Var)
    assert expr.name.value == "x"


def test_parse_lambda():
    """Test parsing lambda expressions."""
    # Simple lambda
    expr = parse_expr("\\x -> x")
    assert isinstance(expr, Lambda)
    assert expr.param.value == "x"
    assert expr.param_type is None
    assert not expr.implicit
    assert isinstance(expr.body, Var)
    assert expr.body.name.value == "x"
    
    # Lambda with type annotation
    expr = parse_expr("\\(x : Int) -> x")
    assert isinstance(expr, Lambda)
    assert expr.param.value == "x"
    assert isinstance(expr.param_type, TypeConstructor)
    assert expr.param_type.name.value == "Int"
    
    # Lambda with implicit parameter
    expr = parse_expr("\\{a : Type} -> \\(x : a) -> x")
    assert isinstance(expr, Lambda)
    assert expr.param.value == "a"
    assert expr.implicit


def test_parse_application():
    """Test parsing function application."""
    # Simple application
    expr = parse_expr("f x")
    assert isinstance(expr, App)
    assert isinstance(expr.function, Var)
    assert expr.function.name.value == "f"
    assert isinstance(expr.argument, Var)
    assert expr.argument.name.value == "x"
    assert not expr.implicit
    
    # Multiple applications
    expr = parse_expr("f x y")
    assert isinstance(expr, App)
    assert isinstance(expr.function, App)
    
    # Implicit application
    expr = parse_expr("f {a}")
    assert isinstance(expr, App)
    assert expr.implicit


def test_parse_let():
    """Test parsing let expressions."""
    expr = parse_expr("let x = 42 in x")
    assert isinstance(expr, Let)
    assert expr.name.value == "x"
    assert expr.type_annotation is None
    assert isinstance(expr.value, Literal)
    assert expr.value.value == 42
    assert isinstance(expr.body, Var)
    
    # Let with type annotation
    expr = parse_expr("let x : Int = 42 in x")
    assert isinstance(expr, Let)
    assert isinstance(expr.type_annotation, TypeConstructor)


def test_parse_case():
    """Test parsing case expressions."""
    source = "case x of { 0 -> true; _ -> false }"
    expr = parse_expr(source)
    assert isinstance(expr, Case)
    assert isinstance(expr.scrutinee, Var)
    assert len(expr.branches) == 2
    
    # First branch
    branch = expr.branches[0]
    assert isinstance(branch.pattern, PatternLiteral)
    assert branch.pattern.value == 0
    assert isinstance(branch.body, Literal)
    assert branch.body.value is True
    
    # Second branch
    branch = expr.branches[1]
    assert isinstance(branch.pattern, PatternWildcard)
    assert isinstance(branch.body, Literal)
    assert branch.body.value is False


def test_parse_type_signature():
    """Test parsing type signatures."""
    ast = parse("id : {a : Type} -> a -> a")
    decl = ast.declarations[0]
    assert isinstance(decl, TypeSignature)
    assert decl.name.value == "id"
    
    ty = decl.type
    assert isinstance(ty, FunctionType)
    assert ty.implicit
    assert ty.param_name.value == "a"
    assert isinstance(ty.param_type, UniverseType)


def test_parse_function_definition():
    """Test parsing function definitions."""
    # Simple function
    ast = parse("f x = x")
    decl = ast.declarations[0]
    assert isinstance(decl, FunctionDef)
    assert decl.name.value == "f"
    assert len(decl.clauses) == 1
    
    clause = decl.clauses[0]
    assert len(clause.patterns) == 1
    assert isinstance(clause.patterns[0], PatternVar)
    assert isinstance(clause.body, Var)
    
    # Function with multiple clauses
    source = """add Z y = y
add (S x) y = S (add x y)"""
    ast = parse(source)
    decl = ast.declarations[0]
    assert isinstance(decl, FunctionDef)
    assert len(decl.clauses) == 2


def test_parse_data_declaration():
    """Test parsing data declarations."""
    # Simple data type
    source = """data Bool : Type where
  True : Bool
  False : Bool"""
    ast = parse(source)
    decl = ast.declarations[0]
    assert isinstance(decl, DataDecl)
    assert decl.name.value == "Bool"
    assert len(decl.constructors) == 2
    
    # Parameterized type
    source = """data List a : Type where
  Nil : List a
  Cons : a -> List a -> List a"""
    ast = parse(source)
    decl = ast.declarations[0]
    assert isinstance(decl, DataDecl)
    assert len(decl.type_params) == 1
    assert decl.type_params[0].value == "a"
    
    # Indexed type
    source = """data Vec (n : Nat) a : Type where
  VNil : Vec Z a
  VCons : {n : Nat} -> a -> Vec n a -> Vec (S n) a"""
    ast = parse(source)
    decl = ast.declarations[0]
    assert isinstance(decl, DataDecl)
    assert len(decl.indices) == 1
    assert decl.indices[0][0].value == "n"


def test_parse_types():
    """Test parsing various type expressions."""
    # Simple types
    ast = parse("x : Int")
    decl = ast.declarations[0]
    ty = decl.type
    assert isinstance(ty, TypeConstructor)
    assert ty.name.value == "Int"
    
    # Function types
    ast = parse("f : Int -> Bool")
    decl = ast.declarations[0]
    ty = decl.type
    assert isinstance(ty, FunctionType)
    assert isinstance(ty.param_type, TypeConstructor)
    assert isinstance(ty.return_type, TypeConstructor)
    
    # Type application
    ast = parse("xs : List Int")
    decl = ast.declarations[0]
    ty = decl.type
    assert isinstance(ty, TypeApp)
    assert isinstance(ty.constructor, TypeConstructor)
    assert ty.constructor.name.value == "List"
    
    # Dependent function type
    ast = parse("f : (n : Nat) -> Vec n Int")
    decl = ast.declarations[0]
    ty = decl.type
    assert isinstance(ty, FunctionType)
    assert ty.param_name.value == "n"


def test_parse_module():
    """Test parsing a complete module."""
    source = """module Example

data Nat : Type where
  Z : Nat
  S : Nat -> Nat

add : Nat -> Nat -> Nat
add Z y = y
add (S x) y = S (add x y)
"""
    ast = parse(source)
    assert isinstance(ast, Module)
    assert ast.name.value == "Example"
    assert len(ast.declarations) == 3
    
    # Data declaration
    assert isinstance(ast.declarations[0], DataDecl)
    # Type signature
    assert isinstance(ast.declarations[1], TypeSignature)
    # Function definition
    assert isinstance(ast.declarations[2], FunctionDef)


def test_parse_complex_expression():
    """Test parsing a complex expression."""
    source = "\\{a : Type} -> \\(xs : List a) -> case xs of { Nil -> 0; (Cons x rest) -> S (length rest) }"
    
    expr = parse_expr(source)
    
    # Outer lambda
    assert isinstance(expr, Lambda)
    assert expr.implicit
    
    # Inner lambda
    expr = expr.body
    assert isinstance(expr, Lambda)
    assert not expr.implicit
    
    # Case expression
    expr = expr.body
    assert isinstance(expr, Case)
    assert len(expr.branches) == 2


def test_parse_errors():
    """Test parse error handling."""
    # Missing arrow in lambda
    with pytest.raises(ParseError) as exc_info:
        parse_expr("\\x x")
    assert "Expected ARROW" in str(exc_info.value)
    
    # Missing equals in let
    with pytest.raises(ParseError) as exc_info:
        parse_expr("let x in x")
    assert "Expected EQUALS" in str(exc_info.value)
    
    # Missing type after colon
    with pytest.raises(ParseError) as exc_info:
        parse("x : ")
    assert "Expected type" in str(exc_info.value)