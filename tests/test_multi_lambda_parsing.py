"""Test multi-parameter lambda parsing."""

import pytest
from ai_lang.parser import parse
from ai_lang.syntax import Lambda, Var, Module, FunctionDef


def test_single_parameter_lambda():
    """Test single parameter lambda."""
    code = r"\x -> x"
    module = parse(f"test : Nat -> Nat\ntest = {code}")
    func_def = module.declarations[1]
    assert isinstance(func_def, FunctionDef)
    body = func_def.clauses[0].body
    assert isinstance(body, Lambda)
    assert body.param.value == "x"
    assert isinstance(body.body, Var)
    assert body.body.name.value == "x"


def test_two_parameter_lambda():
    """Test two parameter lambda."""
    code = r"\x y -> x"
    module = parse(f"test : Nat -> Nat -> Nat\ntest = {code}")
    func_def = module.declarations[1]
    body = func_def.clauses[0].body
    
    # Should be Lambda(x, Lambda(y, Var(x)))
    assert isinstance(body, Lambda)
    assert body.param.value == "x"
    assert isinstance(body.body, Lambda)
    assert body.body.param.value == "y"
    assert isinstance(body.body.body, Var)
    assert body.body.body.name.value == "x"


def test_three_parameter_lambda():
    """Test three parameter lambda."""
    code = r"\x y z -> y"
    module = parse(f"test : Nat -> Nat -> Nat -> Nat\ntest = {code}")
    func_def = module.declarations[1]
    body = func_def.clauses[0].body
    
    # Should be Lambda(x, Lambda(y, Lambda(z, Var(y))))
    assert isinstance(body, Lambda)
    assert body.param.value == "x"
    assert isinstance(body.body, Lambda)
    assert body.body.param.value == "y"
    assert isinstance(body.body.body, Lambda)
    assert body.body.body.param.value == "z"
    assert isinstance(body.body.body.body, Var)
    assert body.body.body.body.name.value == "y"


def test_lambda_with_application():
    """Test lambda with function application in body."""
    code = r"\x y -> f x y"
    module = parse(f"test : Nat -> Nat -> Nat\ntest = {code}")
    func_def = module.declarations[1]
    body = func_def.clauses[0].body
    
    # Should be Lambda(x, Lambda(y, App(App(f, x), y)))
    assert isinstance(body, Lambda)
    assert body.param.value == "x"
    assert isinstance(body.body, Lambda)
    assert body.body.param.value == "y"
    # Body should be an application


def test_lambda_in_expression_context():
    """Test lambda in various expression contexts."""
    # In let binding
    code = """
    test : Nat
    test = let f = \\x y -> x in f Z (S Z)
    """
    module = parse(code)
    # Should parse successfully
    assert len(module.declarations) == 2
    
    # In application
    code = """
    test : Nat
    test = (\\x y -> x) Z (S Z)
    """
    module = parse(code)
    # Should parse successfully
    assert len(module.declarations) == 2


def test_nested_lambda_vs_multi_param():
    """Test that nested lambdas and multi-param lambdas are equivalent."""
    code1 = r"\x -> \y -> x"
    code2 = r"\x y -> x"
    
    module1 = parse(f"test : Nat -> Nat -> Nat\ntest = {code1}")
    module2 = parse(f"test : Nat -> Nat -> Nat\ntest = {code2}")
    
    body1 = module1.declarations[1].clauses[0].body
    body2 = module2.declarations[1].clauses[0].body
    
    # Both should produce the same structure
    assert str(body1) == str(body2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])