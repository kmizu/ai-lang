"""Tests for the lexer."""

import pytest
from ai_lang.lexer import Lexer, Token, TokenType, LexError, lex


def test_simple_tokens():
    """Test lexing simple tokens."""
    source = "( ) { } [ ] -> \\ : ; , . = | _"
    tokens = lex(source)
    
    expected_types = [
        TokenType.LPAREN, TokenType.RPAREN,
        TokenType.LBRACE, TokenType.RBRACE,
        TokenType.LBRACKET, TokenType.RBRACKET,
        TokenType.ARROW, TokenType.LAMBDA,
        TokenType.COLON, TokenType.SEMICOLON,
        TokenType.COMMA, TokenType.DOT,
        TokenType.EQUALS, TokenType.PIPE,
        TokenType.UNDERSCORE,
        TokenType.EOF
    ]
    
    assert len(tokens) == len(expected_types)
    for token, expected_type in zip(tokens, expected_types):
        assert token.type == expected_type


def test_keywords():
    """Test lexing keywords."""
    source = "Type let in case of where data module import true false"
    tokens = lex(source)
    
    expected = [
        (TokenType.TYPE, "Type"),
        (TokenType.LET, "let"),
        (TokenType.IN, "in"),
        (TokenType.CASE, "case"),
        (TokenType.OF, "of"),
        (TokenType.WHERE, "where"),
        (TokenType.DATA, "data"),
        (TokenType.MODULE, "module"),
        (TokenType.IMPORT, "import"),
        (TokenType.BOOL, "true"),
        (TokenType.BOOL, "false"),
        (TokenType.EOF, ""),
    ]
    
    assert len(tokens) == len(expected)
    for token, (expected_type, expected_value) in zip(tokens, expected):
        assert token.type == expected_type
        assert token.value == expected_value


def test_identifiers():
    """Test lexing identifiers."""
    source = "x foo Bar _test test' x123"
    tokens = lex(source)
    
    expected_values = ["x", "foo", "Bar", "_test", "test'", "x123"]
    
    assert len(tokens) == len(expected_values) + 1  # +1 for EOF
    for i, expected_value in enumerate(expected_values):
        assert tokens[i].type == TokenType.IDENT
        assert tokens[i].value == expected_value


def test_numbers():
    """Test lexing numbers."""
    source = "0 42 -5 123456"
    tokens = lex(source)
    
    expected_values = ["0", "42", "-5", "123456"]
    
    assert len(tokens) == len(expected_values) + 1  # +1 for EOF
    for i, expected_value in enumerate(expected_values):
        assert tokens[i].type == TokenType.INT
        assert tokens[i].value == expected_value


def test_strings():
    """Test lexing string literals."""
    source = '"hello" "world\\n" "escaped\\"quote" ""'
    tokens = lex(source)
    
    expected_values = ["hello", "world\n", 'escaped"quote', ""]
    
    assert len(tokens) == len(expected_values) + 1  # +1 for EOF
    for i, expected_value in enumerate(expected_values):
        assert tokens[i].type == TokenType.STRING
        assert tokens[i].value == expected_value


def test_comments():
    """Test that comments are skipped."""
    source = """
    -- Single line comment
    x -- inline comment
    {- Multi
       line
       comment -}
    y
    {- Nested {- comments -} work -}
    z
    """
    tokens = lex(source)
    
    # Filter out newlines for easier testing
    tokens = [t for t in tokens if t.type != TokenType.NEWLINE]
    
    assert len(tokens) == 4  # x, y, z, EOF
    assert tokens[0].type == TokenType.IDENT and tokens[0].value == "x"
    assert tokens[1].type == TokenType.IDENT and tokens[1].value == "y"
    assert tokens[2].type == TokenType.IDENT and tokens[2].value == "z"
    assert tokens[3].type == TokenType.EOF


def test_position_tracking():
    """Test that line and column positions are tracked correctly."""
    source = "let\nx = 42"
    tokens = lex(source)
    
    # Filter out newlines
    tokens = [t for t in tokens if t.type != TokenType.NEWLINE]
    
    assert tokens[0].line == 1 and tokens[0].column == 1  # let
    assert tokens[1].line == 2 and tokens[1].column == 1  # x
    assert tokens[2].line == 2 and tokens[2].column == 3  # =
    assert tokens[3].line == 2 and tokens[3].column == 5  # 42


def test_complex_expression():
    """Test lexing a complex expression."""
    source = "\\(x : Int) -> case x of { 0 -> true; _ -> false }"
    tokens = lex(source)
    
    expected_types = [
        TokenType.LAMBDA,
        TokenType.LPAREN,
        TokenType.IDENT,  # x
        TokenType.COLON,
        TokenType.IDENT,  # Int
        TokenType.RPAREN,
        TokenType.ARROW,
        TokenType.CASE,
        TokenType.IDENT,  # x
        TokenType.OF,
        TokenType.LBRACE,
        TokenType.INT,    # 0
        TokenType.ARROW,
        TokenType.BOOL,   # true
        TokenType.SEMICOLON,
        TokenType.UNDERSCORE,
        TokenType.ARROW,
        TokenType.BOOL,   # false
        TokenType.RBRACE,
        TokenType.EOF,
    ]
    
    assert len(tokens) == len(expected_types)
    for token, expected_type in zip(tokens, expected_types):
        assert token.type == expected_type


def test_unterminated_string_error():
    """Test error on unterminated string."""
    source = '"hello'
    
    with pytest.raises(LexError) as exc_info:
        lex(source)
    
    assert "Unterminated string literal" in str(exc_info.value)


def test_invalid_escape_sequence():
    """Test error on invalid escape sequence."""
    source = '"hello\\x"'
    
    with pytest.raises(LexError) as exc_info:
        lex(source)
    
    assert "Invalid escape sequence" in str(exc_info.value)


def test_unexpected_character():
    """Test error on unexpected character."""
    source = "x + y"
    
    with pytest.raises(LexError) as exc_info:
        lex(source)
    
    assert "Unexpected character '+'" in str(exc_info.value)