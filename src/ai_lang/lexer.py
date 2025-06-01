"""Lexer for ai-lang."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Iterator
import re


class TokenType(Enum):
    """Token types for ai-lang."""
    # Literals
    INT = auto()
    BOOL = auto()
    STRING = auto()
    
    # Identifiers and keywords
    IDENT = auto()
    TYPE = auto()      # Type keyword
    TYPE_ALIAS = auto() # type (lowercase)
    LET = auto()       # let
    IN = auto()        # in
    CASE = auto()      # case
    OF = auto()        # of
    WHERE = auto()     # where
    DATA = auto()      # data
    RECORD = auto()    # record
    MODULE = auto()    # module
    IMPORT = auto()    # import
    EXPORT = auto()    # export
    AS = auto()        # as
    CLASS = auto()     # class
    INSTANCE = auto()  # instance
    
    # Symbols
    LPAREN = auto()    # (
    RPAREN = auto()    # )
    LBRACE = auto()    # {
    RBRACE = auto()    # }
    LBRACKET = auto()  # [
    RBRACKET = auto()  # ]
    ARROW = auto()     # ->
    DOUBLE_ARROW = auto()  # =>
    LAMBDA = auto()    # \
    COLON = auto()     # :
    SEMICOLON = auto() # ;
    COMMA = auto()     # ,
    DOT = auto()       # .
    EQUALS = auto()    # =
    PIPE = auto()      # |
    UNDERSCORE = auto() # _
    CONS = auto()      # ::
    
    # Special
    EOF = auto()
    NEWLINE = auto()


@dataclass
class Token:
    """A lexical token."""
    type: TokenType
    value: str
    line: int
    column: int
    
    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, {self.line}, {self.column})"


class LexError(Exception):
    """Lexical analysis error."""
    def __init__(self, message: str, line: int, column: int):
        super().__init__(f"Lexical error at {line}:{column}: {message}")
        self.line = line
        self.column = column


class Lexer:
    """Lexical analyzer for ai-lang."""
    
    KEYWORDS = {
        'Type': TokenType.TYPE,
        'type': TokenType.TYPE_ALIAS,
        'let': TokenType.LET,
        'in': TokenType.IN,
        'case': TokenType.CASE,
        'of': TokenType.OF,
        'where': TokenType.WHERE,
        'data': TokenType.DATA,
        'record': TokenType.RECORD,
        'module': TokenType.MODULE,
        'import': TokenType.IMPORT,
        'export': TokenType.EXPORT,
        'as': TokenType.AS,
        'class': TokenType.CLASS,
        'instance': TokenType.INSTANCE,
        'true': TokenType.BOOL,
        'false': TokenType.BOOL,
    }
    
    SYMBOLS = {
        '(': TokenType.LPAREN,
        ')': TokenType.RPAREN,
        '{': TokenType.LBRACE,
        '}': TokenType.RBRACE,
        '[': TokenType.LBRACKET,
        ']': TokenType.RBRACKET,
        '->': TokenType.ARROW,
        '=>': TokenType.DOUBLE_ARROW,
        '::': TokenType.CONS,
        '\\': TokenType.LAMBDA,
        ':': TokenType.COLON,
        ';': TokenType.SEMICOLON,
        ',': TokenType.COMMA,
        '.': TokenType.DOT,
        '=': TokenType.EQUALS,
        '|': TokenType.PIPE,
        '_': TokenType.UNDERSCORE,
    }
    
    def __init__(self, source: str):
        self.source = source
        self.position = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []
    
    def current_char(self) -> Optional[str]:
        """Get the current character."""
        if self.position >= len(self.source):
            return None
        return self.source[self.position]
    
    def peek_char(self, offset: int = 1) -> Optional[str]:
        """Peek at a character ahead."""
        pos = self.position + offset
        if pos >= len(self.source):
            return None
        return self.source[pos]
    
    def advance(self) -> None:
        """Move to the next character."""
        if self.position < len(self.source):
            if self.source[self.position] == '\n':
                self.line += 1
                self.column = 1
            else:
                self.column += 1
            self.position += 1
    
    def skip_whitespace(self) -> None:
        """Skip whitespace and comments."""
        while self.current_char() is not None:
            # Skip whitespace
            if self.current_char() in ' \t\r':
                self.advance()
            # Skip newlines (track them for layout)
            elif self.current_char() == '\n':
                token = Token(TokenType.NEWLINE, '\n', self.line, self.column)
                self.tokens.append(token)
                self.advance()
            # Skip single-line comments
            elif self.current_char() == '-' and self.peek_char() == '-':
                self.advance()  # Skip first -
                self.advance()  # Skip second -
                while self.current_char() is not None and self.current_char() != '\n':
                    self.advance()
            # Skip multi-line comments
            elif self.current_char() == '{' and self.peek_char() == '-':
                self.advance()  # Skip {
                self.advance()  # Skip -
                depth = 1
                while depth > 0 and self.current_char() is not None:
                    if self.current_char() == '-' and self.peek_char() == '}':
                        self.advance()
                        self.advance()
                        depth -= 1
                    elif self.current_char() == '{' and self.peek_char() == '-':
                        self.advance()
                        self.advance()
                        depth += 1
                    else:
                        self.advance()
            else:
                break
    
    def read_string(self) -> str:
        """Read a string literal."""
        start_line = self.line
        start_column = self.column
        self.advance()  # Skip opening quote
        
        value = ""
        while self.current_char() is not None and self.current_char() != '"':
            if self.current_char() == '\\':
                self.advance()
                if self.current_char() in 'nrt"\\':
                    escape_map = {'n': '\n', 'r': '\r', 't': '\t', '"': '"', '\\': '\\'}
                    value += escape_map[self.current_char()]
                    self.advance()
                else:
                    raise LexError(f"Invalid escape sequence \\{self.current_char()}", 
                                 self.line, self.column)
            else:
                value += self.current_char()
                self.advance()
        
        if self.current_char() != '"':
            raise LexError("Unterminated string literal", start_line, start_column)
        
        self.advance()  # Skip closing quote
        return value
    
    def read_number(self) -> str:
        """Read a number literal."""
        value = ""
        if self.current_char() == '-':
            value += '-'
            self.advance()
        
        while self.current_char() is not None and self.current_char().isdigit():
            value += self.current_char()
            self.advance()
        
        return value
    
    def read_identifier(self) -> str:
        """Read an identifier or keyword."""
        value = ""
        while (self.current_char() is not None and 
               (self.current_char().isalnum() or self.current_char() in '_\'')):
            value += self.current_char()
            self.advance()
        
        return value
    
    def tokenize(self) -> List[Token]:
        """Tokenize the source code."""
        self.tokens = []
        
        while self.position < len(self.source):
            self.skip_whitespace()
            
            if self.current_char() is None:
                break
            
            start_line = self.line
            start_column = self.column
            
            # String literals
            if self.current_char() == '"':
                value = self.read_string()
                self.tokens.append(Token(TokenType.STRING, value, start_line, start_column))
            
            # Number literals
            elif self.current_char().isdigit() or (self.current_char() == '-' and 
                                                  self.peek_char() and 
                                                  self.peek_char().isdigit()):
                value = self.read_number()
                self.tokens.append(Token(TokenType.INT, value, start_line, start_column))
            
            # Check for standalone underscore first
            elif self.current_char() == '_' and (self.peek_char() is None or 
                 not (self.peek_char().isalnum() or self.peek_char() in '_\'')):
                self.advance()
                self.tokens.append(Token(TokenType.UNDERSCORE, '_', start_line, start_column))
            
            # Identifiers and keywords
            elif self.current_char().isalpha() or self.current_char() == '_':
                value = self.read_identifier()
                token_type = self.KEYWORDS.get(value, TokenType.IDENT)
                self.tokens.append(Token(token_type, value, start_line, start_column))
            
            # Two-character symbols
            elif self.current_char() and self.peek_char() and \
                 self.current_char() + self.peek_char() in self.SYMBOLS:
                symbol = self.current_char() + self.peek_char()
                self.advance()
                self.advance()
                self.tokens.append(Token(self.SYMBOLS[symbol], symbol, start_line, start_column))
            
            # Single-character symbols
            elif self.current_char() in self.SYMBOLS:
                symbol = self.current_char()
                self.advance()
                self.tokens.append(Token(self.SYMBOLS[symbol], symbol, start_line, start_column))
            
            else:
                raise LexError(f"Unexpected character '{self.current_char()}'", 
                             self.line, self.column)
        
        # Add EOF token
        self.tokens.append(Token(TokenType.EOF, '', self.line, self.column))
        
        return self.tokens


def lex(source: str) -> List[Token]:
    """Convenience function to tokenize source code."""
    lexer = Lexer(source)
    return lexer.tokenize()