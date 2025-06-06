"""Parser for ai-lang using recursive descent."""

from typing import List, Optional, Union, Tuple
from dataclasses import dataclass

from .lexer import Token, TokenType, lex
from .syntax import *


class ParseError(Exception):
    """Parse error exception."""
    def __init__(self, message: str, token: Token):
        super().__init__(f"Parse error at {token.line}:{token.column}: {message}")
        self.token = token
        self.line = token.line
        self.column = token.column


class Parser:
    """Recursive descent parser for ai-lang."""
    
    def __init__(self, tokens: List[Token], filename: Optional[str] = None):
        self.tokens = tokens
        self.position = 0
        self.current_token = tokens[0] if tokens else Token(TokenType.EOF, "", 1, 1)
        self.filename = filename
    
    def advance(self) -> None:
        """Move to the next token."""
        if self.position < len(self.tokens) - 1:
            self.position += 1
            self.current_token = self.tokens[self.position]
    
    def peek(self, offset: int = 1) -> Optional[Token]:
        """Look ahead at a token."""
        pos = self.position + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return None
    
    def expect(self, token_type: TokenType) -> Token:
        """Consume a token of the expected type."""
        if self.current_token.type != token_type:
            raise ParseError(
                f"Expected {token_type.name}, got {self.current_token.type.name}",
                self.current_token
            )
        token = self.current_token
        self.advance()
        return token
    
    def match(self, *token_types: TokenType) -> bool:
        """Check if current token matches any of the given types."""
        return self.current_token.type in token_types
    
    def consume(self, token_type: TokenType) -> bool:
        """Consume a token if it matches the type."""
        if self.current_token.type == token_type:
            self.advance()
            return True
        return False
    
    def skip_newlines(self) -> None:
        """Skip newline tokens."""
        while self.current_token.type == TokenType.NEWLINE:
            self.advance()
    
    def make_name(self, token: Token) -> Name:
        """Create a Name with source location from a token."""
        location = SourceLocation(token.line, token.column, self.filename)
        return Name(token.value, location)
    
    def current_location(self) -> SourceLocation:
        """Get the current source location."""
        return SourceLocation(self.current_token.line, self.current_token.column, self.filename)
    
    # Type parsing
    
    def parse_type(self) -> Type:
        """Parse a type expression."""
        result = self.parse_function_type()
        return result
    
    def parse_function_type(self) -> Type:
        """Parse function types (arrows) and constraint types."""
        # First check for constraint types (Eq A => ...)
        constraints = []
        
        # Check if we have constraints before the main type
        if self.current_token.type == TokenType.IDENT:
            # Look ahead to see if this is a constraint
            saved_pos = self.position
            self.advance()
            
            # If we see a type variable followed by =>, this is a constraint
            if self.current_token.type == TokenType.IDENT:
                next_token = self.peek()
                if next_token and (next_token.type == TokenType.COMMA or next_token.type == TokenType.DOUBLE_ARROW):
                    # Restore position and parse constraints
                    self.position = saved_pos
                    self.current_token = self.tokens[self.position]
                    constraints = self.parse_constraints()
                    self.expect(TokenType.DOUBLE_ARROW)
                else:
                    # Not a constraint, restore position
                    self.position = saved_pos
                    self.current_token = self.tokens[self.position]
            else:
                # Not a constraint, restore position
                self.position = saved_pos
                self.current_token = self.tokens[self.position]
        
        # Check for dependent function type
        if self.match(TokenType.LPAREN, TokenType.LBRACE):
            implicit = self.current_token.type == TokenType.LBRACE
            saved_pos = self.position
            self.advance()
            
            # Check if this is actually a dependent function type
            # by looking for "name : type"
            if (self.current_token.type == TokenType.IDENT and 
                self.peek() and self.peek().type == TokenType.COLON):
                # Parse parameter name
                name_token = self.current_token
                param_name = self.make_name(name_token)
                self.advance()
                
                self.expect(TokenType.COLON)
                param_type = self.parse_type()
                
                if implicit:
                    self.expect(TokenType.RBRACE)
                else:
                    self.expect(TokenType.RPAREN)
                
                self.expect(TokenType.ARROW)
                return_type = self.parse_type()
                
                result = FunctionType(param_name, param_type, return_type, implicit)
                
                # If we have constraints, wrap in ConstraintType
                if constraints:
                    return ConstraintType(constraints, result)
                return result
            else:
                # Not a dependent function type, restore and parse normally
                self.position = saved_pos
                self.current_token = self.tokens[self.position]
        
        # Simple function type
        left = self.parse_type_app()
        
        if self.consume(TokenType.ARROW):
            right = self.parse_function_type()
            result = FunctionType(None, left, right, False)
            
            # If we have constraints, wrap in ConstraintType
            if constraints:
                return ConstraintType(constraints, result)
            return result
        
        # If we have constraints but no arrow, wrap the type
        if constraints:
            return ConstraintType(constraints, left)
        
        return left
    
    def parse_type_app(self) -> Type:
        """Parse type application."""
        left = self.parse_atomic_type()
        
        # Parse type applications
        while (self.match(TokenType.IDENT, TokenType.TYPE, TokenType.LPAREN) and 
               not self.match(TokenType.ARROW)):
            arg = self.parse_atomic_type()
            left = TypeApp(left, arg)
        
        return left
    
    def parse_atomic_type(self) -> Type:
        """Parse atomic types."""
        self.skip_newlines()
        
        # Type universe
        if self.consume(TokenType.TYPE):
            # Check for universe level
            if self.current_token.type == TokenType.INT:
                level = int(self.current_token.value)
                self.advance()
                return UniverseType(level)
            return UniverseType()
        
        # Type in parentheses or tuple type
        if self.consume(TokenType.LPAREN):
            # Could be a single type in parens or a tuple
            types = []
            if not self.match(TokenType.RPAREN):
                types.append(self.parse_type())
                while self.consume(TokenType.COMMA):
                    types.append(self.parse_type())
            self.expect(TokenType.RPAREN)
            
            if len(types) == 1:
                # Single type in parentheses
                return types[0]
            else:
                # Tuple type
                return TupleType(types)
        
        # Type constructor or variable
        if self.current_token.type == TokenType.IDENT:
            name_token = self.current_token
            name = self.make_name(name_token)
            self.advance()
            
            # Simple heuristic: uppercase = constructor, lowercase = variable
            if name.value[0].isupper():
                return TypeConstructor(name)
            else:
                return TypeVar(name)
        
        raise ParseError(f"Expected type, got {self.current_token.type.name}", 
                        self.current_token)
    
    # Expression parsing
    
    def parse_expr(self) -> Expr:
        """Parse an expression."""
        return self.parse_lambda()
    
    def parse_lambda(self) -> Expr:
        """Parse lambda abstractions."""
        if self.consume(TokenType.LAMBDA):
            # Parse parameter
            implicit = False
            if self.consume(TokenType.LBRACE):
                implicit = True
            elif not self.consume(TokenType.LPAREN):
                # Simple lambda without parentheses - can have multiple parameters
                params = []
                while self.current_token.type == TokenType.IDENT:
                    param_token = self.expect(TokenType.IDENT)
                    params.append(self.make_name(param_token))
                
                if not params:
                    raise ParseError("Expected parameter after lambda", self.current_token)
                
                self.expect(TokenType.ARROW)
                body = self.parse_expr()
                
                # Build nested lambdas for multiple parameters
                result = body
                for param in reversed(params):
                    result = Lambda(param, None, result, False)
                return result
            
            param_token = self.expect(TokenType.IDENT)
            param = self.make_name(param_token)
            
            # Optional type annotation
            param_type = None
            if self.consume(TokenType.COLON):
                param_type = self.parse_type()
            
            if implicit:
                self.expect(TokenType.RBRACE)
            else:
                self.expect(TokenType.RPAREN)
            
            self.expect(TokenType.ARROW)
            body = self.parse_expr()
            
            return Lambda(param, param_type, body, implicit)
        
        return self.parse_let()
    
    def parse_let(self) -> Expr:
        """Parse let expressions."""
        if self.consume(TokenType.LET):
            name_token = self.expect(TokenType.IDENT)
            name = self.make_name(name_token)
            
            # Optional type annotation
            type_annotation = None
            if self.consume(TokenType.COLON):
                type_annotation = self.parse_type()
            
            self.expect(TokenType.EQUALS)
            value = self.parse_expr()
            self.expect(TokenType.IN)
            body = self.parse_expr()
            
            return Let(name, type_annotation, value, body)
        
        return self.parse_case()
    
    def parse_case(self) -> Expr:
        """Parse case expressions."""
        if self.consume(TokenType.CASE):
            scrutinee = self.parse_expr()
            self.expect(TokenType.OF)
            
            branches = []
            self.skip_newlines()
            
            # Optional opening brace
            has_brace = self.consume(TokenType.LBRACE)
            
            while True:
                self.skip_newlines()
                
                # Parse pattern
                pattern = self.parse_pattern()
                self.expect(TokenType.ARROW)
                branch_body = self.parse_expr()
                
                branches.append(CaseBranch(pattern, branch_body))
                
                # Check for more branches
                if not self.consume(TokenType.SEMICOLON):
                    break
            
            if has_brace:
                self.expect(TokenType.RBRACE)
            
            return Case(scrutinee, branches)
        
        return self.parse_cons_expr()
    
    def parse_cons_expr(self) -> Expr:
        """Parse cons expressions (::)."""
        left = self.parse_app()
        
        if self.consume(TokenType.CONS):
            # Build (Cons left right)
            right = self.parse_cons_expr()
            cons_name = Name("Cons", self.current_location())
            return App(App(Var(cons_name), left), right)
        
        return left
    
    def parse_app(self) -> Expr:
        """Parse function application and postfix operators."""
        left = self.parse_postfix()
        
        # Parse applications
        while self.match(TokenType.IDENT, TokenType.LPAREN, TokenType.INT, 
                         TokenType.BOOL, TokenType.STRING, TokenType.LBRACE, TokenType.LBRACKET):
            # Check for implicit application
            if self.current_token.type == TokenType.LBRACE:
                # Could be implicit application or record update
                saved_pos = self.position
                self.advance()
                
                # Check if it's a record update (has field = expr)
                if (self.current_token.type == TokenType.IDENT and 
                    self.peek() and self.peek().type == TokenType.EQUALS):
                    # Record update
                    fields = []
                    while not self.match(TokenType.RBRACE):
                        field_name_token = self.expect(TokenType.IDENT)
                        field_name = self.make_name(field_name_token)
                        self.expect(TokenType.EQUALS)
                        field_expr = self.parse_expr()
                        fields.append((field_name, field_expr))
                        
                        if not self.match(TokenType.RBRACE):
                            self.expect(TokenType.COMMA)
                    self.expect(TokenType.RBRACE)
                    left = RecordUpdate(left, fields)
                else:
                    # Implicit application
                    arg = self.parse_expr()
                    self.expect(TokenType.RBRACE)
                    left = App(left, arg, implicit=True)
            else:
                # Regular application
                arg = self.parse_postfix()
                left = App(left, arg, implicit=False)
        
        return left
    
    def parse_postfix(self) -> Expr:
        """Parse postfix operators like field access."""
        left = self.parse_atomic_expr()
        
        while True:
            if self.consume(TokenType.DOT):
                # Field access
                field_name_token = self.expect(TokenType.IDENT)
                field_name = self.make_name(field_name_token)
                left = FieldAccess(left, field_name)
            else:
                break
        
        return left
    
    def parse_atomic_expr(self) -> Expr:
        """Parse atomic expressions."""
        self.skip_newlines()
        
        # Literals
        if self.current_token.type == TokenType.INT:
            value = int(self.current_token.value)
            self.advance()
            return Literal(value)
        
        if self.current_token.type == TokenType.BOOL:
            value = self.current_token.value == "true"
            self.advance()
            return Literal(value)
        
        if self.current_token.type == TokenType.STRING:
            value = self.current_token.value
            self.advance()
            return Literal(value)
        
        # Variables (possibly qualified)
        if self.current_token.type == TokenType.IDENT:
            first_token = self.current_token
            first_name = self.make_name(first_token)
            self.advance()
            
            # Check for qualified name
            if self.match(TokenType.DOT) and self.peek() and self.peek().type == TokenType.IDENT:
                # This is a qualified name like Math.add
                self.advance()  # consume dot
                second_token = self.current_token
                second_name = self.make_name(second_token)
                self.advance()
                return Var(second_name, qualifier=first_name)
            else:
                return Var(first_name)
        
        # List literal
        if self.consume(TokenType.LBRACKET):
            elements = []
            while not self.match(TokenType.RBRACKET):
                elements.append(self.parse_expr())
                if not self.match(TokenType.RBRACKET):
                    self.expect(TokenType.COMMA)
            self.expect(TokenType.RBRACKET)
            return ListLiteral(elements)
        
        # Record literal
        if self.consume(TokenType.LBRACE):
            fields = []
            while not self.match(TokenType.RBRACE):
                # Parse field: name = expr
                field_name_token = self.expect(TokenType.IDENT)
                field_name = self.make_name(field_name_token)
                self.expect(TokenType.EQUALS)
                field_expr = self.parse_expr()
                fields.append((field_name, field_expr))
                
                if not self.match(TokenType.RBRACE):
                    self.expect(TokenType.COMMA)
            self.expect(TokenType.RBRACE)
            return RecordLiteral(fields)
        
        # Parenthesized expression or tuple
        if self.consume(TokenType.LPAREN):
            # Could be a single expr in parens or a tuple
            exprs = []
            if not self.match(TokenType.RPAREN):
                exprs.append(self.parse_expr())
                while self.consume(TokenType.COMMA):
                    exprs.append(self.parse_expr())
            self.expect(TokenType.RPAREN)
            
            if len(exprs) == 1:
                # Single expression in parentheses
                return exprs[0]
            else:
                # Tuple literal
                return TupleLiteral(exprs)
        
        raise ParseError(f"Expected expression, got {self.current_token.type.name}",
                        self.current_token)
    
    # Pattern parsing
    
    def parse_pattern(self) -> Pattern:
        """Parse a pattern with infix operators."""
        return self.parse_cons_pattern()
    
    def parse_cons_pattern(self) -> Pattern:
        """Parse cons patterns (x :: xs)."""
        left = self.parse_atomic_pattern()
        
        if self.consume(TokenType.CONS):
            # This is a cons pattern
            right = self.parse_cons_pattern()
            # Convert to (Cons left right)
            cons_name = Name("Cons", self.current_location())
            return PatternConstructor(cons_name, [left, right])
        
        return left
    
    def parse_atomic_pattern(self) -> Pattern:
        """Parse atomic patterns."""
        # Wildcard
        if self.consume(TokenType.UNDERSCORE):
            return PatternWildcard()
        
        # Literals
        if self.current_token.type == TokenType.INT:
            value = int(self.current_token.value)
            self.advance()
            return PatternLiteral(value)
        
        if self.current_token.type == TokenType.BOOL:
            value = self.current_token.value == "true"
            self.advance()
            return PatternLiteral(value)
        
        if self.current_token.type == TokenType.STRING:
            value = self.current_token.value
            self.advance()
            return PatternLiteral(value)
        
        # List pattern
        if self.consume(TokenType.LBRACKET):
            if self.consume(TokenType.RBRACKET):
                # Empty list pattern []
                nil_name = Name("Nil", self.current_location())
                return PatternConstructor(nil_name, [])
            else:
                # Non-empty list - convert to cons pattern
                # [x, y, z] becomes x :: y :: z :: Nil
                patterns = []
                patterns.append(self.parse_pattern())
                while self.consume(TokenType.COMMA):
                    patterns.append(self.parse_pattern())
                self.expect(TokenType.RBRACKET)
                
                # Build right-associative cons pattern
                nil_name = Name("Nil", self.current_location())
                cons_name = Name("Cons", self.current_location())
                result = PatternConstructor(nil_name, [])
                for p in reversed(patterns):
                    result = PatternConstructor(cons_name, [p, result])
                return result
        
        # Record pattern
        if self.consume(TokenType.LBRACE):
            fields = []
            while not self.match(TokenType.RBRACE):
                # Parse field: name = pattern
                field_name_token = self.expect(TokenType.IDENT)
                field_name = self.make_name(field_name_token)
                self.expect(TokenType.EQUALS)
                field_pattern = self.parse_pattern()
                fields.append((field_name, field_pattern))
                
                if not self.match(TokenType.RBRACE):
                    self.expect(TokenType.COMMA)
            self.expect(TokenType.RBRACE)
            return PatternRecord(fields)
        
        # Constructor or variable pattern
        if self.current_token.type == TokenType.IDENT:
            name_token = self.current_token
            name = self.make_name(name_token)
            self.advance()
            
            # Constructor pattern (uppercase)
            if name.value[0].isupper():
                # By default, constructors in patterns have no arguments
                # Arguments are only parsed when inside parentheses
                return PatternConstructor(name, [])
            
            # Variable pattern
            return PatternVar(name)
        
        # Parenthesized pattern or constructor with arguments
        if self.consume(TokenType.LPAREN):
            # Check if it's a constructor application
            if (self.current_token.type == TokenType.IDENT and 
                self.current_token.value[0].isupper()):
                # Constructor with arguments: (S x)
                name_token = self.current_token
                name = self.make_name(name_token)
                self.advance()
                args = []
                while not self.match(TokenType.RPAREN):
                    args.append(self.parse_pattern())
                self.expect(TokenType.RPAREN)
                return PatternConstructor(name, args)
            else:
                # Regular parenthesized pattern
                pattern = self.parse_pattern()
                self.expect(TokenType.RPAREN)
                return pattern
        
        raise ParseError(f"Expected pattern, got {self.current_token.type.name}",
                        self.current_token)
    
    # Declaration parsing
    
    def parse_declaration(self) -> Declaration:
        """Parse a top-level declaration."""
        self.skip_newlines()
        
        # Type class declaration
        if self.consume(TokenType.CLASS):
            return self.parse_class_decl()
        
        # Type class instance declaration
        if self.consume(TokenType.INSTANCE):
            return self.parse_instance_decl()
        
        # Data type declaration
        if self.consume(TokenType.DATA):
            return self.parse_data_decl()
        
        # Record type declaration
        if self.consume(TokenType.RECORD):
            return self.parse_record_decl()
        
        # Type alias
        if self.consume(TokenType.TYPE_ALIAS):
            return self.parse_type_alias()
        
        # Type signature or function definition
        if self.current_token.type == TokenType.IDENT:
            # Look ahead to determine if this is a type signature or function def
            saved_pos = self.position
            name_token = self.current_token
            name = self.make_name(name_token)
            self.advance()
            
            # Type signature
            if self.current_token.type == TokenType.COLON:
                self.advance()  # consume colon
                ty = self.parse_type()
                return TypeSignature(name, ty)
            
            # Function definition - restore and parse patterns
            else:
                # Restore position to re-parse the name as part of patterns
                self.position = saved_pos
                self.current_token = self.tokens[self.position]
                
                # For function definitions, the name is separate from patterns
                self.advance()  # Skip the function name
                
                # Parse the first clause
                patterns = []
                while (self.match(TokenType.IDENT, TokenType.LPAREN, 
                              TokenType.UNDERSCORE, TokenType.INT, 
                              TokenType.BOOL, TokenType.STRING) and
                       not self.match(TokenType.EQUALS)):
                    patterns.append(self.parse_pattern())
                
                self.expect(TokenType.EQUALS)
                body = self.parse_expr()
                
                clauses = [FunctionClause(patterns, body)]
                
                # Parse additional clauses
                while True:
                    # Save position to restore if this isn't another clause
                    saved_pos = self.position
                    self.skip_newlines()
                    
                    # Check if next token is the same function name
                    if (self.current_token.type == TokenType.IDENT and 
                        self.current_token.value == name.value):
                        self.advance()  # Skip function name
                        
                        patterns = []
                        while not self.match(TokenType.EQUALS):
                            patterns.append(self.parse_pattern())
                        
                        self.expect(TokenType.EQUALS)
                        body = self.parse_expr()
                        clauses.append(FunctionClause(patterns, body))
                    else:
                        # Not another clause, restore position
                        self.position = saved_pos
                        self.current_token = self.tokens[self.position]
                        break
                
                return FunctionDef(name, clauses)
        
        # If not a declaration, treat as a top-level expression
        # Create an anonymous function definition
        expr = self.parse_expr()
        return FunctionDef(Name("_main"), [FunctionClause([], expr)])
    
    def parse_data_decl(self) -> DataDecl:
        """Parse a data type declaration."""
        name_token = self.expect(TokenType.IDENT)
        name = self.make_name(name_token)
        
        # Parse type parameters and indices
        type_params = []
        indices = []
        
        # Parse parameters until we hit : or where
        while not self.match(TokenType.COLON, TokenType.WHERE):
            if self.consume(TokenType.LPAREN):
                # Index with type
                idx_name_token = self.expect(TokenType.IDENT)
                idx_name = self.make_name(idx_name_token)
                self.expect(TokenType.COLON)
                idx_type = self.parse_type()
                self.expect(TokenType.RPAREN)
                indices.append((idx_name, idx_type))
            elif self.current_token.type == TokenType.IDENT:
                # Type parameter
                param_token = self.current_token
                param = self.make_name(param_token)
                type_params.append(param)
                self.advance()
            else:
                break
        
        # Parse kind annotation if present
        if self.consume(TokenType.COLON):
            # For now, just parse and ignore the kind
            self.parse_type()
        
        # Parse constructors
        self.expect(TokenType.WHERE)
        self.skip_newlines()
        
        constructors = []
        # Keep track of whether we've seen a blank line (two consecutive newlines)
        # which indicates we've left the constructor block
        last_was_newline = False
        
        while True:
            # Check for blank line (end of constructor block)
            if self.current_token.type == TokenType.NEWLINE:
                if last_was_newline:
                    # Two consecutive newlines - we've left the block
                    break
                last_was_newline = True
                self.advance()
                continue
            else:
                last_was_newline = False
            
            # Skip other whitespace
            self.skip_newlines()
            
            # Check if we're at EOF or no longer have an identifier
            if self.current_token.type != TokenType.IDENT:
                break
                
            # Save position to check if this is a constructor
            saved_pos = self.position
            ctor_name_token = self.current_token
            ctor_name = self.make_name(ctor_name_token)
            self.advance()
            
            # Check if this looks like a constructor (has a colon after name)
            if self.current_token.type == TokenType.COLON:
                # It's a constructor
                self.advance()  # consume colon
                ctor_type = self.parse_type()
                constructors.append(Constructor(ctor_name, ctor_type))
            else:
                # Not a constructor, restore position and exit
                self.position = saved_pos
                self.current_token = self.tokens[self.position]
                break
        
        return DataDecl(name, type_params, indices, constructors)
    
    def parse_record_decl(self) -> RecordDecl:
        """Parse a record type declaration."""
        name_token = self.expect(TokenType.IDENT)
        name = self.make_name(name_token)
        
        # For now, records must have type Type
        self.expect(TokenType.COLON)
        self.expect(TokenType.TYPE)
        self.expect(TokenType.WHERE)
        
        # Parse fields
        fields = []
        self.skip_newlines()
        
        # Fields can be in braces or indented
        has_brace = self.consume(TokenType.LBRACE)
        
        while True:
            self.skip_newlines()
            
            # Check for end
            if has_brace and self.match(TokenType.RBRACE):
                break
            if not has_brace and not self.match(TokenType.IDENT):
                break
            
            # Parse field: name : type
            field_name_token = self.expect(TokenType.IDENT)
            field_name = self.make_name(field_name_token)
            self.expect(TokenType.COLON)
            field_type = self.parse_type()
            fields.append((field_name, field_type))
            
            # Optional separator
            self.consume(TokenType.COMMA) or self.consume(TokenType.SEMICOLON)
        
        if has_brace:
            self.expect(TokenType.RBRACE)
        
        return RecordDecl(name, fields)
    
    def parse_type_alias(self) -> TypeAlias:
        """Parse a type alias declaration."""
        name_token = self.expect(TokenType.IDENT)
        name = self.make_name(name_token)
        
        # Parse type parameters
        params = []
        while (self.current_token.type == TokenType.IDENT and 
               not self.match(TokenType.EQUALS)):
            param_token = self.current_token
            params.append(self.make_name(param_token))
            self.advance()
        
        self.expect(TokenType.EQUALS)
        body = self.parse_type()
        
        return TypeAlias(name, params, body)
    
    def parse_class_decl(self) -> ClassDecl:
        """Parse a type class declaration."""
        # Parse superclass constraints if present
        superclasses = []
        
        # Check if we have constraints before the class name
        if self.current_token.type == TokenType.IDENT:
            # Look ahead to see if this is a constraint
            saved_pos = self.position
            self.advance()
            
            # If we see a type variable followed by =>, this is a constraint
            if self.current_token.type == TokenType.IDENT:
                next_token = self.peek()
                if next_token and (next_token.type == TokenType.COMMA or next_token.type == TokenType.DOUBLE_ARROW):
                    # Restore position and parse constraints
                    self.position = saved_pos
                    self.current_token = self.tokens[self.position]
                    superclasses = self.parse_constraints()
                    self.expect(TokenType.DOUBLE_ARROW)
                else:
                    # Not a constraint, restore position
                    self.position = saved_pos
                    self.current_token = self.tokens[self.position]
            else:
                # Not a constraint, restore position
                self.position = saved_pos
                self.current_token = self.tokens[self.position]
        
        # Parse class name
        class_name_token = self.expect(TokenType.IDENT)
        class_name = self.make_name(class_name_token)
        
        # Parse type parameter
        param_token = self.expect(TokenType.IDENT)
        type_param = self.make_name(param_token)
        
        # Parse where clause
        self.expect(TokenType.WHERE)
        self.skip_newlines()
        
        # Parse methods
        methods = []
        
        # Keep track of whether we've seen a blank line
        last_was_newline = False
        
        while True:
            # Check for blank line (end of class block)
            if self.current_token.type == TokenType.NEWLINE:
                if last_was_newline:
                    # Two consecutive newlines - we've left the block
                    break
                last_was_newline = True
                self.advance()
                continue
            else:
                last_was_newline = False
            
            self.skip_newlines()
            
            # Check if we're at EOF or no longer have an identifier
            if self.current_token.type != TokenType.IDENT:
                break
            
            # Save position to check if this is a method
            saved_pos = self.position
            method_name_token = self.current_token
            method_name = self.make_name(method_name_token)
            self.advance()
            
            # Check if this looks like a method (has a colon after name)
            if self.current_token.type == TokenType.COLON:
                # It's a method
                self.advance()  # consume colon
                method_type = self.parse_type()
                methods.append((method_name, method_type))
            else:
                # Not a method, restore position and exit
                self.position = saved_pos
                self.current_token = self.tokens[self.position]
                break
        
        return ClassDecl(class_name, type_param, superclasses, methods)
    
    def parse_instance_decl(self) -> InstanceDecl:
        """Parse a type class instance declaration."""
        # Parse constraints if present
        constraints = []
        
        # Check if we have constraints before the class name
        if self.current_token.type == TokenType.IDENT:
            # Look ahead to see if this is a constraint
            saved_pos = self.position
            self.advance()
            
            # If we see a type followed by =>, this is a constraint
            if self.current_token.type == TokenType.IDENT or self.current_token.type == TokenType.TYPE:
                next_token = self.peek()
                if next_token and (next_token.type == TokenType.COMMA or next_token.type == TokenType.DOUBLE_ARROW):
                    # Restore position and parse constraints
                    self.position = saved_pos
                    self.current_token = self.tokens[self.position]
                    constraints = self.parse_constraints()
                    self.expect(TokenType.DOUBLE_ARROW)
                else:
                    # Not a constraint, restore position
                    self.position = saved_pos
                    self.current_token = self.tokens[self.position]
            else:
                # Not a constraint, restore position
                self.position = saved_pos
                self.current_token = self.tokens[self.position]
        
        # Parse class name
        class_name_token = self.expect(TokenType.IDENT)
        class_name = self.make_name(class_name_token)
        
        # Parse type
        instance_type = self.parse_type()
        
        # Parse where clause
        self.expect(TokenType.WHERE)
        self.skip_newlines()
        
        # Parse method implementations
        methods = []
        
        # Keep track of whether we've seen a blank line
        last_was_newline = False
        
        while True:
            # Check for blank line (end of instance block)
            if self.current_token.type == TokenType.NEWLINE:
                if last_was_newline:
                    # Two consecutive newlines - we've left the block
                    break
                last_was_newline = True
                self.advance()
                continue
            else:
                last_was_newline = False
            
            self.skip_newlines()
            
            # Check if we're at EOF or no longer have an identifier
            if self.current_token.type != TokenType.IDENT:
                break
            
            # Save position to check if this is a method implementation
            saved_pos = self.position
            method_name_token = self.current_token
            method_name = self.make_name(method_name_token)
            self.advance()
            
            # Check if this looks like a method implementation (has = after name)
            if self.current_token.type == TokenType.EQUALS:
                # It's a method implementation
                self.advance()  # consume equals
                implementation = self.parse_expr()
                methods.append((method_name, implementation))
            else:
                # Not a method, restore position and exit
                self.position = saved_pos
                self.current_token = self.tokens[self.position]
                break
        
        return InstanceDecl(class_name, instance_type, constraints, methods)
    
    def parse_constraints(self) -> List[Tuple[Name, Type]]:
        """Parse type class constraints (e.g., 'Eq A, Ord A')."""
        constraints = []
        
        while True:
            # Parse constraint: ClassName Type
            class_name_token = self.expect(TokenType.IDENT)
            class_name = self.make_name(class_name_token)
            constraint_type = self.parse_type()
            constraints.append((class_name, constraint_type))
            
            # Check for more constraints
            if not self.consume(TokenType.COMMA):
                break
        
        return constraints
    
    def parse_module(self) -> Module:
        """Parse a module (file)."""
        # Optional module declaration
        module_name = None
        if self.consume(TokenType.MODULE):
            name_token = self.expect(TokenType.IDENT)
            module_name = self.make_name(name_token)
            self.skip_newlines()
        
        # Parse imports and exports
        imports = []
        exports = []
        
        while self.match(TokenType.IMPORT, TokenType.EXPORT):
            self.skip_newlines()
            if self.consume(TokenType.IMPORT):
                imports.append(self.parse_import())
            elif self.consume(TokenType.EXPORT):
                exports.append(self.parse_export())
            self.skip_newlines()
        
        # Parse declarations
        declarations = []
        while self.current_token.type != TokenType.EOF:
            self.skip_newlines()
            if self.current_token.type == TokenType.EOF:
                break
            
            # Check for export declarations in the body
            if self.consume(TokenType.EXPORT):
                exports.append(self.parse_export())
            else:
                decl = self.parse_declaration()
                declarations.append(decl)
            self.skip_newlines()
        
        result = Module(module_name, imports, exports, declarations)
        return result
    
    def parse_import(self) -> Import:
        """Parse an import declaration."""
        # import ModuleName
        # import ModuleName as Alias
        # import ModuleName (item1, item2)
        module_name_token = self.expect(TokenType.IDENT)
        module_name = self.make_name(module_name_token)
        
        alias = None
        items = None
        
        if self.consume(TokenType.AS):
            # import Module as Alias
            alias_token = self.expect(TokenType.IDENT)
            alias = self.make_name(alias_token)
        elif self.consume(TokenType.LPAREN):
            # import Module (item1, item2)
            items = []
            while not self.match(TokenType.RPAREN):
                item_name_token = self.expect(TokenType.IDENT)
                item_name = self.make_name(item_name_token)
                item_alias = None
                
                if self.consume(TokenType.AS):
                    item_alias_token = self.expect(TokenType.IDENT)
                    item_alias = self.make_name(item_alias_token)
                
                items.append(ImportItem(item_name, item_alias))
                
                if not self.match(TokenType.RPAREN):
                    self.expect(TokenType.COMMA)
            self.expect(TokenType.RPAREN)
        
        return Import(module_name, alias, items)
    
    def parse_export(self) -> Export:
        """Parse an export declaration."""
        # export name1, name2, ...
        names = []
        
        name_token = self.expect(TokenType.IDENT)
        names.append(self.make_name(name_token))
        
        while self.consume(TokenType.COMMA):
            name_token = self.expect(TokenType.IDENT)
            names.append(self.make_name(name_token))
        
        return Export(names)


def parse(source: str, filename: Optional[str] = None) -> Module:
    """Parse source code into an AST."""
    tokens = lex(source)
    parser = Parser(tokens, filename)
    return parser.parse_module()