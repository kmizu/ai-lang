"""REPL (Read-Eval-Print Loop) for ai-lang.

This module provides an interactive environment for ai-lang programming,
with support for incremental definitions, type checking, and evaluation.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Union
import readline
import os
import sys

from .parser import parse, ParseError
from .lexer import LexError, lex
from .typechecker import TypeChecker, type_check_module, TypeCheckError
from .evaluator import Evaluator, pretty_print_value, EvalError
from .syntax import *
from .core import Value


@dataclass
class ReplState:
    """State of the REPL session."""
    type_checker: TypeChecker
    evaluator: Evaluator
    history: List[str]
    definitions: List[Declaration]
    
    def __init__(self):
        self.type_checker = TypeChecker()
        self.evaluator = Evaluator(self.type_checker)
        self.history = []
        self.definitions = []
        
        # Initialize with built-in types
        self._init_builtins()
    
    def _init_builtins(self):
        """Initialize built-in types and functions."""
        # Add basic data types
        builtin_defs = """
        data Nat : Type where
          Z : Nat
          S : Nat -> Nat
        
        data Bool : Type where
          True : Bool
          False : Bool
        
        data List a : Type where
          Nil : List a
          Cons : a -> List a -> List a
        """
        
        try:
            module = parse(builtin_defs)
            for decl in module.declarations:
                self.add_declaration(decl, silent=True)
        except Exception:
            # If builtins fail to load, continue without them
            pass
    
    def add_declaration(self, decl: Declaration, silent: bool = False) -> Optional[Value]:
        """Add a declaration to the REPL state."""
        # Create a temporary module with all previous declarations plus the new one
        temp_module = Module(None, self.definitions + [decl])
        
        # Type check the entire module
        # This ensures the new declaration is consistent with previous ones
        temp_checker = TypeChecker()
        
        # Copy existing state to temp checker
        temp_checker.data_types = self.type_checker.data_types.copy()
        temp_checker.constructors = self.type_checker.constructors.copy()
        temp_checker.global_types = self.type_checker.global_types.copy()
        
        # Type check the new declaration
        from .typechecker import Context
        ctx = Context()
        temp_checker.check_declaration(decl, ctx)
        
        # If type checking succeeds, update the actual state
        self.type_checker.data_types = temp_checker.data_types
        self.type_checker.constructors = temp_checker.constructors
        self.type_checker.global_types = temp_checker.global_types
        
        # Update evaluator's type checker reference
        self.evaluator.type_checker = self.type_checker
        self.evaluator._init_builtins()
        
        # Evaluate the declaration
        result = None
        if isinstance(decl, FunctionDef):
            self.evaluator.eval_declaration(decl)
            # If it's a constant (no patterns), return its value
            if len(decl.clauses) == 1 and not decl.clauses[0].patterns:
                result = self.evaluator.global_env.get(decl.name.value)
        
        # Add to definitions
        self.definitions.append(decl)
        
        if not silent:
            # Print what was defined
            from .colors import Colors
            if isinstance(decl, DataDecl):
                print(Colors.success(f"Defined data type: {Colors.type_name(decl.name.value)}"))
            elif isinstance(decl, TypeSignature):
                print(Colors.info(f"Type signature: {Colors.bold(decl.name.value)}"))
            elif isinstance(decl, FunctionDef):
                print(Colors.success(f"Defined: {Colors.bold(decl.name.value)}"))
        
        return result
    
    def eval_expression(self, expr: Expr) -> Value:
        """Evaluate a standalone expression."""
        # Create a temporary function to hold the expression
        temp_func = FunctionDef(Name("_repl_expr"), [FunctionClause([], expr)])
        
        # We need to type check the expression
        # For now, we'll skip type checking standalone expressions
        # TODO: Implement expression type checking
        
        # Evaluate the expression
        return self.evaluator.eval_expr(expr, {})
    
    def get_type(self, name: str) -> Optional[str]:
        """Get the type of a defined name."""
        if name in self.type_checker.global_types:
            type_val = self.type_checker.global_types[name]
            # Convert type value to string
            # This is a simplified version
            return str(type_val)
        elif name in self.type_checker.constructors:
            ctor_info = self.type_checker.constructors[name]
            return str(ctor_info.type)
        return None


class Repl:
    """The REPL interface."""
    
    def __init__(self):
        self.state = ReplState()
        self.multiline_buffer = []
        self.in_multiline = False
        
        # Setup readline for better interaction
        self._setup_readline()
    
    def _setup_readline(self):
        """Setup readline with history and completion."""
        # Enable history
        histfile = os.path.expanduser("~/.ai_lang_history")
        try:
            readline.read_history_file(histfile)
        except FileNotFoundError:
            pass
        
        # Save history on exit
        import atexit
        atexit.register(readline.write_history_file, histfile)
        
        # Set up tab completion
        readline.set_completer(self._completer)
        readline.parse_and_bind("tab: complete")
    
    def _completer(self, text: str, state: int) -> Optional[str]:
        """Tab completion for defined names."""
        # Get all defined names
        names = list(self.state.type_checker.global_types.keys())
        names.extend(self.state.type_checker.constructors.keys())
        names.extend(self.state.type_checker.data_types.keys())
        
        # Add keywords
        keywords = ["data", "where", "case", "of", "let", "in", "Type"]
        names.extend(keywords)
        
        # Filter matching names
        matches = [name for name in names if name.startswith(text)]
        
        if state < len(matches):
            return matches[state]
        return None
    
    def run(self):
        """Run the REPL."""
        from .colors import Colors
        print(Colors.bold("ai-lang REPL v0.1.0"))
        print(f"Type {Colors.keyword(':help')} for help, {Colors.keyword(':quit')} to exit")
        print()
        
        while True:
            try:
                # Get input with appropriate prompt
                if self.in_multiline:
                    prompt = f"{Colors.dim('...')} "
                else:
                    prompt = f"{Colors.BRIGHT_BLUE}ai-lang>{Colors.RESET} "
                
                line = input(prompt)
                
                # Handle commands
                if not self.in_multiline and line.startswith(":"):
                    self.handle_command(line)
                    continue
                
                # Handle multiline input
                if self.in_multiline:
                    if line.strip() == "":
                        # Empty line ends multiline input
                        self.in_multiline = False
                        full_input = "\n".join(self.multiline_buffer)
                        self.multiline_buffer = []
                        self.process_input(full_input)
                    else:
                        self.multiline_buffer.append(line)
                else:
                    # Check if this starts a multiline input
                    if line.strip().endswith("where") or line.strip() == "":
                        if line.strip():
                            self.multiline_buffer.append(line)
                            self.in_multiline = True
                    else:
                        self.process_input(line)
                
            except EOFError:
                print("\nGoodbye!")
                break
            except KeyboardInterrupt:
                print("\nUse :quit to exit")
                self.in_multiline = False
                self.multiline_buffer = []
                continue
    
    def handle_command(self, command: str):
        """Handle REPL commands."""
        parts = command.split()
        cmd = parts[0]
        
        if cmd in [":quit", ":q"]:
            print("Goodbye!")
            sys.exit(0)
        
        elif cmd in [":help", ":h"]:
            self.show_help()
        
        elif cmd in [":type", ":t"]:
            if len(parts) < 2:
                print("Usage: :type <name>")
            else:
                name = parts[1]
                type_str = self.state.get_type(name)
                if type_str:
                    print(f"{name} : {type_str}")
                else:
                    print(f"Unknown name: {name}")
        
        elif cmd in [":list", ":l"]:
            self.list_definitions()
        
        elif cmd in [":clear", ":c"]:
            self.state = ReplState()
            print("State cleared")
        
        elif cmd in [":load"]:
            if len(parts) < 2:
                print("Usage: :load <filename>")
            else:
                self.load_file(parts[1])
        
        else:
            print(f"Unknown command: {cmd}")
            print("Type :help for help")
    
    def show_help(self):
        """Show help message."""
        help_text = """
ai-lang REPL Commands:

  :help, :h           Show this help message
  :quit, :q           Exit the REPL
  :type, :t <name>    Show the type of a name
  :list, :l           List all definitions
  :clear, :c          Clear all definitions
  :load <file>        Load definitions from a file

Language Features:

  42                  Natural number literal
  true, false         Boolean literals
  "hello"             String literal
  \\x -> x             Lambda expression
  f x y               Function application
  let x = e in b      Let binding
  case e of { ... }   Pattern matching

Type System:

  Type                Universe of types
  a -> b              Function type
  (x : A) -> B        Dependent function type
  {x : A} -> B        Implicit function type

Examples:

  data Nat : Type where
    Z : Nat
    S : Nat -> Nat

  add : Nat -> Nat -> Nat
  add Z y = y
  add (S x) y = S (add x y)
"""
        print(help_text)
    
    def list_definitions(self):
        """List all definitions in the current session."""
        if not self.state.definitions:
            print("No definitions")
            return
        
        print("Definitions:")
        for decl in self.state.definitions:
            if isinstance(decl, DataDecl):
                print(f"  data {decl.name.value}")
            elif isinstance(decl, TypeSignature):
                print(f"  {decl.name.value} : ...")
            elif isinstance(decl, FunctionDef):
                print(f"  {decl.name.value} = ...")
    
    def load_file(self, filename: str):
        """Load definitions from a file."""
        try:
            with open(filename, 'r') as f:
                content = f.read()
            
            # Parse the file
            module = parse(content)
            
            # Add each declaration
            for decl in module.declarations:
                self.state.add_declaration(decl, silent=True)
            
            print(f"Loaded {len(module.declarations)} definitions from {filename}")
            
        except FileNotFoundError:
            print(f"File not found: {filename}")
        except Exception as e:
            print(f"Error loading file: {e}")
    
    def process_input(self, input_str: str):
        """Process a line of input."""
        if not input_str.strip():
            return
        
        try:
            # Try to parse as a complete module
            module = parse(input_str)
            
            # Process each declaration
            for decl in module.declarations:
                result = self.state.add_declaration(decl)
                
                # If it's a standalone expression (parsed as _main), evaluate and print it
                if isinstance(decl, FunctionDef) and decl.name.value == "_main":
                    if result:
                        print(pretty_print_value(result))
            
        except LexError as e:
            from .colors import Colors
            print(Colors.error(f"Lexical error: {e}"))
        except ParseError as e:
            from .colors import Colors
            print(Colors.error(f"Parse error: {e}"))
        except TypeCheckError as e:
            from .colors import Colors
            print(Colors.error(f"Type error: {e}"))
        except EvalError as e:
            from .colors import Colors
            print(Colors.error(f"Evaluation error: {e}"))
        except Exception as e:
            from .colors import Colors
            print(Colors.error(f"Internal error: {e}"))
            # In development, show traceback
            import traceback
            traceback.print_exc()


def main():
    """Entry point for the REPL."""
    repl = Repl()
    repl.run()


if __name__ == "__main__":
    main()