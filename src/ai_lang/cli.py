"""Command-line interface for ai-lang."""

import click
import sys
import os
from typing import Optional

# Version information
__version__ = "0.1.0"

@click.command()
@click.argument('filename', required=False, type=click.Path(exists=True))
@click.option('--ast', is_flag=True, help='Print the abstract syntax tree')
@click.option('--type-check-only', is_flag=True, help='Only type check, do not evaluate')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed type checking information')
@click.option('--no-builtins', is_flag=True, help='Start REPL without built-in types')
@click.option('--version', is_flag=True, help='Show version information')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
@click.option('--timing', is_flag=True, help='Show timing information')
def main(filename: Optional[str] = None, 
         ast: bool = False,
         type_check_only: bool = False,
         verbose: bool = False,
         no_builtins: bool = False,
         version: bool = False,
         output: Optional[str] = None,
         timing: bool = False) -> None:
    """ai-lang - A dependently-typed programming language.
    
    If FILENAME is provided, compile and run the file.
    Otherwise, start an interactive REPL.
    
    Examples:
    
      ai-lang                    # Start REPL
      
      ai-lang program.ai         # Run a program
      
      ai-lang program.ai --ast   # Show AST
      
      ai-lang -v program.ai      # Verbose output
    """
    
    # Handle version flag
    if version:
        print(f"ai-lang version {__version__}")
        print("A dependently-typed programming language")
        sys.exit(0)
    if filename:
        # Compile and run file
        from ai_lang.parser import parse, ParseError
        from ai_lang.lexer import LexError
        from ai_lang.typechecker import type_check_module, TypeCheckError
        from ai_lang.evaluator import Evaluator, pretty_print_value, EvalError
        
        # Timing support
        if timing:
            import time
            start_time = time.time()
        
        try:
            # Read source file
            if verbose:
                print(f"Reading {filename}...")
            
            with open(filename, 'r') as f:
                source = f.read()
            
            # Parse the source
            if verbose:
                print("Parsing...")
            parse_start = time.time() if timing else 0
            
            module = parse(source)
            
            if timing:
                parse_time = time.time() - parse_start
                print(f"Parse time: {parse_time:.3f}s")
            
            if ast:
                output_content = f"Abstract Syntax Tree:\n{module}"
                if output:
                    with open(output, 'w') as f:
                        f.write(output_content)
                    print(f"AST written to {output}")
                else:
                    print(output_content)
                return
            
            # Type check
            if verbose:
                print("Type checking...")
            type_check_start = time.time() if timing else 0
            
            checker = type_check_module(module)
            
            if timing:
                type_check_time = time.time() - type_check_start
                print(f"Type check time: {type_check_time:.3f}s")
            
            if type_check_only:
                from ai_lang.colors import Colors
                print(Colors.success(f"Type checked successfully ({len(module.declarations)} declarations)"))
                return
            
            # Evaluate
            if verbose:
                print("Evaluating...")
            eval_start = time.time() if timing else 0
            
            evaluator = Evaluator(checker, trace=verbose)
            evaluator.eval_module(module)
            
            if timing:
                eval_time = time.time() - eval_start
                print(f"Evaluation time: {eval_time:.3f}s")
            
            # Output results
            if "main" in evaluator.global_env:
                result = evaluator.global_env["main"]
                result_str = pretty_print_value(result)
                
                if output:
                    with open(output, 'w') as f:
                        f.write(result_str + '\n')
                    print(f"Result written to {output}: {result_str}")
                else:
                    print(f"Result: {result_str}")
            else:
                from ai_lang.colors import Colors
                print(Colors.success(f"Loaded {len(module.declarations)} definitions"))
                if verbose:
                    print(f"\n{Colors.bold('Defined:')}")
                    for name in sorted(checker.global_types.keys()):
                        if not name.startswith('_'):
                            print(f"  {Colors.type_name(name)}")
            
            if timing:
                total_time = time.time() - start_time
                print(f"\nTotal time: {total_time:.3f}s")
                
        except FileNotFoundError:
            error_msg = f"Error: File '{filename}' not found"
            print(error_msg, file=sys.stderr)
            sys.exit(1)
        except (LexError, ParseError) as e:
            error_msg = f"Syntax error: {e}"
            print(error_msg, file=sys.stderr)
            if verbose and hasattr(e, 'location') and e.location:
                show_error_context(source, e.location)
            sys.exit(1)
        except TypeCheckError as e:
            error_msg = f"Type error: {e}"
            print(error_msg, file=sys.stderr)
            if verbose:
                import traceback
                traceback.print_exc()
            if verbose and hasattr(e, 'location') and e.location:
                show_error_context(source, e.location)
            sys.exit(1)
        except EvalError as e:
            error_msg = f"Runtime error: {e}"
            print(error_msg, file=sys.stderr)
            sys.exit(1)
        except KeyboardInterrupt:
            print("\nInterrupted", file=sys.stderr)
            sys.exit(130)
        except Exception as e:
            error_msg = f"Internal error: {e}"
            print(error_msg, file=sys.stderr)
            if verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    else:
        # Start REPL
        from ai_lang.repl import Repl
        repl = Repl()
        
        # Configure REPL based on options
        if no_builtins:
            # Clear built-in types
            repl.state.definitions = []
            repl.state.type_checker = type(repl.state.type_checker)()
            repl.state.evaluator = type(repl.state.evaluator)(repl.state.type_checker)
        
        if verbose:
            repl.state.evaluator.trace = True
        
        try:
            repl.run()
        except KeyboardInterrupt:
            print("\nGoodbye!")
        except Exception as e:
            print(f"REPL error: {e}", file=sys.stderr)
            if verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)


def show_error_context(source: str, location) -> None:
    """Show context around an error location."""
    lines = source.split('\n')
    if hasattr(location, 'line') and 0 < location.line <= len(lines):
        line_num = location.line
        print(f"\nAt line {line_num}:")
        
        # Show surrounding lines
        start = max(0, line_num - 2)
        end = min(len(lines), line_num + 1)
        
        for i in range(start, end):
            if i == line_num - 1:
                print(f"  > {i+1:3d} | {lines[i]}")
                if hasattr(location, 'column'):
                    print(f"        | {' ' * (location.column - 1)}^")
            else:
                print(f"    {i+1:3d} | {lines[i]}")


if __name__ == "__main__":
    main()