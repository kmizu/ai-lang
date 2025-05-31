#!/bin/bash
# Demo script showing ai-lang CLI features

echo "=== ai-lang CLI Demo ==="
echo

echo "1. Show version:"
PYTHONPATH=src python3 src/ai_lang/cli.py --version
echo

echo "2. Type check only:"
PYTHONPATH=src python3 src/ai_lang/cli.py examples/minimal.ai --type-check-only
echo

echo "3. Run with timing:"
PYTHONPATH=src python3 src/ai_lang/cli.py examples/minimal.ai --timing
echo

echo "4. Show AST:"
PYTHONPATH=src python3 src/ai_lang/cli.py examples/minimal.ai --ast | head -20
echo "..."
echo

echo "5. Run with verbose output:"
echo "data Nat : Type where
  Z : Nat  
  S : Nat -> Nat

main : Nat
main = S Z" > /tmp/simple.ai

PYTHONPATH=src python3 src/ai_lang/cli.py /tmp/simple.ai --verbose
echo

echo "6. Save output to file:"
PYTHONPATH=src python3 src/ai_lang/cli.py /tmp/simple.ai --output /tmp/result.txt
cat /tmp/result.txt
echo

echo "7. Interactive REPL (with sample commands):"
echo "zero : Nat
zero = Z
:type zero
:list
:quit" | PYTHONPATH=src python3 src/ai_lang/cli.py

# Cleanup
rm -f /tmp/simple.ai /tmp/result.txt