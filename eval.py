import sys
import ast
from nist_suite import NISTTestSuite

def run_test_on_file(filepath):
    try:
        with open(filepath, 'r') as f: content = f.read().strip()
        
        bits = ast.literal_eval(content)
        
        if not isinstance(bits, list):
            print("Error: File content is not a list.")
            return

        print(f"Loaded {len(bits)} bits from {filepath}")

        suite = NISTTestSuite()
        passed, total = suite.run(bits, verbose=True)
        
        print(f"\n{passed}/{total} passed")

    except FileNotFoundError: print(f"Error: File '{filepath}' not found.")
    except Exception as e: print(f"Error parsing file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2: print("Usage: python3 test_file.py <path_to_bit_file>")
    else: run_test_on_file(sys.argv[1])
