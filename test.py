from nist_suite import NISTTestSuite
import random

suite = NISTTestSuite()

bits = [random.randint(0, 1) for _ in range(1000)]
passed, total = suite.run(bits, verbose=True)
print(f"\n{passed}/{total} passed")
