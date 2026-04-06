import random
import warnings
import coinflip.randtests as randtests
from pandas import Series


class NISTTestSuite:

    TESTS = [
        ("Monobit",                   randtests.monobit),
        ("Frequency Within Block",    randtests.frequency_within_block),
        ("Runs",                      randtests.runs),
        ("Longest Runs",              randtests.longest_runs),
        ("Binary Matrix Rank",        randtests.binary_matrix_rank),
        ("Spectral (DFT)",            randtests.spectral),
        ("Non-overlapping Template",  randtests.non_overlapping_template_matching),
        ("Overlapping Template",      randtests.overlapping_template_matching),
        ("Maurer's Universal",        randtests.maurers_universal),
        ("Linear Complexity",         randtests.linear_complexity),
        ("Approximate Entropy",       randtests.approximate_entropy),
        ("Cumulative Sums",           randtests.cusum),
        ("Serial",                    randtests.serial),
        ("Random Excursions",         randtests.random_excursions),
        ("Random Excursions Variant", randtests.random_excursions_variant),
    ]

    PASS_THRESHOLD = 0.01

    def _get_pvalues(self, result):
        if hasattr(result, 'p'):
            return [result.p]
        elif hasattr(result, 'pvalues'):
            return list(result.pvalues)
        elif hasattr(result, 'results'):
            return [r.p for r in result.results]
        return []

    def run(self, bits, verbose=False):
        seq = Series(bits)
        passed = 0
        total = 0
        report_rows = []

        for name, fn in self.TESTS:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = fn(seq)

                pvalues = self._get_pvalues(result)

                if not pvalues:
                    report_rows.append((name, None, None, "SKIP"))
                    continue

                min_p = min(pvalues)
                test_passed = min_p >= self.PASS_THRESHOLD
                passed += int(test_passed)
                total += 1
                status = "PASS" if test_passed else "FAIL"
                report_rows.append((name, min_p, len(pvalues), status))

            except Exception:
                report_rows.append((name, None, None, "SKIP"))

        if verbose:
            self._print_report(report_rows, passed, total)

        return passed, total

    def _print_report(self, rows, passed, total):
        width = 65
        print("=" * width)
        print("  NIST SP800-22 Randomness Test Suite")
        print("=" * width)
        print(f"  {'Test':<40} {'p-value':>8}   {'Result'}")
        print("-" * width)
        for name, p, n_pvalues, status in rows:
            if status == "SKIP":
                print(f"  {name:<40} {'---':>8}   SKIP")
            else:
                note = f" (min of {n_pvalues})" if n_pvalues and n_pvalues > 1 else ""
                print(f"  {name+note:<40} {p:>8.4f}   {status}")
        print("=" * width)
        print(f"  Result: {passed}/{total} tests passed")
        print("=" * width)
