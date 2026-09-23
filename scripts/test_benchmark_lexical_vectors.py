"""Fail-closed evidence parser tests; synthetic timings are never evidence."""
import copy
import json
import unittest
from benchmark_lexical_vectors import CONTROLS, FIXED, PREFIX, parse_samples, validate_pair, validate_tests


class LexicalVectorEvidenceTests(unittest.TestCase):
    def log(self, before):
        return "\n".join(f"test {PREFIX}{name} ... {state}" for name, state in {
            **{name: "FAILED" if before else "ok" for name in FIXED},
            **{name: "ok" for name in CONTROLS}, "query_only_benchmark": "ignored",
        }.items())

    def test_requires_named_baseline_failures_and_positive_controls(self):
        self.assertEqual(len(validate_tests(self.log(True), 101, True)), 8)
        self.assertEqual(len(validate_tests(self.log(False), 0, False)), 8)
        for text, code, before in [("compile failed", 101, True), (self.log(True), 1, True),
                                   (self.log(False), 0, True), (self.log(True), 101, False)]:
            with self.assertRaises(ValueError):
                validate_tests(text, code, before)

    def test_requires_complete_unique_finite_samples(self):
        lines = ["NLP_VECTOR_BENCH " + json.dumps({"kind": kind, "documents": count,
                    "dimensions": dims, "ns_per_query": 1})
                 for kind in ("lexical", "fuzzy-one", "fuzzy-two")
                 for count in (32, 256) for dims in (128, 1024)]
        self.assertEqual(len(parse_samples("\n".join(lines))), 12)
        for invalid in [lines[:-1], lines + [lines[0]], [line.replace('"ns_per_query": 1', '"ns_per_query": 0') for line in lines]]:
            with self.assertRaises(ValueError):
                parse_samples("\n".join(invalid))

    def test_result_parity_and_zero_vector_work_are_independent_of_timing(self):
        before = {"kind": "lexical", "documents": 32, "dimensions": 128,
                  "query": "strategy", "variants": 1, "iterations": 10,
                  "results": [{"score": 1}], "work": {"vector_reads": 1, "vector_scalars": 4096}}
        after = copy.deepcopy(before)
        after["work"] = {"vector_reads": 0, "vector_scalars": 0}
        validate_pair(before, after)
        wrong = copy.deepcopy(after)
        wrong["results"][0]["score"] = 0
        with self.assertRaises(ValueError):
            validate_pair(before, wrong)
        with self.assertRaises(ValueError):
            validate_pair(before, before)
