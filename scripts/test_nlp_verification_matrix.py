import copy
import unittest
from pathlib import Path

from check_nlp_verification_matrix import DEFAULT_MATRIX, load_matrix, validate_matrix


class NlpVerificationMatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix = load_matrix(DEFAULT_MATRIX)
        self.root = Path(__file__).resolve().parents[1]

    def test_repository_matrix_is_structurally_valid(self) -> None:
        self.assertEqual([], validate_matrix(self.matrix, self.root))

    def test_missing_evidence_is_descriptive_not_a_failure(self) -> None:
        candidate = copy.deepcopy(self.matrix)
        evidence = candidate["capabilities"][0]["evidence"]["coverage"]
        evidence["status"] = "missing"
        evidence["paths"] = []
        evidence["commands"] = []
        evidence["note"] = "Coverage has not been measured yet."

        self.assertEqual([], validate_matrix(candidate, self.root))

    def test_unknown_status_is_rejected(self) -> None:
        candidate = copy.deepcopy(self.matrix)
        candidate["capabilities"][0]["evidence"]["tests"]["status"] = "green"

        errors = validate_matrix(candidate, self.root)

        self.assertTrue(any("status must be one of" in error for error in errors))

    def test_duplicate_capability_ids_are_rejected(self) -> None:
        candidate = copy.deepcopy(self.matrix)
        duplicate = copy.deepcopy(candidate["capabilities"][0])
        candidate["capabilities"].append(duplicate)

        errors = validate_matrix(candidate, self.root)

        self.assertTrue(any("duplicate capability id" in error for error in errors))

    def test_present_or_partial_evidence_needs_a_reproducible_anchor(self) -> None:
        candidate = copy.deepcopy(self.matrix)
        evidence = candidate["capabilities"][0]["evidence"]["tests"]
        evidence["paths"] = []
        evidence["commands"] = []

        errors = validate_matrix(candidate, self.root)

        self.assertTrue(any("needs at least one path or command" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
