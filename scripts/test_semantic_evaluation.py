import unittest

from evaluation.runner import semantic_similarity_report, topic_shift_report


class SemanticEvaluationTests(unittest.TestCase):
    def test_semantic_similarity_report_scores_perfect_ranking(self):
        gold = [
            {"id": "a", "similarity": 1.0},
            {"id": "b", "similarity": 0.5},
            {"id": "c", "similarity": 0.0},
        ]
        predicted = [
            {"id": "a", "similarity": 0.9},
            {"id": "b", "similarity": 0.4},
            {"id": "c", "similarity": 0.1},
        ]

        report = semantic_similarity_report(gold, predicted, "semantic-smoke", "fixture")

        self.assertEqual(report["suite"]["task"], "semantic-textual-similarity")
        self.assertEqual(report["metrics"]["spearman"], 1.0)
        self.assertGreater(report["metrics"]["meanAbsoluteError"], 0.0)
        self.assertNotIn("groups", report)

    def test_semantic_similarity_report_groups_by_gold_metadata(self):
        gold = [
            {"id": "en-a", "group": "monolingual-en", "similarity": 1.0},
            {"id": "en-b", "group": "monolingual-en", "similarity": 0.2},
            {"id": "cross-a", "group": "cross-en-de", "similarity": 0.9},
            {"id": "ungrouped", "similarity": 0.4},
        ]
        predicted = [
            {"id": "en-a", "group": "ignored", "similarity": 0.8},
            {"id": "en-b", "group": "ignored", "similarity": 0.3},
            {"id": "cross-a", "group": "ignored", "similarity": 0.7},
            {"id": "ungrouped", "group": "ignored", "similarity": 0.5},
        ]

        report = semantic_similarity_report(gold, predicted, "semantic-groups", "fixture")

        self.assertEqual([group["group"] for group in report["groups"]], ["cross-en-de", "monolingual-en"])
        self.assertEqual(report["groups"][0]["caseCount"], 1)
        self.assertEqual(report["groups"][1]["caseCount"], 2)
        self.assertEqual(report["groups"][1]["metrics"]["spearman"], 1.0)
        self.assertEqual(report["suite"]["caseCount"], 4)
        self.assertNotIn("group", next(case for case in report["cases"] if case["id"] == "ungrouped"))

    def test_semantic_similarity_rejects_empty_group(self):
        with self.assertRaisesRegex(ValueError, "group must be a non-empty string"):
            semantic_similarity_report(
                [{"id": "a", "group": "  ", "similarity": 1.0}],
                [{"id": "a", "similarity": 1.0}],
                "semantic-groups",
                "fixture",
            )

    def test_topic_shift_report_uses_explicit_shift_indices(self):
        gold = [
            {"id": "a", "shiftIndices": [2, 4]},
            {"id": "b", "shiftIndices": []},
        ]
        predicted = [
            {"id": "a", "shiftIndices": [2]},
            {"id": "b", "shiftIndices": []},
        ]

        report = topic_shift_report(gold, predicted, "topic-shift-smoke", "fixture")

        self.assertEqual(report["suite"]["task"], "topic-shifts")
        self.assertEqual(report["metrics"]["truePositives"], 1)
        self.assertEqual(report["metrics"]["falseNegatives"], 1)
        self.assertEqual(report["metrics"]["falsePositives"], 0)

    def test_topic_shift_report_exposes_group_metrics_without_changing_aggregate(self):
        gold = [
            {"id": "en", "group": "monolingual-en", "shiftIndices": [2, 4]},
            {"id": "de", "group": "monolingual-de", "shiftIndices": [1]},
            {"id": "switch", "group": "language-switch", "shiftIndices": [1, 3]},
            {"id": "plain", "shiftIndices": []},
        ]
        predicted = [
            {"id": "en", "shiftIndices": [2]},
            {"id": "de", "shiftIndices": [1]},
            {"id": "switch", "shiftIndices": [1, 3, 4]},
            {"id": "plain", "shiftIndices": []},
        ]

        report = topic_shift_report(gold, predicted, "topic-shift-groups", "fixture")

        self.assertEqual(report["metrics"]["truePositives"], 4)
        self.assertEqual(report["metrics"]["falseNegatives"], 1)
        self.assertEqual(report["metrics"]["falsePositives"], 1)
        groups = {group["group"]: group for group in report["groups"]}
        self.assertEqual(groups["monolingual-de"]["metrics"]["f1"], 1.0)
        self.assertEqual(groups["language-switch"]["metrics"]["falsePositives"], 1)
        self.assertEqual(groups["monolingual-en"]["metrics"]["falseNegatives"], 1)
        self.assertNotIn("group", next(case for case in report["cases"] if case["id"] == "plain"))


if __name__ == "__main__":
    unittest.main()
