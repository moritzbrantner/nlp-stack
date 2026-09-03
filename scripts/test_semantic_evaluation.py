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


if __name__ == "__main__":
    unittest.main()
