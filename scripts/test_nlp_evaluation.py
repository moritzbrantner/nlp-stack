import json
from pathlib import Path
import unittest

from evaluation.metrics import (
    accuracy,
    macro_f1,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_recall_f1,
    recall_at_k,
    spearman_correlation,
)
from evaluation.runner import boundary_report, load_jsonl


ROOT = Path(__file__).resolve().parents[1]


class NlpEvaluationTests(unittest.TestCase):
    def test_precision_recall_f1_counts_duplicate_events(self) -> None:
        metrics = precision_recall_f1(["a", "a", "b"], ["a", "b", "b"])
        self.assertEqual(metrics.true_positives, 2)
        self.assertEqual(metrics.false_positives, 1)
        self.assertEqual(metrics.false_negatives, 1)
        self.assertAlmostEqual(metrics.f1, 2 / 3)

    def test_classification_metrics_cover_accuracy_and_macro_f1(self) -> None:
        gold = ["person", "person", "location"]
        predicted = ["person", "location", "location"]
        self.assertAlmostEqual(accuracy(gold, predicted), 2 / 3)
        self.assertAlmostEqual(macro_f1(gold, predicted), 2 / 3)

    def test_retrieval_metrics_are_rank_sensitive(self) -> None:
        self.assertAlmostEqual(
            mean_reciprocal_rank(
                [["noise", "target"], ["target", "noise"]],
                [{"target"}, {"target"}],
            ),
            0.75,
        )
        self.assertEqual(recall_at_k(["a", "b", "c"], {"b", "d"}, 2), 0.5)
        self.assertGreater(
            ndcg_at_k(["a", "b"], {"a", "b"}, 2),
            ndcg_at_k(["x", "a"], {"a", "b"}, 2),
        )

    def test_spearman_correlation_handles_direction(self) -> None:
        self.assertAlmostEqual(
            spearman_correlation([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
            1.0,
        )
        self.assertAlmostEqual(
            spearman_correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]),
            -1.0,
        )

    def test_boundary_report_aggregates_case_scoped_events(self) -> None:
        gold = [
            {"id": "a", "boundaryByteEnds": [4, 9]},
            {"id": "b", "boundaryByteEnds": [5]},
        ]
        predictions = [
            {"id": "a", "boundaryByteEnds": [4, 8]},
            {"id": "b", "boundaryByteEnds": [5]},
        ]
        report = boundary_report(gold, predictions, "suite-v1", "test-system")
        self.assertEqual(report["metrics"]["truePositives"], 2)
        self.assertEqual(report["metrics"]["falsePositives"], 1)
        self.assertEqual(report["metrics"]["falseNegatives"], 1)
        self.assertAlmostEqual(report["metrics"]["f1"], 2 / 3)

    def test_committed_sentence_boundary_baseline_is_reproducible(self) -> None:
        corpus = load_jsonl(ROOT / "evaluation/corpora/sentence-boundaries-v1.jsonl")
        predictions = load_jsonl(
            ROOT / "evaluation/baselines/sentence-boundaries-v1.predictions.jsonl"
        )
        expected = json.loads(
            (
                ROOT / "evaluation/baselines/text-core-sentence-boundaries-v1.json"
            ).read_text(encoding="utf-8")
        )
        actual = boundary_report(
            corpus,
            predictions,
            suite="sentence-boundaries-v1",
            system="text-core/builtin-sentence-boundaries",
            source_revision=expected["sourceRevision"],
        )
        self.assertEqual(actual["schemaVersion"], expected["schemaVersion"])
        self.assertEqual(actual["sourceRevision"], expected["sourceRevision"])
        self.assertEqual(actual["suite"], expected["suite"])
        self.assertEqual(actual["system"], expected["system"])
        self.assertEqual(actual["metrics"], expected["metrics"])


if __name__ == "__main__":
    unittest.main()
