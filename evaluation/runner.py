from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from evaluation.metrics import precision_recall_f1, spearman_correlation


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen_ids = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        case_id = value.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"{path}:{line_number}: id must be a non-empty string")
        if case_id in seen_ids:
            raise ValueError(f"{path}:{line_number}: duplicate id {case_id}")
        seen_ids.add(case_id)
        records.append(value)
    return records


def _indexed(records: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    return {str(record["id"]): record for record in records}


def _require_matching_ids(
    gold_records: Sequence[Mapping[str, Any]],
    prediction_records: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Mapping[str, Any]], Dict[str, Mapping[str, Any]]]:
    gold_by_id = _indexed(gold_records)
    predictions_by_id = _indexed(prediction_records)
    if set(gold_by_id) != set(predictions_by_id):
        missing = sorted(set(gold_by_id) - set(predictions_by_id))
        extra = sorted(set(predictions_by_id) - set(gold_by_id))
        raise ValueError(
            f"prediction ids differ from corpus ids; missing={missing}, extra={extra}"
        )
    return gold_by_id, predictions_by_id


def _integer_list(record: Mapping[str, Any], field: str) -> List[int]:
    value = record.get(field)
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value
    ):
        raise ValueError(
            f"{record.get('id', '<unknown>')}: {field} must be a list of non-negative integers"
        )
    return list(value)


def _float_value(record: Mapping[str, Any], field: str) -> float:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{record.get('id', '<unknown>')}: {field} must be a number"
        )
    return float(value)


def _group_value(record: Mapping[str, Any]) -> Optional[str]:
    value = record.get("group")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{record.get('id', '<unknown>')}: group must be a non-empty string when present"
        )
    return value.strip()


def _semantic_similarity_metrics(
    gold_scores: Sequence[float], predicted_scores: Sequence[float]
) -> Dict[str, float]:
    if len(gold_scores) != len(predicted_scores):
        raise ValueError("semantic similarity metrics require equal-length score lists")
    mean_absolute_error = (
        sum(abs(gold - predicted) for gold, predicted in zip(gold_scores, predicted_scores))
        / len(gold_scores)
        if gold_scores
        else 0.0
    )
    return {
        "spearman": spearman_correlation(gold_scores, predicted_scores),
        "meanAbsoluteError": mean_absolute_error,
    }


def boundary_report(
    gold_records: Sequence[Mapping[str, Any]],
    prediction_records: Sequence[Mapping[str, Any]],
    suite: str,
    system: str,
    source_revision: Optional[str] = None,
) -> Dict[str, Any]:
    gold_by_id, predictions_by_id = _require_matching_ids(
        gold_records, prediction_records
    )

    gold_events: List[Tuple[str, int]] = []
    predicted_events: List[Tuple[str, int]] = []
    cases: List[Dict[str, Any]] = []

    for case_id in gold_by_id:
        gold_ends = _integer_list(gold_by_id[case_id], "boundaryByteEnds")
        predicted_ends = _integer_list(predictions_by_id[case_id], "boundaryByteEnds")
        case_metrics = precision_recall_f1(gold_ends, predicted_ends)
        gold_events.extend((case_id, end) for end in gold_ends)
        predicted_events.extend((case_id, end) for end in predicted_ends)
        cases.append(
            {
                "id": case_id,
                "metrics": case_metrics.as_dict(),
                "goldBoundaryByteEnds": gold_ends,
                "predictedBoundaryByteEnds": predicted_ends,
            }
        )

    result: Dict[str, Any] = {
        "schemaVersion": 1,
        "suite": {
            "id": suite,
            "task": "sentence-boundaries",
            "caseCount": len(gold_records),
        },
        "system": system,
        "metrics": precision_recall_f1(gold_events, predicted_events).as_dict(),
        "cases": cases,
    }
    if source_revision is not None:
        result["sourceRevision"] = source_revision
    return result


def semantic_similarity_report(
    gold_records: Sequence[Mapping[str, Any]],
    prediction_records: Sequence[Mapping[str, Any]],
    suite: str,
    system: str,
    source_revision: Optional[str] = None,
) -> Dict[str, Any]:
    gold_by_id, predictions_by_id = _require_matching_ids(
        gold_records, prediction_records
    )
    gold_scores: List[float] = []
    predicted_scores: List[float] = []
    cases: List[Dict[str, Any]] = []
    grouped_scores: Dict[str, Dict[str, List[float]]] = {}

    for case_id in gold_by_id:
        gold_record = gold_by_id[case_id]
        gold_score = _float_value(gold_record, "similarity")
        predicted_score = _float_value(predictions_by_id[case_id], "similarity")
        group = _group_value(gold_record)
        gold_scores.append(gold_score)
        predicted_scores.append(predicted_score)
        case: Dict[str, Any] = {
            "id": case_id,
            "goldSimilarity": gold_score,
            "predictedSimilarity": predicted_score,
            "absoluteError": abs(gold_score - predicted_score),
        }
        if group is not None:
            case["group"] = group
            grouped = grouped_scores.setdefault(group, {"gold": [], "predicted": []})
            grouped["gold"].append(gold_score)
            grouped["predicted"].append(predicted_score)
        cases.append(case)

    result: Dict[str, Any] = {
        "schemaVersion": 1,
        "suite": {
            "id": suite,
            "task": "semantic-textual-similarity",
            "caseCount": len(gold_records),
        },
        "system": system,
        "metrics": _semantic_similarity_metrics(gold_scores, predicted_scores),
        "cases": cases,
    }
    if grouped_scores:
        result["groups"] = [
            {
                "group": group,
                "caseCount": len(grouped_scores[group]["gold"]),
                "metrics": _semantic_similarity_metrics(
                    grouped_scores[group]["gold"], grouped_scores[group]["predicted"]
                ),
            }
            for group in sorted(grouped_scores)
        ]
    if source_revision is not None:
        result["sourceRevision"] = source_revision
    return result


def topic_shift_report(
    gold_records: Sequence[Mapping[str, Any]],
    prediction_records: Sequence[Mapping[str, Any]],
    suite: str,
    system: str,
    source_revision: Optional[str] = None,
) -> Dict[str, Any]:
    gold_by_id, predictions_by_id = _require_matching_ids(
        gold_records, prediction_records
    )
    gold_events: List[Tuple[str, int]] = []
    predicted_events: List[Tuple[str, int]] = []
    cases: List[Dict[str, Any]] = []
    grouped_events: Dict[str, Dict[str, List[Tuple[str, int]]]] = {}
    group_case_counts: Dict[str, int] = {}

    for case_id in gold_by_id:
        gold_record = gold_by_id[case_id]
        gold_indices = _integer_list(gold_record, "shiftIndices")
        predicted_indices = _integer_list(predictions_by_id[case_id], "shiftIndices")
        group = _group_value(gold_record)
        case_metrics = precision_recall_f1(gold_indices, predicted_indices)
        gold_events.extend((case_id, end) for end in gold_indices)
        predicted_events.extend((case_id, end) for end in predicted_indices)
        case: Dict[str, Any] = {
            "id": case_id,
            "metrics": case_metrics.as_dict(),
            "goldShiftIndices": gold_indices,
            "predictedShiftIndices": predicted_indices,
        }
        if group is not None:
            case["group"] = group
            grouped = grouped_events.setdefault(group, {"gold": [], "predicted": []})
            grouped["gold"].extend((case_id, index) for index in gold_indices)
            grouped["predicted"].extend((case_id, index) for index in predicted_indices)
            group_case_counts[group] = group_case_counts.get(group, 0) + 1
        cases.append(case)

    result: Dict[str, Any] = {
        "schemaVersion": 1,
        "suite": {
            "id": suite,
            "task": "topic-shifts",
            "caseCount": len(gold_records),
        },
        "system": system,
        "metrics": precision_recall_f1(gold_events, predicted_events).as_dict(),
        "cases": cases,
    }
    if grouped_events:
        result["groups"] = [
            {
                "group": group,
                "caseCount": group_case_counts[group],
                "metrics": precision_recall_f1(
                    grouped_events[group]["gold"], grouped_events[group]["predicted"]
                ).as_dict(),
            }
            for group in sorted(grouped_events)
        ]
    if source_revision is not None:
        result["sourceRevision"] = source_revision
    return result


def write_report(report: Mapping[str, Any], path: Optional[Path]) -> None:
    rendered = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path is None:
        print(rendered, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def _add_common_report_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--source-revision")
    parser.add_argument("--output", type=Path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic NLP evaluation suites.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    boundaries = subparsers.add_parser("boundaries")
    _add_common_report_arguments(boundaries)
    boundaries.add_argument("--min-f1", type=float)

    semantic_similarity = subparsers.add_parser("semantic-similarity")
    _add_common_report_arguments(semantic_similarity)

    topic_shifts = subparsers.add_parser("topic-shifts")
    _add_common_report_arguments(topic_shifts)

    args = parser.parse_args(argv)
    if args.command == "boundaries":
        report = boundary_report(
            load_jsonl(args.gold),
            load_jsonl(args.predictions),
            suite=args.suite,
            system=args.system,
            source_revision=args.source_revision,
        )
        write_report(report, args.output)
        if args.min_f1 is not None and float(report["metrics"]["f1"]) < args.min_f1:
            return 1
        return 0
    if args.command == "semantic-similarity":
        report = semantic_similarity_report(
            load_jsonl(args.gold),
            load_jsonl(args.predictions),
            suite=args.suite,
            system=args.system,
            source_revision=args.source_revision,
        )
        write_report(report, args.output)
        return 0
    if args.command == "topic-shifts":
        report = topic_shift_report(
            load_jsonl(args.gold),
            load_jsonl(args.predictions),
            suite=args.suite,
            system=args.system,
            source_revision=args.source_revision,
        )
        write_report(report, args.output)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
