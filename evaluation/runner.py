from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from evaluation.metrics import precision_recall_f1


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


def _integer_list(record: Mapping[str, Any], field: str) -> List[int]:
    value = record.get(field)
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value
    ):
        raise ValueError(f"{record.get('id', '<unknown>')}: {field} must be a list of non-negative integers")
    return list(value)


def boundary_report(
    gold_records: Sequence[Mapping[str, Any]],
    prediction_records: Sequence[Mapping[str, Any]],
    suite: str,
    system: str,
    source_revision: Optional[str] = None,
) -> Dict[str, Any]:
    gold_by_id = _indexed(gold_records)
    predictions_by_id = _indexed(prediction_records)
    if set(gold_by_id) != set(predictions_by_id):
        missing = sorted(set(gold_by_id) - set(predictions_by_id))
        extra = sorted(set(predictions_by_id) - set(gold_by_id))
        raise ValueError(f"prediction ids differ from corpus ids; missing={missing}, extra={extra}")

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


def write_report(report: Mapping[str, Any], path: Optional[Path]) -> None:
    rendered = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path is None:
        print(rendered, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic NLP evaluation suites.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    boundaries = subparsers.add_parser("boundaries")
    boundaries.add_argument("--gold", type=Path, required=True)
    boundaries.add_argument("--predictions", type=Path, required=True)
    boundaries.add_argument("--suite", required=True)
    boundaries.add_argument("--system", required=True)
    boundaries.add_argument("--source-revision")
    boundaries.add_argument("--output", type=Path)
    boundaries.add_argument("--min-f1", type=float)

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
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
