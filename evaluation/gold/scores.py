"""Typed score records: every score carries the system's model/revision and the corpus revision."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from evaluation.gold.manifest import GoldCorpus

SCORE_SCHEMA_VERSION = 1


class ScoreRecordError(ValueError):
    pass


@dataclass(frozen=True)
class SystemInfo:
    """The system that produced predictions.

    ``backend`` names the adapter (for example ``stanza-sidecar``), ``model`` the
    model id and ``revision`` its pinned revision. ``details`` holds backend
    specifics such as the Stanza version or model packages.
    """

    backend: str
    model: str
    revision: str
    details: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "SystemInfo":
        for key in ("backend", "model", "revision"):
            if not isinstance(value.get(key), str) or not value[key]:
                raise ScoreRecordError(f"system.{key} must be a non-empty string")
        details = value.get("details", {})
        if not isinstance(details, dict):
            raise ScoreRecordError("system.details must be an object")
        return cls(value["backend"], value["model"], value["revision"], details)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model,
            "revision": self.revision,
            "details": dict(self.details),
        }


def load_systems(path: Path) -> Dict[str, SystemInfo]:
    """Reads a ``{language: SystemInfo}`` JSON file written by a backend adapter."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ScoreRecordError(f"{path}: expected an object keyed by language")
    return {language: SystemInfo.from_json(value) for language, value in data.items()}


@dataclass(frozen=True)
class ScoreRecord:
    task: str
    language: str
    corpora: Sequence[GoldCorpus]
    system: SystemInfo
    result: Mapping[str, Any]
    source_revision: Optional[str] = None
    measured_at: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        value: Dict[str, Any] = {
            "schemaVersion": SCORE_SCHEMA_VERSION,
            "task": self.task,
            "language": self.language,
            "corpora": [corpus.revision() for corpus in self.corpora],
            "system": self.system.as_dict(),
        }
        value.update(self.result)
        if self.source_revision is not None:
            value["sourceRevision"] = self.source_revision
        if self.measured_at is not None:
            value["measuredAt"] = self.measured_at
        return value


def read_report(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schemaVersion") != SCORE_SCHEMA_VERSION:
        raise ScoreRecordError(f"{path}: unsupported schemaVersion")
    for key in ("task", "language", "corpora", "system", "metrics"):
        if key not in value:
            raise ScoreRecordError(f"{path}: missing {key}")
    SystemInfo.from_json(value["system"])
    return value


def metric_value(report: Mapping[str, Any], metric: str) -> float:
    """Reads ``metric`` (for example ``las`` or ``entities``) as an F1, or a plain number."""
    value = report["metrics"].get(metric)
    if isinstance(value, Mapping):
        value = value.get("f1")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScoreRecordError(f"report has no numeric metric {metric!r}")
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ScoreRecordError(f"metric {metric!r} must be a finite score in [0, 1], got {value!r}")
    return float(value)


TABLE_METRICS: Dict[str, Sequence[str]] = {
    "syntax": ("upos", "ufeats", "lemma", "uas", "las"),
    "entities": ("entities",),
    "retrieval": ("mrrAt10", "recallAt1", "ndcgAt10"),
}


def render_table(reports: Sequence[Mapping[str, Any]]) -> str:
    """A Markdown table per task: one row per system/corpus, one column per metric."""
    lines: List[str] = []
    for task, metrics in TABLE_METRICS.items():
        rows = [report for report in reports if report["task"] == task]
        if not rows:
            continue
        rows.sort(key=lambda report: (report["system"]["backend"], report["language"], _corpus_label(report)))
        lines.append(f"### {task}")
        lines.append("")
        lines.append("| backend | model @ revision | lang | corpus | " + " | ".join(metrics) + " |")
        lines.append("| --- | --- | --- | --- | " + " | ".join("---" for _ in metrics) + " |")
        for report in rows:
            system = report["system"]
            cells = [_cell(report, metric) for metric in metrics]
            lines.append(
                f"| {system['backend']} | {system['model']} @ {system['revision'][:8]} | "
                f"{report['language']} | {_corpus_label(report)} | " + " | ".join(cells) + " |"
            )
        lines.append("")
    return "\n".join(lines)


def _cell(report: Mapping[str, Any], metric: str) -> str:
    try:
        return f"{metric_value(report, metric):.3f}"
    except ScoreRecordError:
        return "–"


def _corpus_label(report: Mapping[str, Any]) -> str:
    return " → ".join(corpus["id"] for corpus in report["corpora"])
