"""Model-default gates: a backend may be a default only if its recorded scores meet thresholds.

``gates.json`` lists, per capability and backend, the pinned model per language, the
threshold status (``proposed`` until the owner agrees, then ``agreed``) and one
requirement per (language, corpus, metric). Each requirement names the committed
score report that holds its measurement. ``check_gates`` verifies that the report
exists, was measured on the pinned corpus revision with the gated model and
revision, and meets the threshold.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from evaluation.gold.manifest import GoldCorpus, load_manifest
from evaluation.gold.scores import metric_value, read_report

GATES_PATH = Path(__file__).with_name("gates.json")
STATUSES = ("proposed", "agreed")


class GateError(ValueError):
    pass


@dataclass(frozen=True)
class Requirement:
    language: str
    corpus: str
    metric: str
    threshold: float
    report: str

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "Requirement":
        try:
            requirement = cls(
                language=str(value["language"]),
                corpus=str(value["corpus"]),
                metric=str(value["metric"]),
                threshold=float(value["threshold"]),
                report=str(value["report"]),
            )
        except KeyError as error:
            raise GateError(f"requirement missing {error}") from None
        if not 0.0 <= requirement.threshold <= 1.0:
            raise GateError(f"threshold out of range: {requirement}")
        return requirement


@dataclass(frozen=True)
class ModelPin:
    model: str
    revision: str


@dataclass(frozen=True)
class Gate:
    id: str
    capability: str
    backend: str
    models: Mapping[str, ModelPin]
    status: str
    requirements: Sequence[Requirement]

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "Gate":
        gate = cls(
            id=str(value["id"]),
            capability=str(value["capability"]),
            backend=str(value["backend"]),
            models={
                language: ModelPin(str(pin["model"]), str(pin["revision"]))
                for language, pin in value["models"].items()
            },
            status=str(value["status"]),
            requirements=[Requirement.from_json(item) for item in value["requirements"]],
        )
        if gate.status not in STATUSES:
            raise GateError(f"{gate.id}: status must be one of {STATUSES}")
        if not gate.requirements:
            raise GateError(f"{gate.id}: no requirements")
        for requirement in gate.requirements:
            if requirement.language not in gate.models:
                raise GateError(f"{gate.id}: no pinned model for {requirement.language}")
        return gate


@dataclass(frozen=True)
class RequirementResult:
    gate: str
    requirement: Requirement
    score: Optional[float]
    passed: bool
    problem: Optional[str] = None


def load_gates(path: Path = GATES_PATH) -> List[Gate]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schemaVersion") != 1:
        raise GateError(f"{path}: unsupported schemaVersion")
    gates = [Gate.from_json(item) for item in data["gates"]]
    if len({gate.id for gate in gates}) != len(gates):
        raise GateError(f"{path}: duplicate gate ids")
    return gates


def _model_matches(gate: Gate, report: Mapping[str, Any]) -> bool:
    system = report["system"]
    pin = gate.models[report["language"]]
    return (
        system["backend"] == gate.backend
        and system["model"] == pin.model
        and system["revision"] == pin.revision
    )


def check_requirement(
    gate: Gate, requirement: Requirement, base: Path, corpora: Mapping[str, GoldCorpus]
) -> RequirementResult:
    def failed(problem: str, score: Optional[float] = None) -> RequirementResult:
        return RequirementResult(gate.id, requirement, score, False, problem)

    path = base / requirement.report
    if not path.is_file():
        return failed(f"report {requirement.report} not found")
    report = read_report(path)
    if report["language"] != requirement.language:
        return failed(f"report language {report['language']} != {requirement.language}")
    corpus_ids = [corpus["id"] for corpus in report["corpora"]]
    if requirement.corpus not in corpus_ids:
        return failed(f"report corpora {corpus_ids} do not include {requirement.corpus}")
    for recorded in report["corpora"]:
        pinned = corpora.get(recorded["id"])
        if pinned is None or recorded["sha256"] != pinned.sha256:
            return failed(f"report measured {recorded['id']} at a revision that is no longer pinned")
    if not _model_matches(gate, report):
        system = report["system"]
        return failed(f"report system {system['backend']} {system['model']}@{system['revision']} is not the gated model")
    score = metric_value(report, requirement.metric)
    if score < requirement.threshold:
        return failed(f"{requirement.metric} {score:.4f} < {requirement.threshold}", score)
    return RequirementResult(gate.id, requirement, score, True)


def check_gates(
    gates: Sequence[Gate], base: Path, corpora: Optional[Sequence[GoldCorpus]] = None
) -> List[RequirementResult]:
    by_id = {corpus.id: corpus for corpus in (corpora if corpora is not None else load_manifest())}
    return [
        check_requirement(gate, requirement, base, by_id)
        for gate in gates
        for requirement in gate.requirements
    ]
