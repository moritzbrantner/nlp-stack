"""Pinned external gold corpora: manifest, digest-verified fetch and lookup."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

MANIFEST_PATH = Path(__file__).with_name("manifest.json")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GOLD_DIR = REPOSITORY_ROOT / ".test-corpora" / "gold"

TASKS = ("syntax", "entities", "retrieval")
FORMATS = ("conllu", "conll02", "uner-iob2", "xquad")


class GoldCorpusError(Exception):
    """A gold corpus is unknown, missing, or does not match its pinned digest."""


@dataclass(frozen=True)
class GoldCorpus:
    id: str
    task: str
    language: str
    format: str
    split: str
    source: str
    source_revision: str
    url: str
    file: str
    bytes: int
    sha256: str
    license: str
    license_url: str

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "GoldCorpus":
        corpus = cls(
            id=_text(value, "id"),
            task=_text(value, "task"),
            language=_text(value, "language"),
            format=_text(value, "format"),
            split=_text(value, "split"),
            source=_text(value, "source"),
            source_revision=_text(value, "sourceRevision"),
            url=_text(value, "url"),
            file=_text(value, "file"),
            bytes=int(value["bytes"]),
            sha256=_text(value, "sha256"),
            license=_text(value, "license"),
            license_url=_text(value, "licenseUrl"),
        )
        if corpus.task not in TASKS:
            raise GoldCorpusError(f"{corpus.id}: unknown task {corpus.task!r}")
        if corpus.format not in FORMATS:
            raise GoldCorpusError(f"{corpus.id}: unknown format {corpus.format!r}")
        if len(corpus.sha256) != 64 or len(corpus.source_revision) != 40:
            raise GoldCorpusError(f"{corpus.id}: sha256 and sourceRevision must be full digests")
        if corpus.source_revision not in corpus.url:
            raise GoldCorpusError(f"{corpus.id}: url must be pinned to sourceRevision")
        if "/" in corpus.file or corpus.file.startswith("."):
            raise GoldCorpusError(f"{corpus.id}: file must be a plain file name")
        return corpus

    def revision(self) -> Dict[str, str]:
        """The corpus identity recorded with every score."""
        return {"id": self.id, "sourceRevision": self.source_revision, "sha256": self.sha256}

    def path(self, gold_dir: Path) -> Path:
        return gold_dir / self.id / self.file


def _text(value: Mapping[str, Any], field: str) -> str:
    item = value.get(field)
    if not isinstance(item, str) or not item:
        raise GoldCorpusError(f"manifest entry {value.get('id', '?')}: {field} must be a non-empty string")
    return item


def load_manifest(path: Path = MANIFEST_PATH) -> List[GoldCorpus]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schemaVersion") != 1:
        raise GoldCorpusError(f"{path}: unsupported schemaVersion")
    corpora = [GoldCorpus.from_json(entry) for entry in data["corpora"]]
    ids = [corpus.id for corpus in corpora]
    if len(ids) != len(set(ids)):
        raise GoldCorpusError(f"{path}: duplicate corpus ids")
    return corpora


def corpus_by_id(corpus_id: str, corpora: Optional[Sequence[GoldCorpus]] = None) -> GoldCorpus:
    for corpus in corpora if corpora is not None else load_manifest():
        if corpus.id == corpus_id:
            return corpus
    raise GoldCorpusError(f"unknown gold corpus {corpus_id!r}")


def gold_dir_from_env() -> Path:
    value = os.environ.get("NLP_GOLD_DIR")
    return Path(value) if value else DEFAULT_GOLD_DIR


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(corpus: GoldCorpus, gold_dir: Path) -> Path:
    """Returns the corpus path if it exists and matches the pinned digest."""
    path = corpus.path(gold_dir)
    if not path.is_file():
        raise GoldCorpusError(
            f"{corpus.id}: missing {path}; run `python3 -m evaluation.gold fetch`"
        )
    actual = sha256_file(path)
    if actual != corpus.sha256:
        raise GoldCorpusError(f"{corpus.id}: sha256 {actual} does not match pinned {corpus.sha256}")
    return path


def fetch(corpus: GoldCorpus, gold_dir: Path, opener=urllib.request.urlopen) -> Path:
    """Downloads a corpus to a temporary file and keeps it only if the digest matches."""
    path = corpus.path(gold_dir)
    if path.is_file() and sha256_file(path) == corpus.sha256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".download-")
    try:
        digest = hashlib.sha256()
        with os.fdopen(handle, "wb") as output, opener(corpus.url) as response:
            for chunk in iter(lambda: response.read(1 << 20), b""):
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != corpus.sha256:
            raise GoldCorpusError(
                f"{corpus.id}: downloaded sha256 {digest.hexdigest()} does not match pinned {corpus.sha256}"
            )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path
