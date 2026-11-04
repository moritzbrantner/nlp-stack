"""Digest-pinned Stanza model files: download, verify and locate them.

``models.lock.json`` pins the Stanza release, the ``resources.json`` it reads and,
per language, the Hugging Face commit, the UD packages and the SHA-256 of every
model file. Nothing here falls back to Stanza's own downloader: a file that is
missing or whose digest differs is an error.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

SIDECAR_DIR = Path(__file__).resolve().parent.parent
DEFAULT_LOCK = SIDECAR_DIR / "models.lock.json"
DEFAULT_MODEL_DIR = SIDECAR_DIR / ".models"
HF_RESOLVE = "https://huggingface.co/{repository}/resolve/{revision}/{path}"
CHUNK = 1 << 20


class ModelError(Exception):
    """A pinned model file is missing, corrupt or could not be fetched."""


@dataclass(frozen=True)
class PinnedFile:
    url: str
    destination: Path
    size: int
    sha256: str


@dataclass(frozen=True)
class LanguagePin:
    language: str
    repository: str
    revision: str
    packages: dict
    files: tuple


@dataclass(frozen=True)
class ModelLock:
    stanza_version: str
    resources: PinnedFile
    languages: dict

    def language(self, language: str) -> LanguagePin | None:
        return self.languages.get(language)


def load_lock(lock_path: Path = DEFAULT_LOCK, model_dir: Path = DEFAULT_MODEL_DIR) -> ModelLock:
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    if lock.get("schema") != 1:
        raise ModelError(f"unsupported models lock schema {lock.get('schema')!r}")
    model_dir = Path(model_dir)
    resources = lock["resources"]
    languages = {}
    for language, pin in lock["languages"].items():
        files = []
        for entry in pin["files"]:
            # Stanza reads <dir>/<lang>/<processor>/<package>.pt.
            relative = entry["path"].removeprefix("models/")
            files.append(
                PinnedFile(
                    url=HF_RESOLVE.format(repository=pin["repository"], revision=pin["revision"], path=entry["path"]),
                    destination=model_dir / language / relative,
                    size=entry["bytes"],
                    sha256=entry["sha256"],
                )
            )
        languages[language] = LanguagePin(
            language=language,
            repository=pin["repository"],
            revision=pin["revision"],
            packages=dict(pin["packages"]),
            files=tuple(files),
        )
    return ModelLock(
        stanza_version=lock["stanza_version"],
        resources=PinnedFile(
            url=resources["url"],
            destination=model_dir / "resources.json",
            size=resources["bytes"],
            sha256=resources["sha256"],
        ),
        languages=languages,
    )


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(pinned: PinnedFile) -> str | None:
    """``None`` when the file is present with the pinned size and digest, else why not."""
    path = pinned.destination
    if not path.is_file():
        return f"{path} is missing"
    size = path.stat().st_size
    if size != pinned.size:
        return f"{path} has {size} bytes, expected {pinned.size}"
    actual = sha256_of(path)
    if actual != pinned.sha256:
        return f"{path} has SHA-256 {actual}, expected {pinned.sha256}"
    return None


def verify_language(lock: ModelLock, language: str) -> list[str]:
    """Problems with the shared resources file and ``language``'s model files."""
    pin = lock.language(language)
    if pin is None:
        return [f"`{language}` is not in the models lock"]
    problems = [problem for problem in [verify_file(lock.resources)] if problem]
    problems.extend(problem for problem in map(verify_file, pin.files) if problem)
    return problems


def fetch_file(pinned: PinnedFile, *, opener=urllib.request.urlopen) -> bool:
    """Ensures ``pinned`` is present and verified. Returns whether it was downloaded.

    The download goes to a temporary file beside the destination and is renamed into
    place only after its digest matches, so an interrupted run never leaves a file
    that later passes for a verified one.
    """
    if verify_file(pinned) is None:
        return False
    pinned.destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    fd, temporary = tempfile.mkstemp(prefix=pinned.destination.name + ".", suffix=".part", dir=pinned.destination.parent)
    try:
        with os.fdopen(fd, "wb") as out, opener(pinned.url, timeout=120) as response:
            while chunk := response.read(CHUNK):
                digest.update(chunk)
                size += len(chunk)
                out.write(chunk)
        actual = digest.hexdigest()
        if size != pinned.size or actual != pinned.sha256:
            raise ModelError(
                f"{pinned.url}: got {size} bytes with SHA-256 {actual}, "
                f"expected {pinned.size} bytes with SHA-256 {pinned.sha256}"
            )
        os.replace(temporary, pinned.destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return True
