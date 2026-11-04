"""Regenerates ``models.lock.json`` from Stanza's resources and Hugging Face.

Run only when deliberately moving the pins (for example a new Stanza release):

    python3 -I sidecars/stanza/tools/lock_models.py --stanza-version 1.15.0 \
        --resources-commit <stanza-resources commit> > sidecars/stanza/models.lock.json

The lock records, per language, the default UD packages of the pinned Stanza
release, the Hugging Face commit of ``stanfordnlp/stanza-<lang>`` at tag
``v<version>``, and the SHA-256 and size of every model file (the LFS object id).
``setup.sh`` downloads exactly these files and refuses any digest mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request

LANGUAGES = ("en", "de", "es")
PROCESSORS = ("tokenize", "mwt", "pos", "lemma", "depparse")
RESOURCES_URL = "https://raw.githubusercontent.com/stanfordnlp/stanza-resources/{commit}/resources_{version}.json"
HF_API = "https://huggingface.co/api/models/stanfordnlp/stanza-{language}"


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def fetch_json(url: str):
    return json.loads(fetch(url))


def model_file(language: str, revision: str, path: str) -> dict:
    directory, _, name = path.rpartition("/")
    listing = fetch_json(f"{HF_API.format(language=language)}/tree/{revision}/{directory}")
    for entry in listing:
        if entry["path"] == path:
            lfs = entry.get("lfs") or {}
            if "oid" not in lfs:
                raise SystemExit(f"{language}:{path} is not an LFS file; cannot pin its digest")
            return {"path": path, "bytes": entry["size"], "sha256": lfs["oid"]}
    raise SystemExit(f"{language}:{path} ({name}) not found at {revision}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stanza-version", required=True)
    parser.add_argument("--resources-commit", required=True)
    args = parser.parse_args()

    resources_url = RESOURCES_URL.format(commit=args.resources_commit, version=args.stanza_version)
    resources_bytes = fetch(resources_url)
    resources = json.loads(resources_bytes)
    lock = {
        "schema": 1,
        "stanza_version": args.stanza_version,
        "resources": {
            "url": resources_url,
            "bytes": len(resources_bytes),
            "sha256": hashlib.sha256(resources_bytes).hexdigest(),
        },
        "languages": {},
    }
    for language in LANGUAGES:
        entry = resources[language]
        defaults = entry["default_processors"]
        revision = fetch_json(f"{HF_API.format(language=language)}/revision/v{args.stanza_version}")["sha"]
        packages = {processor: defaults[processor] for processor in PROCESSORS}
        paths = []
        for processor, package in packages.items():
            paths.append(f"models/{processor}/{package}.pt")
            for dependency in entry[processor][package].get("dependencies", []):
                path = f"models/{dependency['model']}/{dependency['package']}.pt"
                if path not in paths:
                    paths.append(path)
        lock["languages"][language] = {
            "repository": f"stanfordnlp/stanza-{language}",
            "revision": revision,
            "packages": packages,
            "files": [model_file(language, revision, path) for path in paths],
        }
    json.dump(lock, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
