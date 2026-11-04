"""Downloads and verifies the pinned Stanza model files (idempotent)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .models import DEFAULT_LOCK, DEFAULT_MODEL_DIR, ModelError, fetch_file, load_lock


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--languages", default="en,de,es", help="comma-separated ISO 639-1 codes")
    args = parser.parse_args(argv)

    lock = load_lock(args.lock, args.model_dir)
    languages = [language.strip() for language in args.languages.split(",") if language.strip()]
    try:
        downloaded = fetch_file(lock.resources)
        print(f"{'downloaded' if downloaded else 'verified'} {lock.resources.destination}")
        for language in languages:
            pin = lock.language(language)
            if pin is None:
                raise ModelError(f"`{language}` is not in the models lock")
            for pinned in pin.files:
                downloaded = fetch_file(pinned)
                print(f"{'downloaded' if downloaded else 'verified'} {pinned.destination}")
    except ModelError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
