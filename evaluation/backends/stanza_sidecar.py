"""Stanza sidecar adapter: gold sentence texts in, CoNLL-U predictions plus system pins out.

Each gold sentence is sent as one sentence (the sidecar does not re-split it), so
the scores use gold sentence segmentation and Stanza's own tokenization and
multi-word expansion. Needs a running sidecar (``sidecars/stanza/serve``)::

    python3 -m evaluation.backends.stanza_sidecar --corpus ud-de-gsd-test-2.17 --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from evaluation.gold.conllu import Sentence, Token, Word, read_conllu, write_conllu
from evaluation.gold.manifest import corpus_by_id, gold_dir_from_env, verify

BACKEND = "stanza-sidecar"
DEFAULT_URL = "http://127.0.0.1:8765"
BATCH = 16


class StanzaSidecarError(RuntimeError):
    pass


def _request(url: str, body: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise StanzaSidecarError(f"{url}: HTTP {error.code}: {error.read().decode('utf-8', 'replace')}") from None
    except urllib.error.URLError as error:
        raise StanzaSidecarError(f"{url}: sidecar not reachable ({error.reason}); start sidecars/stanza/serve") from None


def system_info(health: Mapping[str, Any], language: str) -> Dict[str, Any]:
    entry = health["languages"].get(language)
    if not entry or not entry.get("ready"):
        raise StanzaSidecarError(f"sidecar language {language} not ready: {entry and entry.get('problems')}")
    return {
        "backend": BACKEND,
        "model": entry["repository"],
        "revision": entry["revision"],
        "details": {
            "stanzaVersion": health["stanza_version"],
            "protocol": health["protocol"],
            "resourcesSha256": health["resources_sha256"],
            "packages": entry["packages"],
            "segmentation": "gold sentences, Stanza tokenization and multi-word expansion",
        },
    }


def sentence_from_response(sent_id: str, text: str, tokens: Sequence[Mapping[str, Any]]) -> Sentence:
    """Turns sidecar tokens into a CoNLL-U sentence; token forms are taken from the text."""
    sentence = Sentence(sent_id, text)
    for index, token in enumerate(tokens):
        start, end = token["start_char"], token["end_char"]
        # UD leaves SpaceAfter unmarked on a sentence's last token.
        following = tokens[index + 1]["start_char"] if index + 1 < len(tokens) else end + 1
        words = [
            Word(
                id=int(word["id"]),
                form=word["text"],
                lemma=word.get("lemma") or "_",
                upos=word.get("upos") or "_",
                xpos=word.get("xpos") or "_",
                feats=word.get("feats") or "_",
                head=int(word.get("head") or 0),
                deprel=word.get("deprel") or "_",
            )
            for word in token["words"]
        ]
        surface = text[start:end]
        if len(words) == 1 and words[0].form != surface:
            # A single-word token's FORM is its surface text; Stanza may report a
            # normalized word text (for example `<UNK>` for some URLs).
            words[0] = replace(words[0], form=surface)
        sentence.tokens.append(Token(surface, words, space_after=following > end))
    return sentence


def predict(url: str, sentences: Sequence[Sentence], language: str) -> List[Sentence]:
    predictions: List[Sentence] = []
    for offset in range(0, len(sentences), BATCH):
        batch = sentences[offset : offset + BATCH]
        response = _request(
            f"{url}/v1/analyze",
            {"sentences": [{"id": i, "language": language, "text": s.text} for i, s in enumerate(batch)]},
        )
        for gold, result in zip(batch, response["sentences"]):
            predictions.append(sentence_from_response(gold.sent_id, gold.text, result["tokens"]))
        print(f"\r{language}: {len(predictions)}/{len(sentences)}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return predictions


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--url", default=os.environ.get("NLP_STANZA_URL", DEFAULT_URL))
    parser.add_argument("--gold-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    gold_dir = args.gold_dir or gold_dir_from_env()
    health = _request(f"{args.url}/v1/health")
    systems: Dict[str, Any] = {}
    args.out.mkdir(parents=True, exist_ok=True)
    for corpus_id in args.corpus:
        corpus = corpus_by_id(corpus_id)
        if corpus.task != "syntax":
            raise SystemExit(f"{corpus_id} is not a syntax corpus")
        systems[corpus.language] = system_info(health, corpus.language)
        gold = read_conllu(verify(corpus, gold_dir))
        predictions = predict(args.url, gold, corpus.language)
        with (args.out / f"{corpus.id}.stanza.conllu").open("w", encoding="utf-8") as handle:
            write_conllu(predictions, handle)
    (args.out / "stanza.systems.json").write_text(json.dumps(systems, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
