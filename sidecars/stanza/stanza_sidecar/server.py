"""HTTP/JSON server exposing pinned Stanza pipelines to nlp-stack.

Endpoints (protocol 1):

``GET /v1/health``
    Stanza version and, per language, the pinned packages, model revision and
    whether its verified model files are present.

``POST /v1/analyze``
    ``{"sentences": [{"id": 0, "language": "de", "text": "..."}]}``. Each entry is
    one sentence; Stanza tokenizes it (including multi-word token expansion) without
    re-splitting it. Every token carries Stanza's code-point ``start_char`` and
    ``end_char`` relative to the sentence text; the Rust adapter converts them to
    UTF-8 byte spans.

Errors are ``{"error": {"kind": ..., "message": ..., "language": ...}}`` with kinds
``bad_request`` (400), ``not_found`` (404), ``unsupported_language`` (422),
``model_missing`` (503) and ``internal`` (500). A language whose pinned models are
missing or corrupt is refused; nothing is downloaded or substituted at serve time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from . import PROTOCOL_VERSION
from .models import DEFAULT_LOCK, DEFAULT_MODEL_DIR, ModelLock, load_lock, verify_language

SERVICE = "nlp-stack-stanza-sidecar"
DEFAULT_PORT = 8765
DEFAULT_ORIGINS = (
    "https://moritzbrantner.github.io",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:6006",
    "http://127.0.0.1:6006",
)
MAX_BODY_BYTES = 4 << 20
PROCESSORS = ("tokenize", "mwt", "pos", "lemma", "depparse")


class RequestError(Exception):
    def __init__(self, status: int, kind: str, message: str, language: str | None = None):
        super().__init__(message)
        self.status = status
        self.kind = kind
        self.message = message
        self.language = language

    def body(self) -> dict:
        error = {"kind": self.kind, "message": self.message}
        if self.language is not None:
            error["language"] = self.language
        return {"error": error}


def stanza_pipeline_factory(lock: ModelLock, model_dir: Path):
    """Builds a CPU pipeline for one language from the verified pinned files only."""

    def build(language: str):
        import stanza  # Imported lazily so the protocol is testable without Stanza.

        if stanza.__version__ != lock.stanza_version:
            raise RuntimeError(f"installed Stanza {stanza.__version__} differs from the pinned {lock.stanza_version}")
        packages = lock.language(language).packages
        return stanza.Pipeline(
            language,
            dir=str(model_dir),
            # package=None: only these processors, not Stanza's whole default set.
            package=None,
            processors={processor: packages[processor] for processor in PROCESSORS},
            download_method=None,
            tokenize_no_ssplit=True,
            use_gpu=False,
            logging_level="WARN",
        )

    return build


class Oracle:
    """Language-routed Stanza pipelines over the pinned models, loaded on first use."""

    def __init__(self, lock: ModelLock, languages, pipeline_factory, *, verify=verify_language):
        self.lock = lock
        self.pipeline_factory = pipeline_factory
        self.pipelines = {}
        self.problems = {}
        for language in languages:
            self.problems[language] = verify(lock, language)

    def health(self) -> dict:
        languages = {}
        for language, problems in self.problems.items():
            pin = self.lock.language(language)
            languages[language] = {
                "ready": not problems,
                "loaded": language in self.pipelines,
                "problems": problems,
                "repository": pin.repository if pin else None,
                "revision": pin.revision if pin else None,
                "packages": pin.packages if pin else {},
            }
        return {
            "service": SERVICE,
            "protocol": PROTOCOL_VERSION,
            "stanza_version": self.lock.stanza_version,
            "resources_sha256": self.lock.resources.sha256,
            "languages": languages,
        }

    def pipeline(self, language: str):
        if language not in self.problems:
            served = ", ".join(sorted(self.problems)) or "none"
            raise RequestError(422, "unsupported_language", f"`{language}` is not served (served: {served})", language)
        problems = self.problems[language]
        if problems:
            raise RequestError(
                503,
                "model_missing",
                f"pinned Stanza models for `{language}` are not usable: {'; '.join(problems)}; run sidecars/stanza/setup.sh",
                language,
            )
        if language not in self.pipelines:
            self.pipelines[language] = self.pipeline_factory(language)
        return self.pipelines[language]

    def analyze(self, request) -> dict:
        if not isinstance(request, dict) or not isinstance(request.get("sentences"), list):
            raise RequestError(400, "bad_request", "body must be an object with a `sentences` array")
        results = []
        for position, sentence in enumerate(request["sentences"]):
            if not isinstance(sentence, dict):
                raise RequestError(400, "bad_request", f"sentence {position} is not an object")
            text, language = sentence.get("text"), sentence.get("language")
            if not isinstance(text, str) or not isinstance(language, str):
                raise RequestError(400, "bad_request", f"sentence {position} needs string `text` and `language`")
            pin = self.lock.language(language)
            pipeline = self.pipeline(language)
            results.append(
                {
                    "id": sentence.get("id", position),
                    "language": language,
                    "models": {"repository": pin.repository, "revision": pin.revision, "packages": pin.packages},
                    "tokens": analyze_sentence(pipeline, text),
                }
            )
        return {
            "protocol": PROTOCOL_VERSION,
            "stanza_version": self.lock.stanza_version,
            "sentences": results,
        }


def analyze_sentence(pipeline, text: str) -> list[dict]:
    """Tokens of one sentence with their syntactic words.

    Line breaks become spaces (same code-point length, so offsets are unchanged) so
    Stanza keeps the caller's sentence as one. Should Stanza still return several
    sentences, their words are renumbered into one sequence so heads stay valid.
    """
    document = pipeline(text.replace("\r", " ").replace("\n", " "))
    tokens = []
    offset = 0
    for sentence in document.sentences:
        count = 0
        for token in sentence.tokens:
            words = []
            for word in token.words:
                count += 1
                words.append(
                    {
                        "id": word.id + offset,
                        "text": word.text,
                        "lemma": word.lemma,
                        "upos": word.upos,
                        "xpos": word.xpos,
                        "feats": word.feats,
                        "head": (word.head + offset) if word.head else 0,
                        "deprel": word.deprel,
                    }
                )
            tokens.append(
                {
                    "start_char": token.start_char,
                    "end_char": token.end_char,
                    "text": token.text,
                    "words": words,
                }
            )
        offset += count
    return tokens


def make_handler(oracle: Oracle, allowed_origins):
    allowed = frozenset(allowed_origins)

    class Handler(BaseHTTPRequestHandler):
        server_version = SERVICE
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):  # noqa: A002 - BaseHTTPRequestHandler's name
            sys.stderr.write("%s %s\n" % (self.address_string(), format % args))

        def cors_headers(self):
            origin = self.headers.get("Origin")
            if origin and origin in allowed:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                # Chrome's Private Network Access preflight for a public page calling localhost.
                self.send_header("Access-Control-Allow-Private-Network", "true")

        def send_json(self, status: int, body: dict):
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.cors_headers()
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self):  # noqa: N802
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.cors_headers()
            self.end_headers()

        def do_GET(self):  # noqa: N802
            if self.path == "/v1/health":
                self.send_json(200, oracle.health())
            else:
                self.send_json(404, RequestError(404, "not_found", f"no route {self.path}").body())

        def do_POST(self):  # noqa: N802
            try:
                if self.path != "/v1/analyze":
                    raise RequestError(404, "not_found", f"no route {self.path}")
                length = int(self.headers.get("Content-Length") or 0)
                if length <= 0 or length > MAX_BODY_BYTES:
                    raise RequestError(400, "bad_request", f"body must be 1..{MAX_BODY_BYTES} bytes")
                try:
                    request = json.loads(self.rfile.read(length))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise RequestError(400, "bad_request", f"invalid JSON: {error}") from error
                self.send_json(200, oracle.analyze(request))
            except RequestError as error:
                self.send_json(error.status, error.body())
            except Exception as error:  # noqa: BLE001 - reported to the client as `internal`
                self.send_json(500, RequestError(500, "internal", f"{type(error).__name__}: {error}").body())

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="nlp-stack Stanza oracle sidecar")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("NLP_STANZA_PORT", DEFAULT_PORT)))
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--model-dir", type=Path, default=Path(os.environ.get("NLP_STANZA_MODEL_DIR", DEFAULT_MODEL_DIR)))
    parser.add_argument("--languages", default="en,de,es")
    parser.add_argument(
        "--allow-origin",
        action="append",
        help="browser origin allowed by CORS (repeatable; defaults to the Pages and dev origins)",
    )
    args = parser.parse_args(argv)

    lock = load_lock(args.lock, args.model_dir)
    languages = [language.strip() for language in args.languages.split(",") if language.strip()]
    oracle = Oracle(lock, languages, stanza_pipeline_factory(lock, args.model_dir))
    for language, problems in oracle.problems.items():
        state = "ready" if not problems else "NOT READY: " + "; ".join(problems)
        print(f"{language}: {state}", file=sys.stderr)
    server = HTTPServer((args.host, args.port), make_handler(oracle, args.allow_origin or DEFAULT_ORIGINS))
    print(f"{SERVICE} listening on http://{args.host}:{server.server_port}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
