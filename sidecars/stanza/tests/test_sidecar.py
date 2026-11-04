"""Offline tests of the sidecar protocol and model pinning (no Stanza needed)."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stanza_sidecar.models import ModelError, fetch_file, load_lock, verify_language  # noqa: E402
from stanza_sidecar.server import Oracle, make_handler  # noqa: E402


def word(id, text, lemma, upos, head, deprel, feats=None):
    return SimpleNamespace(id=id, text=text, lemma=lemma, upos=upos, xpos=None, feats=feats, head=head, deprel=deprel)


def token(start, end, text, *words):
    return SimpleNamespace(start_char=start, end_char=end, text=text, words=list(words))


class FakePipeline:
    """Returns a canned German analysis of "Er geht zum Bahnhof." and records inputs."""

    def __init__(self):
        self.inputs = []

    def __call__(self, text):
        self.inputs.append(text)
        tokens = [
            token(0, 2, "Er", word(1, "Er", "er", "PRON", 2, "nsubj", "Case=Nom")),
            token(3, 7, "geht", word(2, "geht", "gehen", "VERB", 0, "root")),
            token(8, 11, "zum", word(3, "zu", "zu", "ADP", 5, "case"), word(4, "dem", "der", "DET", 5, "det")),
            token(12, 19, "Bahnhof", word(5, "Bahnhof", "Bahnhof", "NOUN", 2, "obl")),
            token(19, 20, ".", word(6, ".", ".", "PUNCT", 2, "punct")),
        ]
        return SimpleNamespace(sentences=[SimpleNamespace(tokens=tokens)])


def write_lock(directory: Path, payload: bytes) -> Path:
    digest = hashlib.sha256(payload).hexdigest()
    lock = {
        "schema": 1,
        "stanza_version": "1.15.0",
        "resources": {"url": "https://example.invalid/resources.json", "bytes": len(payload), "sha256": digest},
        "languages": {
            "de": {
                "repository": "stanfordnlp/stanza-de",
                "revision": "abc",
                "packages": {"tokenize": "combined_nocharlm", "pos": "combined_charlm"},
                "files": [{"path": "models/pos/combined_charlm.pt", "bytes": len(payload), "sha256": digest}],
            }
        },
    }
    path = directory / "models.lock.json"
    path.write_text(json.dumps(lock), encoding="utf-8")
    return path


class ModelPinningTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.payload = b"pinned model bytes"
        self.lock = load_lock(write_lock(self.root, self.payload), self.root / "models")

    def tearDown(self):
        self.temporary.cleanup()

    def opener(self, payload):
        calls = []

        def open_url(url, timeout):
            calls.append(url)
            return io.BytesIO(payload)

        return open_url, calls

    def test_layout_follows_stanza_directory_convention(self):
        pinned = self.lock.language("de").files[0]
        self.assertEqual(pinned.destination, self.root / "models" / "de" / "pos" / "combined_charlm.pt")
        self.assertEqual(
            pinned.url, "https://huggingface.co/stanfordnlp/stanza-de/resolve/abc/models/pos/combined_charlm.pt"
        )

    def test_fetch_verifies_and_is_idempotent(self):
        pinned = self.lock.language("de").files[0]
        open_url, calls = self.opener(self.payload)
        self.assertTrue(fetch_file(pinned, opener=open_url))
        self.assertFalse(fetch_file(pinned, opener=open_url))
        self.assertEqual(len(calls), 1)
        self.assertEqual(pinned.destination.read_bytes(), self.payload)

    def test_digest_mismatch_leaves_no_file(self):
        pinned = self.lock.language("de").files[0]
        open_url, _ = self.opener(b"tampered model byte")
        with self.assertRaises(ModelError):
            fetch_file(pinned, opener=open_url)
        self.assertEqual(list(pinned.destination.parent.iterdir()), [])

    def test_verify_reports_missing_and_corrupt_files(self):
        problems = verify_language(self.lock, "de")
        self.assertEqual(len(problems), 2)
        self.assertTrue(all("missing" in problem for problem in problems))
        pinned = self.lock.language("de").files[0]
        pinned.destination.parent.mkdir(parents=True)
        pinned.destination.write_bytes(b"x" * len(self.payload))
        self.assertIn("SHA-256", verify_language(self.lock, "de")[1])
        self.assertEqual(verify_language(self.lock, "fr"), ["`fr` is not in the models lock"])


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.lock = load_lock(write_lock(root, b"x"), root / "models")
        self.pipeline = FakePipeline()
        problems = {"de": [], "es": ["/models/es/pos/combined_charlm.pt is missing"]}
        self.oracle = Oracle(
            self.lock,
            ["de", "es"],
            lambda language: self.pipeline,
            verify=lambda lock, language: problems[language],
        )
        self.server = HTTPServer(("127.0.0.1", 0), make_handler(self.oracle, ["http://localhost:3000"]))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.temporary.cleanup()

    def request(self, method, path, body=None, headers=None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.base + path, data=data, method=method, headers=headers or {})
        if data is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, dict(response.headers), response.read()
        except urllib.error.HTTPError as error:
            return error.code, dict(error.headers), error.read()

    def test_health_reports_versions_and_readiness(self):
        status, _, body = self.request("GET", "/v1/health")
        health = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(health["protocol"], 1)
        self.assertEqual(health["stanza_version"], "1.15.0")
        self.assertTrue(health["languages"]["de"]["ready"])
        self.assertFalse(health["languages"]["es"]["ready"])
        self.assertEqual(health["languages"]["de"]["revision"], "abc")

    def test_analyze_returns_tokens_with_multiword_expansion(self):
        status, _, body = self.request(
            "POST", "/v1/analyze", {"sentences": [{"id": 7, "language": "de", "text": "Er geht\nzum Bahnhof."}]}
        )
        self.assertEqual(status, 200)
        result = json.loads(body)
        self.assertEqual(result["stanza_version"], "1.15.0")
        sentence = result["sentences"][0]
        self.assertEqual(sentence["id"], 7)
        self.assertEqual(sentence["models"]["packages"]["pos"], "combined_charlm")
        zum = sentence["tokens"][2]
        self.assertEqual((zum["start_char"], zum["end_char"]), (8, 11))
        self.assertEqual([w["text"] for w in zum["words"]], ["zu", "dem"])
        # Line breaks are blanked with the same length so offsets stay valid.
        self.assertEqual(self.pipeline.inputs, ["Er geht zum Bahnhof."])

    def test_missing_models_and_unserved_languages_are_typed_errors(self):
        status, _, body = self.request("POST", "/v1/analyze", {"sentences": [{"language": "es", "text": "Hola."}]})
        self.assertEqual(status, 503)
        self.assertEqual(json.loads(body)["error"]["kind"], "model_missing")
        status, _, body = self.request("POST", "/v1/analyze", {"sentences": [{"language": "fr", "text": "Salut."}]})
        self.assertEqual(status, 422)
        self.assertEqual(json.loads(body)["error"], {
            "kind": "unsupported_language",
            "message": "`fr` is not served (served: de, es)",
            "language": "fr",
        })
        status, _, body = self.request("POST", "/v1/analyze", {"text": "no sentences"})
        self.assertEqual((status, json.loads(body)["error"]["kind"]), (400, "bad_request"))

    def test_cors_only_for_allowed_origins(self):
        status, headers, _ = self.request("OPTIONS", "/v1/analyze", headers={"Origin": "http://localhost:3000"})
        self.assertEqual(status, 204)
        self.assertEqual(headers.get("Access-Control-Allow-Origin"), "http://localhost:3000")
        self.assertEqual(headers.get("Access-Control-Allow-Private-Network"), "true")
        _, headers, _ = self.request("GET", "/v1/health", headers={"Origin": "https://evil.example"})
        self.assertNotIn("Access-Control-Allow-Origin", headers)


if __name__ == "__main__":
    unittest.main()
