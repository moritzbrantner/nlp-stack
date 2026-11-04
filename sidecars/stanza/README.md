# Stanza sidecar

A local HTTP/JSON service that runs pinned [Stanza](https://stanfordnlp.github.io/stanza/)
pipelines for English, German and Spanish (tokenization with multi-word expansion,
lemmas, UPOS/XPOS, UD features, dependency syntax). nlp-stack uses it as an oracle
through `text_linguistics::StanzaSidecar` (feature `stanza`) and from the workbench.

```sh
bash sidecars/stanza/setup.sh     # uv venv in .venv, models in .models (both ignored)
sidecars/stanza/serve             # http://127.0.0.1:8765; --port or NLP_STANZA_PORT
```

- `pyproject.toml` / `uv.lock` pin Python 3.12, Stanza 1.15.0 and CPU-only torch.
  `setup.sh` runs `uv sync --frozen`, so lock drift is refused.
- `models.lock.json` pins, per language, the Hugging Face revision of
  `stanfordnlp/stanza-<lang>`, the default UD packages of Stanza 1.15.0 and the
  SHA-256 and size of every model file, plus the `resources.json` Stanza reads.
  Downloads go to a temporary file and are kept only when the digest matches;
  re-running skips verified files. `tools/lock_models.py` regenerates the lock when
  the pins move deliberately.
- `serve` verifies the files again at start-up and refuses a language whose files
  are missing or corrupt (`model_missing`, HTTP 503). It never downloads.

Protocol 1:

- `GET /v1/health`: Stanza version, protocol, and per language `ready`, problems,
  repository, revision and packages.
- `POST /v1/analyze` with `{"sentences": [{"id": 0, "language": "de", "text": "..."}]}`.
  Each entry is one sentence; Stanza does not re-split it. The answer has, per
  sentence, the models used and the tokens with Stanza's code-point `start_char`
  and `end_char` relative to the sentence and their words (`id`, `text`, `lemma`,
  `upos`, `xpos`, `feats`, `head`, `deprel`). The Rust adapter converts offsets to
  UTF-8 byte spans.
- Errors: `{"error": {"kind", "message", "language"?}}` with `bad_request` (400),
  `not_found` (404), `unsupported_language` (422), `model_missing` (503),
  `internal` (500).

CORS is answered only for the Pages origin and the local dev/Storybook origins
(`--allow-origin` replaces the list), including Chrome's private-network preflight.
The server listens on 127.0.0.1 by default and handles one request at a time.

`python3 -m unittest discover -s sidecars/stanza/tests` checks the protocol and the
digest handling offline, without Stanza.
