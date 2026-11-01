# @moritzbrantner/text-linguistics-wasm

WASM package for `text-linguistics`.

```bash
bun run --cwd packages/text-linguistics-wasm build
bun run text-wasm:bench:all
```

Benchmark numbers are local to the current browser, machine, and build profile.

## Browser NER (`@moritzbrantner/text-linguistics-wasm/browser-ner`)

The canonical model-backed browser NER used by the Pages workbench, for consumers that
need it without copying inference. Importing it makes no network or model requests.

```js
import { createBrowserNer } from "@moritzbrantner/text-linguistics-wasm/browser-ner";

const ner = createBrowserNer();
await ner.prepare({ onProgress: ({ detail }) => console.log(detail) }); // explicit download/load
const result = await ner.analyze(
  { language: "en", segments: [{ id: "cue-1", text: "Ada Lovelace met Babbage.", byteStart: 0 }] },
  { signal: controller.signal },
);
// result.status: "completed" | "unsupported-language"
// result.entities[i]: { segmentId, label, text, score, start, end, byteStart, byteEnd }
// result.model: { id, revision (pinned commit), supportedLanguages: ["en", "de", "es"], ... }
ner.dispose();
```

- The model is `Xenova/distilbert-base-multilingual-cased-ner-hrl` (the ONNX conversion of
  `Davlan/distilbert-base-multilingual-cased-ner-hrl`, AFL-3.0) pinned to a Hugging Face
  commit, int8-quantized (`dtype: "q8"`, about 135 MB), and loaded with Transformers.js in a
  module worker owned by the adapter. Labels: PER, ORG, LOC, DATE.
- Sub-word predictions are aggregated to whole words (first sub-word label wins); spans are
  exact slices of the caller's segment text, with absolute UTF-8 byte offsets.
- English, German and Spanish are supported. Other or unknown languages return
  `status: "unsupported-language"` without loading the model. The caller supplies the
  language; the module does not detect it.
- Exact-span PER/ORG/LOC micro F1 on `evaluation/corpora/named-entities-multilingual-v1.jsonl`
  (Transformers.js 4.2 in Bun, 2026-11-01): en 0.971, de 0.985, es 1.000. The previous
  English-only `onnx-community/distilbert-NER-ONNX` (q4) scored en 0.971, de 0.609,
  es 0.444.
- Every model file (config, tokenizer and weights) is fetched at the pinned commit and kept
  in the browser's Cache Storage (`transformers-cache`). Once cached, startup sends no
  request to Hugging Face, so the module works offline. A failed load rejects with a
  `BrowserNerError` (`stage: "model"`) whose message says the download failed (offline?).
- Aborting a request terminates the worker and rejects every in-flight request, so a
  canceled request never resolves. The next request starts a fresh worker.
- Consume it from a pushed revision of this repository; no npm publication is required, and
  the model is fetched only at runtime, never during a consumer build.
