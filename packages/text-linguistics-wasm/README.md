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
// result.model: { id, revision (pinned commit), supportedLanguages: ["en"], ... }
ner.dispose();
```

- The model is `onnx-community/distilbert-NER-ONNX` pinned to a Hugging Face commit and
  loaded with Transformers.js in a module worker owned by the adapter.
- Sub-word predictions are aggregated to whole words (first sub-word label wins); spans are
  exact slices of the caller's segment text, with absolute UTF-8 byte offsets.
- Only English is supported. Other or unknown languages return
  `status: "unsupported-language"` without loading the model.
- Aborting a request terminates the worker and rejects every in-flight request, so a
  canceled request never resolves. The next request starts a fresh worker.
- Consume it from a pushed revision of this repository; no npm publication is required, and
  the model is fetched only at runtime, never during a consumer build.
