// Module worker for canonical browser NER. Spawned only by `createBrowserNer` on an
// explicit `prepare()` / `analyze()`; it loads Transformers.js and the pinned model lazily.
import {
  BROWSER_NER_MODEL,
  BROWSER_NER_TRANSFORMERS_MODULE_URL,
  projectBrowserNerSegments,
} from "./browser-ner-core.js";

let classifierPromise = null;

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}

function classifier(id, transformersModuleUrl) {
  if (!classifierPromise) {
    classifierPromise = import(/* @vite-ignore */ transformersModuleUrl ?? BROWSER_NER_TRANSFORMERS_MODULE_URL)
      .then(({ pipeline }) =>
        pipeline("token-classification", BROWSER_NER_MODEL.id, {
          revision: BROWSER_NER_MODEL.revision,
          dtype: BROWSER_NER_MODEL.dtype,
          progress_callback: (detail) => self.postMessage({ type: "progress", id, detail }),
        }),
      )
      .catch((error) => {
        classifierPromise = null;
        throw error;
      });
  }
  return classifierPromise;
}

self.addEventListener("message", async (event) => {
  const { type, id, segments, transformersModuleUrl } = event.data ?? {};
  if (!Number.isInteger(id)) return;
  let stage = "model";
  try {
    const ner = await classifier(id, transformersModuleUrl);
    if (type === "prepare") {
      self.postMessage({ type: "ready", id });
      return;
    }
    if (type !== "analyze") throw new Error(`Unknown browser NER request type ${JSON.stringify(type)}.`);
    stage = "analysis";
    // A batched (array) input always yields one prediction list per segment.
    const raw = segments.length === 0 ? [] : await ner(segments.map((segment) => segment.text), { ignore_labels: [] });
    self.postMessage({ type: "result", id, ...projectBrowserNerSegments(segments, raw) });
  } catch (error) {
    self.postMessage({ type: "error", id, stage, message: errorMessage(error) });
  }
});
