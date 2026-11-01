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

// Transformers.js 4.2 `pipeline()` probes `config.json`, `tokenizer_config.json` and the model
// files at the default `main` revision even when `revision` is passed (its file-list and
// progress-size checks drop the option). Those probes miss the cache, which is keyed by the
// pinned URL, so a cached model still hits huggingface.co and loses its tokenizer offline.
// Baking the pinned commit into `env.remotePathTemplate` makes every URL, probe or download,
// the pinned one; cached files are then served from Cache Storage without any request.
function pinRemoteRevision(env, revision) {
  env.remotePathTemplate = env.remotePathTemplate.replaceAll("{revision}", encodeURIComponent(revision));
}

function loadError(error) {
  return new Error(
    `Browser NER model ${BROWSER_NER_MODEL.id} could not be loaded; the download failed (offline?): ${errorMessage(error)}`,
  );
}

function classifier(id, transformersModuleUrl) {
  if (!classifierPromise) {
    classifierPromise = import(/* @vite-ignore */ transformersModuleUrl ?? BROWSER_NER_TRANSFORMERS_MODULE_URL)
      .then(async ({ env, pipeline }) => {
        pinRemoteRevision(env, BROWSER_NER_MODEL.revision);
        const ner = await pipeline("token-classification", BROWSER_NER_MODEL.id, {
          revision: BROWSER_NER_MODEL.revision,
          dtype: BROWSER_NER_MODEL.dtype,
          progress_callback: (detail) => self.postMessage({ type: "progress", id, detail }),
        });
        // Transformers.js silently omits a tokenizer whose files it could not reach.
        if (typeof ner?.tokenizer !== "function") throw new Error("the tokenizer files are unavailable");
        return ner;
      })
      .catch((error) => {
        classifierPromise = null;
        throw loadError(error);
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
