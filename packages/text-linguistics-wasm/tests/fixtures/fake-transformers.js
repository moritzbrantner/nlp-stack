// Stand-in for the parts of Transformers.js 4.2 that decide which URLs a `pipeline()` call
// touches. Like the real library: URLs come from `env.remoteHost` + `env.remotePathTemplate`,
// Cache Storage (keyed by that URL) is checked before any request, `pipeline()`'s config,
// tokenizer-existence and progress-size probes use the default `main` revision, the actual
// loads use the requested `revision`, and a tokenizer whose probe fails is silently omitted.
export const env = {};
export const state = { cache: new Set(), requests: [], online: true };

/** A fresh library instance (page reload): default env, no request log; Cache Storage persists. */
export function restart({ online }) {
  env.remoteHost = "https://huggingface.co/";
  env.remotePathTemplate = "{model}/resolve/{revision}/";
  state.requests = [];
  state.online = online;
}
restart({ online: true });

function remoteUrl(model, revision, file) {
  const path = env.remotePathTemplate.replaceAll("{model}", model).replaceAll("{revision}", encodeURIComponent(revision));
  return `${env.remoteHost}${path}${file}`;
}

function request(url) {
  state.requests.push(url);
  if (!state.online) throw new TypeError("Failed to fetch");
}

/** `getModelFile`: cache first, else download and cache. */
async function load(model, revision, file) {
  const url = remoteUrl(model, revision, file);
  if (state.cache.has(url)) return;
  request(url);
  state.cache.add(url);
}

/** `get_file_metadata`: cache first, else an uncached `Range: bytes=0-0` request. */
async function exists(model, file) {
  const url = remoteUrl(model, "main", file);
  if (state.cache.has(url)) return true;
  try {
    request(url);
    return true;
  } catch {
    return false;
  }
}

export async function pipeline(task, model, { revision = "main", progress_callback = null } = {}) {
  const files = ["config.json", "onnx/model_quantized.onnx"];
  await load(model, "main", "config.json");
  if (await exists(model, "tokenizer_config.json")) files.push("tokenizer.json", "tokenizer_config.json");
  if (progress_callback) for (const file of files) await exists(model, file);
  for (const file of files) {
    await load(model, revision, file);
    progress_callback?.({ status: "done", name: model, file });
  }
  const results = { task };
  if (files.includes("tokenizer.json")) results.tokenizer = () => ({});
  const ner = (texts) => {
    results.tokenizer(texts);
    return texts.map(() => []);
  };
  if (results.tokenizer) ner.tokenizer = results.tokenizer;
  return ner;
}
