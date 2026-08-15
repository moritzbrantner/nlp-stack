let wasmModulePromise;

export async function init() {
  const wasmEntry = "./pkg/moenarch_text_classification_wasm.js";
  wasmModulePromise ??= import(/* @vite-ignore */ wasmEntry).then(async (module) => {
    if (typeof module.default === "function") {
      await module.default();
    }
    return module;
  });
  return wasmModulePromise;
}

export async function packageSurface() {
  const module = await init();
  return fromWasm(module.packageSurface());
}

export async function runOperation(request) {
  const module = await init();
  return fromWasm(module.runOperation(request));
}

function fromWasm(value) {
  if (value instanceof Map) {
    return Object.fromEntries(Array.from(value.entries(), ([key, entry]) => [key, fromWasm(entry)]));
  }
  if (Array.isArray(value)) {
    return value.map(fromWasm);
  }
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([key, entry]) => [key, fromWasm(entry)]));
  }
  return value;
}
