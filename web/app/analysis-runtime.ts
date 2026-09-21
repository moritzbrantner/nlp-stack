import type { SurfaceRequest, SurfaceResponse } from "../../packages/nlp-app-ui/dist/package-surface/index.js";

export type AnalysisRuntime = {
  dispose: () => void;
  runOperation: (request: SurfaceRequest, options: { signal: AbortSignal }) => Promise<SurfaceResponse>;
};
type RuntimeFactory = { createSession: () => AnalysisRuntime };
type RuntimeWindow = Window & { nlpStackTextAnalysis?: { ready: Promise<RuntimeFactory> } };
const readyEvent = "nlp-stack-text-analysis-ready";
const scriptId = "nlp-stack-text-analysis-runtime";
const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
let factoryPromise: Promise<RuntimeFactory> | null = null;

function loadFactory(): Promise<RuntimeFactory> {
  const registered = (window as RuntimeWindow).nlpStackTextAnalysis;
  if (registered) return registered.ready;
  if (factoryPromise) return factoryPromise;
  factoryPromise = new Promise<RuntimeFactory>((resolve, reject) => {
    const script = document.createElement("script");
    script.id = scriptId;
    script.type = "module";
    script.src = `${basePath}/nlp-analysis-runtime.js`;
    const cleanup = () => {
      clearTimeout(timer);
      window.removeEventListener(readyEvent, onReady);
      script.removeEventListener("error", onError);
    };
    const fail = (message: string) => {
      cleanup();
      script.remove();
      reject(new Error(message));
    };
    const onReady = () => {
      cleanup();
      const factory = (window as RuntimeWindow).nlpStackTextAnalysis;
      if (factory) resolve(factory.ready);
      else fail("The analysis runtime did not register correctly.");
    };
    const onError = () => fail("Failed to load the analysis runtime. Try again.");
    const timer = setTimeout(() => fail("Loading the analysis runtime exceeded 30 seconds. Try again."), 30_000);
    window.addEventListener(readyEvent, onReady, { once: true });
    script.addEventListener("error", onError, { once: true });
    document.head.append(script);
  }).catch((error: unknown) => {
    factoryPromise = null;
    throw error;
  });
  return factoryPromise;
}

export function loadAnalysisRuntime(signal: AbortSignal): Promise<AnalysisRuntime> {
  return new Promise((resolve, reject) => {
    const abort = () => reject(new DOMException("Analysis cancelled.", "AbortError"));
    if (signal.aborted) {
      abort();
      return;
    }
    signal.addEventListener("abort", abort, { once: true });
    loadFactory().then((factory) => {
      signal.removeEventListener("abort", abort);
      if (!signal.aborted) resolve(factory.createSession());
    }).catch((error: unknown) => {
      signal.removeEventListener("abort", abort);
      reject(error);
    });
  });
}
