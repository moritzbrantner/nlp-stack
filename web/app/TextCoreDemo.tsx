"use client";

import {
  PackageSurfaceWorkbench,
  type PackageAppConfig,
  type PackageSurface,
  type SurfaceRequest,
  type SurfaceResponse,
} from "@moritzbrantner/nlp-app-ui/package-surface";

type TextCoreRuntime = {
  packageSurface: () => PackageSurface;
  runOperation: (request: SurfaceRequest) => SurfaceResponse;
};

type RuntimeHandle = {
  ready: Promise<TextCoreRuntime>;
};

type RuntimeWindow = Window & {
  nlpStackTextCore?: RuntimeHandle;
};

const runtimeReadyEvent = "nlp-stack-text-core-ready";
const runtimeScriptId = "nlp-stack-text-core-runtime";
const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

let runtimePromise: Promise<TextCoreRuntime> | null = null;

function loadRuntime(): Promise<TextCoreRuntime> {
  if (runtimePromise) {
    return runtimePromise;
  }

  const runtimeWindow = window as RuntimeWindow;
  const existing = runtimeWindow.nlpStackTextCore?.ready;
  if (existing) {
    runtimePromise = existing;
    return existing;
  }

  runtimePromise = waitForRuntimeRegistration().then(() => {
    const registered = (window as RuntimeWindow).nlpStackTextCore?.ready;
    if (!registered) {
      throw new Error("The text-core Wasm runtime registered without a ready promise.");
    }
    return registered;
  });

  ensureRuntimeScript();
  return runtimePromise;
}

function waitForRuntimeRegistration(): Promise<void> {
  return new Promise((resolve, reject) => {
    const errorEvent = "nlp-stack-text-core-error";
    const cleanup = () => {
      window.removeEventListener(runtimeReadyEvent, onReady);
      window.removeEventListener(errorEvent, onError);
    };
    const onReady = () => {
      cleanup();
      resolve();
    };
    const onError = () => {
      cleanup();
      reject(new Error("Failed to load the text-core Wasm runtime."));
    };

    window.addEventListener(runtimeReadyEvent, onReady, { once: true });
    window.addEventListener(errorEvent, onError, { once: true });
  });
}

function ensureRuntimeScript(): void {
  if (document.getElementById(runtimeScriptId)) {
    return;
  }

  const script = document.createElement("script");
  script.id = runtimeScriptId;
  script.type = "module";
  script.src = `${basePath}/nlp-runtime.js`;
  document.head.append(script);
}

const packageAppConfig = {
  library: "text-core",
  title: "Text Core",
  description: "Shared text documents, tokenization, spans, Unicode boundaries, and statistics.",
  domain: "text",
  wasm: {
    init: loadRuntime,
    packageSurface: async () => (await loadRuntime()).packageSurface(),
    runOperation: async (request: SurfaceRequest) => (await loadRuntime()).runOperation(request),
  },
  operationGroups: [
    {
      id: "text",
      label: "Text operations",
      description: "Run the deterministic text-core surface locally in WebAssembly.",
      operations: ["text.tokenize", "text.normalize", "text.statistics", "text.boundaries"],
    },
    {
      id: "inspect",
      label: "Inspect",
      description: "Inspect the package surface and supported operations.",
      operations: ["describe"],
    },
  ],
  defaultRuntime: "client-wasm",
  defaultOperation: "text.tokenize",
  defaultPresetId: "tokenize-browser-text",
  featuredOperations: ["text.tokenize", "text.normalize", "text.statistics", "text.boundaries", "describe"],
  presets: [
    {
      id: "tokenize-browser-text",
      label: "Tokenize browser text",
      operation: "text.tokenize",
      description: "Return span-aware tokens and the script profile from Rust.",
      input: {
        text: "Rust and WebAssembly keep the NLP path deterministic.",
        includePunctuation: true,
        lowercase: true,
      },
    },
    {
      id: "normalize-caption",
      label: "Normalize a caption",
      operation: "text.normalize",
      description: "Normalize casing and whitespace without leaving the browser.",
      input: {
        text: "  RUST   NLP   in the BROWSER  ",
        lowercase: true,
        normalizeWhitespace: true,
      },
    },
    {
      id: "inspect-boundaries",
      label: "Inspect Unicode boundaries",
      operation: "text.boundaries",
      description: "Return words, sentences, paragraphs, and grapheme boundaries.",
      input: {
        text: "Hello Berlin. Grüß Gott — NLP stays Unicode-safe.",
        keepApostrophes: true,
      },
    },
  ],
  workbench: {
    layout: "focused",
    inputChrome: "compact",
    showLandscapeContract: true,
    sidePanels: {
      runtime: false,
      models: false,
      files: false,
      support: false,
    },
  },
} satisfies PackageAppConfig;

export function TextCoreDemo() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
