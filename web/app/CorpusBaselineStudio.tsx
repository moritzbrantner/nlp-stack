"use client";

import {
  useRef,
  useState,
  type DragEvent,
  type FormEvent,
} from "react";

import type {
  SurfaceRequest,
  SurfaceResponse,
} from "../../packages/nlp-app-ui/dist/package-surface/index.js";
import {
  browserTextFileAccept,
  ingestBrowserFile,
  type BrowserTextIngestResult,
  type OcrLanguage,
} from "./browser-text-ingest";
import {
  corpusBaselineFilename,
  createCorpusBaseline,
  serializeCorpusBaseline,
  type CorpusBaseline,
  type SemanticCorpusBaselineOptions,
} from "./corpus-baseline";

type TextAnalysisRuntime = {
  runOperation: (request: SurfaceRequest) => SurfaceResponse;
};

type RuntimeHandle = { ready: Promise<TextAnalysisRuntime> };
type RuntimeWindow = Window & { nlpStackTextAnalysis?: RuntimeHandle };
type JsonRecord = Record<string, unknown>;

const runtimeReadyEvent = "nlp-stack-text-analysis-ready";
const runtimeErrorEvent = "nlp-stack-text-analysis-error";
const runtimeScriptId = "nlp-stack-text-analysis-runtime";
const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

let runtimePromise: Promise<TextAnalysisRuntime> | null = null;

export function CorpusBaselineStudio() {
  const fileInput = useRef<HTMLInputElement>(null);
  const [label, setLabel] = useState("My semantic corpus");
  const [manualSource, setManualSource] = useState("pasted-text");
  const [manualText, setManualText] = useState("");
  const [sources, setSources] = useState<BrowserTextIngestResult[]>([]);
  const [ocrLanguage, setOcrLanguage] = useState<OcrLanguage>("eng+deu");
  const [baseline, setBaseline] = useState<CorpusBaseline | null>(null);
  const [busy, setBusy] = useState(false);
  const [phase, setPhase] = useState("Add documents or pasted text to define the baseline corpus.");
  const [error, setError] = useState<string | null>(null);

  function corpusChanged(nextSources: BrowserTextIngestResult[], message: string) {
    setSources(nextSources);
    setBaseline(null);
    setError(null);
    setPhase(message);
  }

  async function ingest(files: File[]) {
    if (files.length === 0) return;
    setBusy(true);
    setError(null);
    setBaseline(null);

    try {
      const ingested: BrowserTextIngestResult[] = [];
      for (let index = 0; index < files.length; index += 1) {
        const file = files[index]!;
        const result = await ingestBrowserFile(
          file,
          { ocrLanguage, ocrScannedPdfPages: true },
          (message) => setPhase(`${index + 1}/${files.length} · ${message}`),
        );
        ingested.push(result);
      }

      const nextSources = [...sources, ...ingested];
      corpusChanged(
        nextSources,
        `Added ${ingested.length} source${ingested.length === 1 ? "" : "s"}. Build the baseline to capture text plus semantic evidence.`,
      );
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to read the selected files.");
      setPhase("Document ingestion stopped.");
    } finally {
      setBusy(false);
    }
  }

  function addPastedText(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const text = manualText.trim();
    if (!text) {
      setError("Paste text before adding it to the corpus.");
      return;
    }

    const sourceLabel = manualSource.trim() || `pasted-text-${sources.length + 1}`;
    corpusChanged(
      [...sources, { sourceLabel, text, method: "text" }],
      `Added ${sourceLabel}. Build the baseline to refresh semantic evidence.`,
    );
    setManualText("");
  }

  async function buildBaseline() {
    if (sources.length === 0) {
      setError("Add at least one source before building a corpus baseline.");
      return;
    }

    const items = sources.map((source, index) => ({
      id: `${documentId(source.sourceLabel)}-${index + 1}`,
      source: source.sourceLabel,
      text: source.text.trim(),
    }));
    const options = semanticCorpusOptions(items.reduce((sum, item) => sum + item.text.length, 0));

    setBusy(true);
    setError(null);
    setPhase("Running semantic corpus analysis in Rust/Wasm…");

    try {
      const runtime = await loadRuntime();
      const response = await Promise.resolve(
        runtime.runOperation({
          operation: "analysis.semantic-corpus",
          input: {
            items,
            ...options,
          },
        }),
      );
      const result = surfaceResult(response);
      const nextBaseline = createCorpusBaseline({
        label,
        documents: sources.map((source, index) => ({
          id: items[index]!.id,
          source: source.sourceLabel,
          text: source.text,
          ingestion: {
            method: source.method,
            ...(source.pageCount ? { pageCount: source.pageCount } : {}),
            ...(source.ocrPageCount ? { ocrPageCount: source.ocrPageCount } : {}),
          },
        })),
        options,
        result,
      });

      setBaseline(nextBaseline);
      setPhase(
        `Baseline ready with ${sources.length} source${sources.length === 1 ? "" : "s"}. Exporting preserves the extracted text, analysis options, and Rust semantic-corpus result.`,
      );
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to build the corpus baseline.");
      setPhase("Baseline analysis stopped.");
    } finally {
      setBusy(false);
    }
  }

  function exportBaseline() {
    if (!baseline) return;

    const blob = new Blob([serializeCorpusBaseline(baseline)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = corpusBaselineFilename(baseline.label);
    document.body.append(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    const files = Array.from(event.dataTransfer.files ?? []);
    if (files.length > 0) void ingest(files);
  }

  return (
    <div className="grid gap-6">
      <label className="grid max-w-xl gap-2" htmlFor="corpus-baseline-label">
        <span className="text-sm font-semibold text-ink">Baseline label</span>
        <input
          id="corpus-baseline-label"
          className="min-h-11 rounded-md border border-line bg-surface px-3 text-base text-ink outline-none focus:border-accent focus:ring-2 focus:ring-accent-soft sm:text-sm"
          value={label}
          disabled={busy}
          onChange={(event) => {
            setLabel(event.target.value);
            setBaseline(null);
            setPhase("Baseline label changed. Rebuild before exporting.");
          }}
        />
      </label>

      <div
        className="rounded-lg border border-dashed border-line bg-surface px-5 py-6"
        onDragOver={(event) => event.preventDefault()}
        onDrop={onDrop}
      >
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h4 className="text-base font-semibold text-ink">Add source documents</h4>
            <p className="mt-1 max-w-2xl text-sm leading-6 text-muted">
              Text, Markdown, CSV, JSON, XML and HTML are read directly. PDFs and images can use browser OCR. Files stay local to this page.
            </p>
          </div>
          <button
            className="min-h-11 rounded-md bg-ink px-4 py-2 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-50"
            type="button"
            disabled={busy}
            onClick={() => fileInput.current?.click()}
          >
            Choose documents
          </button>
        </div>
        <input
          ref={fileInput}
          className="sr-only"
          aria-label="Add corpus documents"
          type="file"
          accept={browserTextFileAccept}
          multiple
          disabled={busy}
          onChange={(event) => {
            const files = Array.from(event.target.files ?? []);
            if (files.length > 0) void ingest(files);
            event.target.value = "";
          }}
        />
        <div className="mt-4 flex flex-wrap items-center gap-3 border-t border-line pt-4">
          <label className="text-sm font-medium text-ink" htmlFor="corpus-ocr-language">OCR language</label>
          <select
            id="corpus-ocr-language"
            className="min-h-11 rounded-md border border-line bg-surface px-3 text-base text-ink sm:text-sm"
            value={ocrLanguage}
            disabled={busy}
            onChange={(event) => setOcrLanguage(event.target.value as OcrLanguage)}
          >
            <option value="eng+deu">English + German</option>
            <option value="eng">English</option>
            <option value="deu">German</option>
            <option value="spa">Spanish</option>
          </select>
        </div>
      </div>

      <form className="grid gap-3 rounded-lg border border-line bg-surface p-5" onSubmit={addPastedText}>
        <div>
          <h4 className="text-base font-semibold text-ink">Or add pasted text</h4>
          <p className="mt-1 text-sm leading-6 text-muted">
            Give the text a source label so the exported semantic evidence keeps its provenance.
          </p>
        </div>
        <label className="grid gap-2" htmlFor="manual-corpus-source">
          <span className="text-sm font-medium text-ink">Source label</span>
          <input
            id="manual-corpus-source"
            className="min-h-11 rounded-md border border-line bg-white px-3 text-base text-ink sm:text-sm"
            value={manualSource}
            disabled={busy}
            onChange={(event) => setManualSource(event.target.value)}
          />
        </label>
        <label className="grid gap-2" htmlFor="manual-corpus-text">
          <span className="text-sm font-medium text-ink">Text</span>
          <textarea
            id="manual-corpus-text"
            className="min-h-40 resize-y rounded-md border border-line bg-white px-3 py-2 text-base leading-7 text-ink sm:text-sm"
            value={manualText}
            disabled={busy}
            onChange={(event) => setManualText(event.target.value)}
          />
        </label>
        <button
          className="min-h-11 justify-self-start rounded-md border border-line bg-white px-4 py-2 text-sm font-semibold text-ink disabled:cursor-not-allowed disabled:opacity-50"
          type="submit"
          disabled={busy || !manualText.trim()}
        >
          Add text to corpus
        </button>
      </form>

      {sources.length > 0 ? (
        <section aria-labelledby="baseline-sources-heading">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div>
              <h4 id="baseline-sources-heading" className="text-base font-semibold text-ink">Baseline sources</h4>
              <p className="mt-1 text-sm text-muted">Order is preserved in the exported baseline and semantic analysis request.</p>
            </div>
            <button
              className="min-h-11 rounded-md px-3 py-2 text-sm font-semibold text-muted hover:text-ink disabled:opacity-50"
              type="button"
              disabled={busy}
              onClick={() => corpusChanged([], "Corpus cleared. Add sources to define a new baseline.")}
            >
              Clear
            </button>
          </div>
          <ol className="mt-3 grid gap-2">
            {sources.map((source, index) => (
              <li key={`${source.sourceLabel}-${index}`} className="flex flex-wrap items-center justify-between gap-3 border-b border-line py-3 text-sm">
                <div className="min-w-0">
                  <p className="truncate font-medium text-ink">{source.sourceLabel}</p>
                  <p className="mt-1 text-xs text-muted">
                    {source.method}
                    {source.pageCount ? ` · ${source.pageCount} pages` : ""}
                    {source.ocrPageCount ? ` · OCR on ${source.ocrPageCount} page${source.ocrPageCount === 1 ? "" : "s"}` : ""}
                  </p>
                </div>
                <button
                  className="min-h-11 px-2 text-sm font-semibold text-muted hover:text-ink disabled:opacity-50"
                  type="button"
                  disabled={busy}
                  onClick={() => corpusChanged(
                    sources.filter((_, sourceIndex) => sourceIndex !== index),
                    `${source.sourceLabel} removed. Rebuild the baseline before exporting.`,
                  )}
                >
                  Remove
                </button>
              </li>
            ))}
          </ol>
        </section>
      ) : null}

      <div className="flex flex-wrap items-center gap-3 border-t border-line pt-5">
        <button
          className="min-h-11 rounded-md bg-accent px-5 py-2 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-50"
          type="button"
          disabled={busy || sources.length === 0}
          onClick={() => void buildBaseline()}
        >
          {busy ? "Working…" : "Build semantic baseline"}
        </button>
        <button
          className="min-h-11 rounded-md border border-line bg-surface px-5 py-2 text-sm font-semibold text-ink disabled:cursor-not-allowed disabled:opacity-50"
          type="button"
          disabled={!baseline || busy}
          onClick={exportBaseline}
        >
          Export JSON
        </button>
        <p className="text-sm text-muted" aria-live="polite">{phase}</p>
      </div>

      {baseline ? (
        <p className="rounded-md border border-line bg-surface px-4 py-3 text-sm leading-6 text-muted">
          Export format <code className="font-mono text-ink">{baseline.kind}</code> schema {baseline.schemaVersion}. The file intentionally contains the original extracted text, source provenance, deterministic analysis options, and the Rust semantic-corpus result; it contains no generated timestamp.
        </p>
      ) : null}

      {error ? (
        <p className="rounded-md border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-900" role="alert">{error}</p>
      ) : null}
    </div>
  );
}

function semanticCorpusOptions(totalTextCharacters: number): SemanticCorpusBaselineOptions {
  return {
    topTerms: Math.max(64, Math.min(totalTextCharacters, 5000)),
    minConceptUnits: 2,
    neighborsPerUnit: 4,
    neighborThreshold: 0.25,
    clusterThreshold: 0.6,
  };
}

async function loadRuntime(): Promise<TextAnalysisRuntime> {
  if (runtimePromise) return runtimePromise;
  const runtimeWindow = window as RuntimeWindow;
  if (runtimeWindow.nlpStackTextAnalysis?.ready) {
    runtimePromise = runtimeWindow.nlpStackTextAnalysis.ready;
    return runtimePromise;
  }

  runtimePromise = waitForRuntimeRegistration().then(() => {
    const registered = (window as RuntimeWindow).nlpStackTextAnalysis?.ready;
    if (!registered) {
      throw new Error("The text-analysis Wasm runtime registered without a ready promise.");
    }
    return registered;
  });
  ensureRuntimeScript();
  return runtimePromise;
}

function waitForRuntimeRegistration(): Promise<void> {
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      window.removeEventListener(runtimeReadyEvent, onReady);
      window.removeEventListener(runtimeErrorEvent, onError);
    };
    const onReady = () => {
      cleanup();
      resolve();
    };
    const onError = () => {
      cleanup();
      reject(new Error("Failed to load the text-analysis Wasm runtime."));
    };

    window.addEventListener(runtimeReadyEvent, onReady, { once: true });
    window.addEventListener(runtimeErrorEvent, onError, { once: true });
  });
}

function ensureRuntimeScript() {
  if (document.getElementById(runtimeScriptId)) return;
  const script = document.createElement("script");
  script.id = runtimeScriptId;
  script.type = "module";
  script.src = `${basePath}/nlp-analysis-runtime.js`;
  document.head.append(script);
}

function surfaceResult(response: SurfaceResponse): JsonRecord {
  const value = asRecord(response.value);
  const nested = asRecord(value?.result);
  return nested ?? value ?? {};
}

function asRecord(value: unknown): JsonRecord | null {
  return value !== null && typeof value === "object" && !Array.isArray(value) ? value as JsonRecord : null;
}

function documentId(sourceLabel: string): string {
  return sourceLabel.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 64) || "browser-text";
}
