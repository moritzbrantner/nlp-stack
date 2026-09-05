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

type TextAnalysisRuntime = {
  runOperation: (request: SurfaceRequest) => SurfaceResponse;
};

type RuntimeHandle = { ready: Promise<TextAnalysisRuntime> };
type RuntimeWindow = Window & { nlpStackTextAnalysis?: RuntimeHandle };
type JsonRecord = Record<string, unknown>;
type ResultTab = "overview" | "keywords" | "entities" | "linguistics" | "semantics" | "technical";

const runtimeReadyEvent = "nlp-stack-text-analysis-ready";
const runtimeErrorEvent = "nlp-stack-text-analysis-error";
const runtimeScriptId = "nlp-stack-text-analysis-runtime";
const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
const sampleText =
  "Alice presented the semantic search roadmap in Berlin. Rust text analysis extracts keywords, entities, linguistic evidence, and deterministic semantic structure. Bob asked how retrieval scales to larger corpora. Alice explained that exact similarity remains the deterministic baseline. The team agreed to keep browser decoding separate from NLP ownership.";

let runtimePromise: Promise<TextAnalysisRuntime> | null = null;

export function TextAnalysisStudio() {
  const fileInput = useRef<HTMLInputElement>(null);
  const [text, setText] = useState(sampleText);
  const [source, setSource] = useState<BrowserTextIngestResult | null>(null);
  const [ocrLanguage, setOcrLanguage] = useState<OcrLanguage>("eng+deu");
  const [phase, setPhase] = useState("Ready for text or a document.");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [documentReport, setDocumentReport] = useState<JsonRecord | null>(null);
  const [semanticReport, setSemanticReport] = useState<JsonRecord | null>(null);
  const [activeTab, setActiveTab] = useState<ResultTab>("overview");

  async function analyze(nextText = text, sourceLabel = source?.sourceLabel ?? "pasted-text") {
    const trimmed = nextText.trim();
    if (!trimmed) {
      setError("Add text or upload a document before running analysis.");
      return;
    }

    setBusy(true);
    setError(null);
    setPhase("Loading the text-analysis Rust/Wasm runtime…");
    try {
      const runtime = await loadRuntime();
      const id = documentId(sourceLabel);
      setPhase("Running document and semantic analysis in Rust/Wasm…");
      const [documentResponse, semanticResponse] = await Promise.all([
        Promise.resolve(
          runtime.runOperation({
            operation: "analysis.document",
            input: {
              id,
              text: trimmed,
              profile: "deterministic",
              keywordLimit: 16,
              summarySentences: 5,
              ngramSizes: [2, 3],
              shingleSizes: [3, 5],
              linguistics: { mode: "heuristicBalanced" },
              embedding: { mode: "hashed", dimensions: 128, useIdf: false },
            },
          }),
        ),
        Promise.resolve(
          runtime.runOperation({
            operation: "analysis.semantic-map",
            input: {
              id,
              text: trimmed,
              neighborsPerUnit: 4,
              neighborThreshold: 0.25,
              clusterThreshold: 0.6,
              includeLinguisticGraph: true,
              includeNeighborhoodEvidence: false,
            },
          }),
        ),
      ]);

      setDocumentReport(surfaceResult(documentResponse));
      setSemanticReport(surfaceResult(semanticResponse));
      setActiveTab("overview");
      setPhase("Analysis ready. Results were produced locally by text-analysis Wasm.");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to analyze this text.");
      setPhase("Analysis stopped.");
    } finally {
      setBusy(false);
    }
  }

  async function ingest(file: File) {
    setBusy(true);
    setError(null);
    setDocumentReport(null);
    setSemanticReport(null);
    try {
      const result = await ingestBrowserFile(
        file,
        { ocrLanguage, ocrScannedPdfPages: true },
        setPhase,
      );
      setText(result.text);
      setSource(result);
      setPhase(`Extracted text from ${result.sourceLabel}.`);
      setBusy(false);
      await analyze(result.text, result.sourceLabel);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to read this file.");
      setPhase("Document ingestion stopped.");
      setBusy(false);
    }
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    const file = event.dataTransfer.files?.[0];
    if (file) {
      void ingest(file);
    }
  }

  function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void analyze();
  }

  const lexical = asRecord(documentReport?.lexical);
  const core = asRecord(documentReport?.core);
  const enrichedStats = asRecord(documentReport?.enrichedStats);
  const linguistic = asRecord(documentReport?.linguistic);
  const semantic = asRecord(semanticReport?.semantic);
  const keywords = asRecordArray(lexical?.keywords);
  const phraseKeywords = asRecordArray(lexical?.phraseKeywords);
  const topTerms = asRecordArray(lexical?.topTerms);
  const entities = asRecordArray(lexical?.ruleEntities);
  const summary = asRecordArray(lexical?.extractiveSummary);
  const clusters = asRecordArray(semantic?.clusters);
  const units = asRecordArray(semantic?.units);
  const timeline = asRecordArray(semantic?.timeline);
  const unitsById = new Map(units.map((unit) => [stringValue(unit.id), unit]));

  return (
    <div className="grid gap-8">
      <form className="grid gap-5" onSubmit={onSubmit}>
        <div
          className="rounded-lg border border-dashed border-line bg-surface px-5 py-6"
          onDragOver={(event) => event.preventDefault()}
          onDrop={onDrop}
        >
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h3 className="text-base font-semibold text-ink">Drop a document or image</h3>
              <p className="mt-1 max-w-2xl text-sm leading-6 text-muted">
                Text, Markdown, CSV, JSON, XML and HTML are parsed directly. PDFs use their text layer and fall back to OCR for scanned pages. Images are OCRed in the browser.
              </p>
            </div>
            <button
              className="min-h-11 rounded-md bg-ink px-4 py-2 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-50"
              type="button"
              disabled={busy}
              onClick={() => fileInput.current?.click()}
            >
              Choose file
            </button>
          </div>
          <input
            ref={fileInput}
            className="sr-only"
            aria-label="Upload document"
            type="file"
            accept={browserTextFileAccept}
            disabled={busy}
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) {
                void ingest(file);
              }
              event.target.value = "";
            }}
          />
          <div className="mt-4 flex flex-wrap items-center gap-3 border-t border-line pt-4">
            <label className="text-sm font-medium text-ink" htmlFor="ocr-language">OCR language</label>
            <select
              id="ocr-language"
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
            {source ? (
              <span className="text-sm text-muted">
                {source.sourceLabel} · {source.method}
                {source.pageCount ? ` · ${source.pageCount} pages` : ""}
                {source.ocrPageCount ? ` · OCR on ${source.ocrPageCount} page${source.ocrPageCount === 1 ? "" : "s"}` : ""}
              </span>
            ) : null}
          </div>
        </div>

        <label className="grid gap-2" htmlFor="analysis-text">
          <span className="text-sm font-semibold text-ink">Text to analyze</span>
          <textarea
            id="analysis-text"
            className="min-h-64 w-full resize-y rounded-lg border border-line bg-surface px-4 py-3 text-base leading-7 text-ink outline-none focus:border-accent focus:ring-2 focus:ring-accent-soft"
            value={text}
            disabled={busy}
            onChange={(event) => {
              setText(event.target.value);
              setSource(null);
            }}
          />
        </label>

        <div className="flex flex-wrap items-center gap-4">
          <button
            className="min-h-11 rounded-md bg-accent px-5 py-2 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-50"
            type="submit"
            disabled={busy || !text.trim()}
          >
            {busy ? "Working…" : "Analyze text"}
          </button>
          <p className="text-sm text-muted" aria-live="polite">{phase}</p>
        </div>
        {error ? (
          <p className="rounded-md border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-900" role="alert">
            {error}
          </p>
        ) : null}
      </form>

      {documentReport ? (
        <section className="border-t border-line pt-7" aria-labelledby="analysis-results-heading">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <p className="text-sm font-semibold uppercase tracking-[0.12em] text-accent">Local Rust result</p>
              <h3 id="analysis-results-heading" className="mt-1 text-2xl font-semibold text-ink">Analysis</h3>
            </div>
            <p className="text-sm text-muted">No text-analysis API request is required.</p>
          </div>

          <div className="mt-5 flex gap-1 overflow-x-auto border-b border-line" role="tablist" aria-label="Analysis sections">
            {([
              ["overview", "Overview"],
              ["keywords", "Keywords"],
              ["entities", "Entities"],
              ["linguistics", "Linguistics"],
              ["semantics", "Semantics"],
              ["technical", "Technical"],
            ] as const).map(([id, label]) => (
              <button
                key={id}
                className={`min-h-11 whitespace-nowrap border-b-2 px-3 py-2 text-sm font-semibold ${activeTab === id ? "border-accent text-accent" : "border-transparent text-muted hover:text-ink"}`}
                type="button"
                role="tab"
                aria-selected={activeTab === id}
                onClick={() => setActiveTab(id)}
              >
                {label}
              </button>
            ))}
          </div>

          <div className="pt-6">
            {activeTab === "overview" ? (
              <OverviewPanel
                documentReport={documentReport}
                core={core}
                lexical={lexical}
                enrichedStats={enrichedStats}
                summary={summary}
                clusters={clusters}
              />
            ) : null}
            {activeTab === "keywords" ? (
              <KeywordsPanel keywords={keywords} phraseKeywords={phraseKeywords} topTerms={topTerms} />
            ) : null}
            {activeTab === "entities" ? <EntitiesPanel entities={entities} /> : null}
            {activeTab === "linguistics" ? <LinguisticsPanel linguistic={linguistic} /> : null}
            {activeTab === "semantics" ? (
              <SemanticsPanel clusters={clusters} timeline={timeline} unitsById={unitsById} semantic={semantic} />
            ) : null}
            {activeTab === "technical" ? (
              <TechnicalPanel documentReport={documentReport} semanticReport={semanticReport} />
            ) : null}
          </div>
        </section>
      ) : null}
    </div>
  );
}

function OverviewPanel({
  documentReport,
  core,
  lexical,
  enrichedStats,
  summary,
  clusters,
}: {
  documentReport: JsonRecord;
  core: JsonRecord | null;
  lexical: JsonRecord | null;
  enrichedStats: JsonRecord | null;
  summary: JsonRecord[];
  clusters: JsonRecord[];
}) {
  const tokens = asArray(core?.tokens).length;
  const sentences = asArray(core?.sentences).length;
  const paragraphs = asArray(core?.paragraphs).length;
  const scriptProfile = asRecord(core?.scriptProfile);
  const readability = asRecord(lexical?.readability);
  const sentiment = asRecord(lexical?.sentiment);

  return (
    <div className="grid gap-8">
      <section aria-labelledby="summary-heading">
        <h4 id="summary-heading" className="text-lg font-semibold text-ink">Extractive summary</h4>
        {summary.length ? (
          <ol className="mt-3 grid gap-3">
            {summary.map((item, index) => (
              <li key={`${stringValue(item.index)}-${index}`} className="border-l-2 border-line pl-4 text-sm leading-6 text-ink">
                {stringValue(item.text)}
              </li>
            ))}
          </ol>
        ) : <p className="mt-2 text-sm text-muted">No summary sentences were produced.</p>}
      </section>

      <section aria-labelledby="facts-heading">
        <h4 id="facts-heading" className="text-lg font-semibold text-ink">Document facts</h4>
        <dl className="mt-3 grid gap-x-8 gap-y-3 text-sm sm:grid-cols-2 lg:grid-cols-3">
          <Fact label="Language" value={stringValue(documentReport.language, "undetermined")} />
          <Fact label="Dominant script" value={stringValue(scriptProfile?.dominantScript, "undetermined")} />
          <Fact label="Tokens" value={String(tokens)} />
          <Fact label="Sentences" value={String(sentences)} />
          <Fact label="Paragraphs" value={String(paragraphs)} />
          <Fact label="Lexical density" value={formatNumber(enrichedStats?.lexicalDensity)} />
          <Fact label="Shannon entropy" value={formatNumber(enrichedStats?.shannonEntropy)} />
          <Fact label="Average sentence words" value={formatNumber(readability?.averageSentenceWords)} />
          <Fact label="Average word characters" value={formatNumber(readability?.averageWordChars)} />
        </dl>
      </section>

      <section className="grid gap-6 lg:grid-cols-2">
        <div>
          <h4 className="text-lg font-semibold text-ink">Sentiment evidence</h4>
          <JsonTable value={sentiment} empty="No lexical sentiment evidence." />
        </div>
        <div>
          <h4 className="text-lg font-semibold text-ink">Leading semantic concepts</h4>
          {clusters.length ? (
            <ul className="mt-3 grid gap-3">
              {clusters.slice(0, 6).map((cluster) => (
                <li key={stringValue(cluster.id)} className="text-sm leading-6">
                  <span className="font-medium text-ink">{stringValue(cluster.representativeText)}</span>
                  <span className="ml-2 text-muted">mean similarity {formatNumber(cluster.meanSimilarity)}</span>
                </li>
              ))}
            </ul>
          ) : <p className="mt-2 text-sm text-muted">No semantic clusters were produced.</p>}
        </div>
      </section>
    </div>
  );
}

function KeywordsPanel({ keywords, phraseKeywords, topTerms }: { keywords: JsonRecord[]; phraseKeywords: JsonRecord[]; topTerms: JsonRecord[] }) {
  return (
    <div className="grid gap-8 lg:grid-cols-3">
      <RankedTextList title="Keywords" items={keywords} />
      <RankedTextList title="Phrase keywords" items={phraseKeywords} />
      <RankedTextList title="Top terms" items={topTerms} />
    </div>
  );
}

function RankedTextList({ title, items }: { title: string; items: JsonRecord[] }) {
  return (
    <section>
      <h4 className="text-lg font-semibold text-ink">{title}</h4>
      {items.length ? (
        <ol className="mt-3 grid gap-2">
          {items.map((item, index) => (
            <li key={`${stringValue(item.text ?? item.term)}-${index}`} className="flex items-baseline justify-between gap-3 border-b border-line py-2 text-sm">
              <span className="font-medium text-ink">{stringValue(item.text ?? item.term ?? item.terms)}</span>
              <span className="shrink-0 text-muted">
                {typeof item.score === "number" ? formatNumber(item.score) : typeof item.count === "number" ? `${item.count}×` : ""}
              </span>
            </li>
          ))}
        </ol>
      ) : <p className="mt-2 text-sm text-muted">No items.</p>}
    </section>
  );
}

function EntitiesPanel({ entities }: { entities: JsonRecord[] }) {
  return (
    <section>
      <h4 className="text-lg font-semibold text-ink">Rule-based entity evidence</h4>
      <p className="mt-1 max-w-3xl text-sm leading-6 text-muted">These are deterministic mentions from the lexical analysis, not claims about a person or organization beyond the source text.</p>
      {entities.length ? (
        <div className="mt-4 overflow-x-auto">
          <table className="w-full min-w-[32rem] border-collapse text-left text-sm">
            <thead><tr className="border-b border-line text-muted"><th className="py-2 pr-4 font-medium">Mention</th><th className="py-2 pr-4 font-medium">Kind</th><th className="py-2 font-medium">Span</th></tr></thead>
            <tbody>
              {entities.map((entity, index) => (
                <tr key={`${stringValue(entity.text)}-${index}`} className="border-b border-line/70">
                  <td className="py-3 pr-4 font-medium text-ink">{stringValue(entity.text)}</td>
                  <td className="py-3 pr-4 text-muted">{stringValue(entity.kind)}</td>
                  <td className="py-3 text-muted"><JsonInline value={entity.span} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : <p className="mt-3 text-sm text-muted">No rule-based entity mentions were found.</p>}
    </section>
  );
}

function LinguisticsPanel({ linguistic }: { linguistic: JsonRecord | null }) {
  const sections = [
    ["language", "Language"], ["tokenizer", "Tokenizer"], ["lemmas", "Lemmas"], ["morphology", "Morphology"],
    ["pos", "Part of speech"], ["chunks", "Chunks"], ["dependencies", "Dependencies"], ["entities", "Linguistic entities"],
    ["canonicalEntities", "Canonical entities"], ["coreference", "Coreference"], ["events", "Events"], ["relations", "Relations"],
    ["discourse", "Discourse"], ["outline", "Outline"], ["topics", "Topics"], ["style", "Style"],
  ] as const;

  if (!linguistic) {
    return <p className="text-sm text-muted">No linguistic section was produced.</p>;
  }

  return (
    <div className="grid gap-3">
      {sections.map(([key, label]) => (
        <details key={key} className="rounded-md border border-line bg-surface px-4 py-3" open={key === "topics" || key === "style" || key === "outline"}>
          <summary className="cursor-pointer text-sm font-semibold text-ink">{label}</summary>
          <JsonBlock value={linguistic[key]} />
        </details>
      ))}
    </div>
  );
}

function SemanticsPanel({ clusters, timeline, unitsById, semantic }: { clusters: JsonRecord[]; timeline: JsonRecord[]; unitsById: Map<string, JsonRecord>; semantic: JsonRecord | null }) {
  return (
    <div className="grid gap-8">
      <section>
        <h4 className="text-lg font-semibold text-ink">Concept clusters</h4>
        <div className="mt-3 grid gap-4 lg:grid-cols-2">
          {clusters.map((cluster) => (
            <article key={stringValue(cluster.id)} className="border-l-2 border-line pl-4">
              <p className="text-sm font-semibold text-ink">{stringValue(cluster.representativeText)}</p>
              <p className="mt-1 text-xs leading-5 text-muted">
                {stringValue(cluster.id)} · {asArray(cluster.memberUnitIds).length} units · mean similarity {formatNumber(cluster.meanSimilarity)}
              </p>
            </article>
          ))}
        </div>
      </section>

      <section>
        <h4 className="text-lg font-semibold text-ink">Semantic trajectory</h4>
        <p className="mt-1 text-sm leading-6 text-muted">Ordered source units with their deterministic concept assignment and change from the preceding semantic state.</p>
        {timeline.length ? (
          <ol className="mt-4 grid gap-3">
            {timeline.map((point, index) => {
              const unit = unitsById.get(stringValue(point.unitId));
              return (
                <li key={`${stringValue(point.unitId)}-${index}`} className="grid gap-1 border-l-2 border-line pl-4 sm:grid-cols-[minmax(0,1fr)_auto] sm:gap-4">
                  <span className="text-sm leading-6 text-ink">{stringValue(unit?.text, stringValue(point.unitId))}</span>
                  <span className="text-xs text-muted">{stringValue(point.clusterId)} · shift {formatNumber(point.semanticShift)}</span>
                </li>
              );
            })}
          </ol>
        ) : <p className="mt-3 text-sm text-muted">No semantic timeline was produced.</p>}
      </section>

      <section>
        <h4 className="text-lg font-semibold text-ink">Hotspots and neighborhood evidence</h4>
        <JsonBlock value={{ hotspots: semantic?.hotspots, neighbors: semantic?.neighbors }} />
      </section>
    </div>
  );
}

function TechnicalPanel({ documentReport, semanticReport }: { documentReport: JsonRecord; semanticReport: JsonRecord | null }) {
  return (
    <div className="grid gap-4">
      <details className="rounded-md border border-line bg-surface px-4 py-3" open>
        <summary className="cursor-pointer text-sm font-semibold text-ink">Document analysis JSON</summary>
        <JsonBlock value={documentReport} />
      </details>
      <details className="rounded-md border border-line bg-surface px-4 py-3">
        <summary className="cursor-pointer text-sm font-semibold text-ink">Semantic analysis JSON</summary>
        <JsonBlock value={semanticReport} />
      </details>
    </div>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return <div className="border-b border-line pb-2"><dt className="text-muted">{label}</dt><dd className="mt-1 font-medium text-ink">{value}</dd></div>;
}

function JsonTable({ value, empty }: { value: JsonRecord | null; empty: string }) {
  if (!value || Object.keys(value).length === 0) {
    return <p className="mt-2 text-sm text-muted">{empty}</p>;
  }
  return (
    <dl className="mt-3 grid gap-2 text-sm">
      {Object.entries(value).map(([key, entry]) => (
        <div key={key} className="flex items-start justify-between gap-4 border-b border-line py-2">
          <dt className="text-muted">{humanize(key)}</dt>
          <dd className="max-w-[65%] text-right font-medium text-ink"><JsonInline value={entry} /></dd>
        </div>
      ))}
    </dl>
  );
}

function JsonInline({ value }: { value: unknown }) {
  if (value == null) return <>—</>;
  if (typeof value === "number") return <>{formatNumber(value)}</>;
  if (typeof value === "string" || typeof value === "boolean") return <>{String(value)}</>;
  return <>{JSON.stringify(value)}</>;
}

function JsonBlock({ value }: { value: unknown }) {
  return (
    <pre className="mt-3 max-h-[36rem] overflow-auto whitespace-pre-wrap break-words rounded-md bg-zinc-950 p-4 text-xs leading-5 text-zinc-100">
      {JSON.stringify(value ?? null, null, 2)}
    </pre>
  );
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
    if (!registered) throw new Error("The text-analysis Wasm runtime registered without a ready promise.");
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
    const onReady = () => { cleanup(); resolve(); };
    const onError = () => { cleanup(); reject(new Error("Failed to load the text-analysis Wasm runtime.")); };
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

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function asRecordArray(value: unknown): JsonRecord[] {
  return asArray(value).map(asRecord).filter((entry): entry is JsonRecord => entry !== null);
}

function stringValue(value: unknown, fallback = "—"): string {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) return value.map((entry) => stringValue(entry, "")).filter(Boolean).join(" ");
  return fallback;
}

function formatNumber(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value)
    ? new Intl.NumberFormat(undefined, { maximumFractionDigits: 3 }).format(value)
    : "—";
}

function humanize(value: string): string {
  return value.replace(/([a-z])([A-Z])/g, "$1 $2").replace(/^./, (letter) => letter.toUpperCase());
}

function documentId(sourceLabel: string): string {
  return sourceLabel.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 64) || "browser-text";
}
