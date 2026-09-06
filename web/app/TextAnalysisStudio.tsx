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
  defaultTextAnalysisExample,
  textAnalysisExamples,
  type ExampleResultView,
  type TextAnalysisExample,
} from "./text-analysis-examples";

type TextAnalysisRuntime = {
  runOperation: (request: SurfaceRequest) => SurfaceResponse;
};

type RuntimeHandle = { ready: Promise<TextAnalysisRuntime> };
type RuntimeWindow = Window & { nlpStackTextAnalysis?: RuntimeHandle };
type JsonRecord = Record<string, unknown>;
type ResultTab = ExampleResultView | "entities" | "technical";

const runtimeReadyEvent = "nlp-stack-text-analysis-ready";
const runtimeErrorEvent = "nlp-stack-text-analysis-error";
const runtimeScriptId = "nlp-stack-text-analysis-runtime";
const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
const resultGroups: { label: string; tabs: [ResultTab, string][] }[] = [
  {
    label: "Corpus",
    tabs: [
      ["word-corpus", "Word profile"],
      ["semantic-corpus", "Corpus themes"],
    ],
  },
  {
    label: "Document",
    tabs: [
      ["overview", "Overview"],
      ["keywords", "Keywords"],
      ["entities", "Entities"],
      ["linguistics", "Linguistics"],
      ["semantic-map", "Semantic map"],
    ],
  },
  { label: "Inspect", tabs: [["technical", "Technical"]] },
];

let runtimePromise: Promise<TextAnalysisRuntime> | null = null;

export function TextAnalysisStudio() {
  const fileInput = useRef<HTMLInputElement>(null);
  const [text, setText] = useState(defaultTextAnalysisExample.text);
  const [selectedExampleId, setSelectedExampleId] = useState<string | null>(defaultTextAnalysisExample.id);
  const [source, setSource] = useState<BrowserTextIngestResult | null>(null);
  const [corpusSources, setCorpusSources] = useState<BrowserTextIngestResult[]>([]);
  const [analysisSourceLabel, setAnalysisSourceLabel] = useState<string | null>(null);
  const [analysisCorpusSourceLabels, setAnalysisCorpusSourceLabels] = useState<string[]>([]);
  const [ocrLanguage, setOcrLanguage] = useState<OcrLanguage>("eng+deu");
  const [phase, setPhase] = useState("Choose an example, paste text, or upload one or more documents.");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [documentReport, setDocumentReport] = useState<JsonRecord | null>(null);
  const [semanticReport, setSemanticReport] = useState<JsonRecord | null>(null);
  const [corpusReport, setCorpusReport] = useState<JsonRecord | null>(null);
  const [activeTab, setActiveTab] = useState<ResultTab>(defaultTextAnalysisExample.focus);

  const selectedExample = selectedExampleId
    ? textAnalysisExamples.find((example) => example.id === selectedExampleId) ?? null
    : null;

  function clearResults() {
    setDocumentReport(null);
    setSemanticReport(null);
    setCorpusReport(null);
    setAnalysisSourceLabel(null);
    setAnalysisCorpusSourceLabels([]);
  }

  function selectExample(example: TextAnalysisExample) {
    setSelectedExampleId(example.id);
    setText(example.text);
    setSource(null);
    setCorpusSources([]);
    setError(null);
    clearResults();
    setActiveTab(example.focus);
    setPhase(`${example.label} loaded. Run analysis to inspect ${example.demonstrates}.`);
  }

  async function analyze(
    nextText = text,
    sourceLabel = source?.sourceLabel ?? selectedExample?.label ?? "pasted-text",
    focus: ResultTab = selectedExample?.focus ?? "word-corpus",
    suppliedCorpusSources: BrowserTextIngestResult[] = corpusSources,
  ) {
    const trimmed = nextText.trim();
    if (!trimmed) {
      setError("Add text or upload a document before running analysis.");
      return;
    }

    const corpusItems = suppliedCorpusSources.length > 0
      ? suppliedCorpusSources.map((item, index) => ({
          id: `${documentId(item.sourceLabel)}-${index + 1}`,
          source: item.sourceLabel,
          text: item.text.trim(),
        }))
      : [{ id: documentId(sourceLabel), source: sourceLabel, text: trimmed }];
    const isMultiDocumentCorpus = corpusItems.length > 1;
    const resolvedFocus = focus === "semantic-corpus" && !isMultiDocumentCorpus ? "semantic-map" : focus;

    setBusy(true);
    setError(null);
    setPhase("Loading the text-analysis Rust/Wasm runtime…");
    try {
      const runtime = await loadRuntime();
      const id = documentId(sourceLabel);
      setPhase(
        isMultiDocumentCorpus
          ? `Running document analysis and corpus themes across ${corpusItems.length} sources in Rust/Wasm…`
          : "Running document and semantic-map analysis in Rust/Wasm…",
      );
      const [documentResponse, semanticResponse, corpusResponse] = await Promise.all([
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
        Promise.resolve(
          runtime.runOperation({
            operation: "analysis.semantic-corpus",
            input: {
              items: corpusItems,
              topTerms: Math.max(64, Math.min(corpusItems.reduce((sum, item) => sum + item.text.length, 0), 5000)),
              minConceptUnits: 2,
              neighborsPerUnit: 4,
              neighborThreshold: 0.25,
              clusterThreshold: 0.6,
            },
          }),
        ),
      ]);

      setDocumentReport(surfaceResult(documentResponse));
      setSemanticReport(surfaceResult(semanticResponse));
      setCorpusReport(surfaceResult(corpusResponse));
      setAnalysisSourceLabel(sourceLabel);
      setAnalysisCorpusSourceLabels(corpusItems.map((item) => item.source));
      setActiveTab(resolvedFocus);
      setPhase(
        isMultiDocumentCorpus
          ? `Analysis ready. Corpus themes use ${corpusItems.length} supplied sources; document tabs use ${sourceLabel}.`
          : "Analysis ready. Single-document semantic structure is available in Semantic map; add multiple documents for Corpus themes.",
      );
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to analyze this text.");
      setPhase("Analysis stopped.");
    } finally {
      setBusy(false);
    }
  }

  async function ingest(files: File[]) {
    if (files.length === 0) return;
    setBusy(true);
    setError(null);
    setSelectedExampleId(null);
    clearResults();
    try {
      const results: BrowserTextIngestResult[] = [];
      for (let index = 0; index < files.length; index += 1) {
        const file = files[index]!;
        const result = await ingestBrowserFile(
          file,
          { ocrLanguage, ocrScannedPdfPages: true },
          (message) => setPhase(`${index + 1}/${files.length} · ${message}`),
        );
        results.push(result);
      }

      const first = results[0]!;
      setText(first.text);
      setSource(first);
      setCorpusSources(results);
      setPhase(
        results.length > 1
          ? `Extracted ${results.length} documents. ${first.sourceLabel} is the active document.`
          : `Extracted text from ${first.sourceLabel}.`,
      );
      setBusy(false);
      await analyze(
        first.text,
        first.sourceLabel,
        results.length > 1 ? "semantic-corpus" : "word-corpus",
        results,
      );
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to read the selected files.");
      setPhase("Document ingestion stopped.");
      setCorpusSources([]);
      setBusy(false);
    }
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    const files = Array.from(event.dataTransfer.files ?? []);
    if (files.length > 0) void ingest(files);
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
  const corpusLexical = asRecord(corpusReport?.lexical);
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
        <ExampleSelector
          selectedExampleId={selectedExampleId}
          disabled={busy}
          onSelect={selectExample}
        />

        <div
          className="rounded-lg border border-dashed border-line bg-surface px-5 py-6"
          onDragOver={(event) => event.preventDefault()}
          onDrop={onDrop}
        >
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h3 className="text-base font-semibold text-ink">Or use your own documents</h3>
              <p className="mt-1 max-w-2xl text-sm leading-6 text-muted">
                Upload one document for document analysis, or select/drop several together for genuine cross-source corpus themes. Text, Markdown, CSV, JSON, XML and HTML are parsed directly; PDFs and images can use browser OCR.
              </p>
            </div>
            <button
              className="min-h-11 rounded-md bg-ink px-4 py-2 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-50"
              type="button"
              disabled={busy}
              onClick={() => fileInput.current?.click()}
            >
              Choose files
            </button>
          </div>
          <input
            ref={fileInput}
            className="sr-only"
            aria-label="Upload documents"
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
            {source && corpusSources.length <= 1 ? (
              <span className="text-sm text-muted">
                {source.sourceLabel} · {source.method}
                {source.pageCount ? ` · ${source.pageCount} pages` : ""}
                {source.ocrPageCount ? ` · OCR on ${source.ocrPageCount} page${source.ocrPageCount === 1 ? "" : "s"}` : ""}
              </span>
            ) : null}
            {corpusSources.length > 1 ? (
              <span className="text-sm font-medium text-accent">{corpusSources.length} documents selected for corpus themes</span>
            ) : null}
          </div>
          {corpusSources.length > 1 ? (
            <div className="mt-3 flex flex-wrap gap-2" aria-label="Corpus sources">
              {corpusSources.map((item, index) => (
                <span
                  key={`${item.sourceLabel}-${index}`}
                  className={`rounded-full border px-2.5 py-1 text-xs ${index === 0 ? "border-accent bg-accent-soft text-accent" : "border-line bg-white text-muted"}`}
                >
                  {item.sourceLabel}{index === 0 ? " · active document" : ""}
                </span>
              ))}
            </div>
          ) : null}
        </div>

        <label className="grid gap-2" htmlFor="analysis-text">
          <span className="flex flex-wrap items-center justify-between gap-2 text-sm font-semibold text-ink">
            <span>Text to analyze</span>
            {selectedExample ? <span className="font-normal text-muted">Example: {selectedExample.label}</span> : null}
            {corpusSources.length > 1 ? <span className="font-normal text-muted">Active document: {corpusSources[0]?.sourceLabel}</span> : null}
          </span>
          <textarea
            id="analysis-text"
            className="min-h-64 w-full resize-y rounded-lg border border-line bg-surface px-4 py-3 text-base leading-7 text-ink outline-none focus:border-accent focus:ring-2 focus:ring-accent-soft"
            value={text}
            disabled={busy}
            onChange={(event) => {
              setText(event.target.value);
              setSelectedExampleId(null);
              setSource(null);
              setCorpusSources([]);
              clearResults();
              setPhase("Edited text ready for single-document analysis. Upload multiple files together for corpus themes.");
            }}
          />
        </label>

        <div className="flex flex-wrap items-center gap-4">
          <button
            className="min-h-11 rounded-md bg-accent px-5 py-2 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-50"
            type="submit"
            disabled={busy || !text.trim()}
          >
            {busy ? "Working…" : corpusSources.length > 1 ? "Analyze corpus" : "Analyze text"}
          </button>
          <p className="text-sm text-muted" aria-live="polite">{phase}</p>
        </div>
        {error ? (
          <p className="rounded-md border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-900" role="alert">{error}</p>
        ) : null}
      </form>

      {documentReport ? (
        <section className="border-t border-line pt-7" aria-labelledby="analysis-results-heading">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <p className="text-sm font-semibold uppercase tracking-[0.12em] text-accent">Local Rust result</p>
              <h3 id="analysis-results-heading" className="mt-1 text-2xl font-semibold text-ink">Analysis</h3>
              {analysisCorpusSourceLabels.length > 1 ? (
                <p className="mt-1 text-sm text-muted">
                  Corpus: {analysisCorpusSourceLabels.length} sources · active document: {analysisSourceLabel}
                </p>
              ) : analysisSourceLabel ? <p className="mt-1 text-sm text-muted">Source: {analysisSourceLabel}</p> : null}
            </div>
            <p className="max-w-xl text-sm leading-6 text-muted">
              Rust/Wasm owns document evidence and corpus statistics. Corpus themes are only presented as cross-source evidence when multiple documents were supplied; the browser owns ingestion and presentation.
            </p>
          </div>

          <ResultNavigation activeTab={activeTab} onSelect={setActiveTab} />

          <div className="pt-6">
            {activeTab === "word-corpus" ? <WordCorpusPanel lexical={corpusLexical} /> : null}
            {activeTab === "semantic-corpus" ? <SemanticCorpusPanel report={corpusReport} /> : null}
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
            {activeTab === "keywords" ? <KeywordsPanel keywords={keywords} phraseKeywords={phraseKeywords} topTerms={topTerms} /> : null}
            {activeTab === "entities" ? <EntitiesPanel entities={entities} /> : null}
            {activeTab === "linguistics" ? <LinguisticsPanel linguistic={linguistic} /> : null}
            {activeTab === "semantic-map" ? <SemanticMapPanel clusters={clusters} timeline={timeline} unitsById={unitsById} semantic={semantic} /> : null}
            {activeTab === "technical" ? <TechnicalPanel documentReport={documentReport} semanticReport={semanticReport} corpusReport={corpusReport} /> : null}
          </div>
        </section>
      ) : null}
    </div>
  );
}

function ExampleSelector({
  selectedExampleId,
  disabled,
  onSelect,
}: {
  selectedExampleId: string | null;
  disabled: boolean;
  onSelect: (example: TextAnalysisExample) => void;
}) {
  return (
    <section aria-labelledby="examples-heading">
      <div className="mb-3 flex flex-wrap items-end justify-between gap-2">
        <div>
          <h3 id="examples-heading" className="text-base font-semibold text-ink">Try an example</h3>
          <p className="mt-1 text-sm leading-6 text-muted">Each sample stresses a different part of the single-document NLP surface. Upload multiple files together to exercise corpus themes.</p>
        </div>
        <span className="text-xs text-muted">{textAnalysisExamples.length} examples</span>
      </div>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {textAnalysisExamples.map((example) => {
          const selected = selectedExampleId === example.id;
          return (
            <button
              key={example.id}
              className={`min-h-32 rounded-lg border p-4 text-left transition ${selected ? "border-accent bg-accent-soft" : "border-line bg-surface hover:border-zinc-400"}`}
              type="button"
              disabled={disabled}
              aria-pressed={selected}
              onClick={() => onSelect(example)}
            >
              <span className="text-xs font-semibold uppercase tracking-[0.1em] text-muted">{example.category}</span>
              <span className="mt-1 block text-base font-semibold text-ink">{example.label}</span>
              <span className="mt-2 block text-sm leading-5 text-muted">{example.description}</span>
              <span className="mt-3 block text-xs font-medium text-accent">Shows: {example.demonstrates}</span>
            </button>
          );
        })}
      </div>
    </section>
  );
}

function ResultNavigation({ activeTab, onSelect }: { activeTab: ResultTab; onSelect: (tab: ResultTab) => void }) {
  return (
    <div className="mt-6 flex gap-7 overflow-x-auto border-b border-line pb-1" role="tablist" aria-label="Analysis sections">
      {resultGroups.map((group) => (
        <div key={group.label} className="min-w-max" role="group" aria-label={group.label}>
          <p className="px-2 text-[0.68rem] font-semibold uppercase tracking-[0.12em] text-muted">{group.label}</p>
          <div className="mt-1 flex gap-1">
            {group.tabs.map(([id, label]) => (
              <button
                key={id}
                className={`min-h-11 whitespace-nowrap border-b-2 px-3 py-2 text-sm font-semibold ${activeTab === id ? "border-accent text-accent" : "border-transparent text-muted hover:text-ink"}`}
                type="button"
                role="tab"
                aria-selected={activeTab === id}
                onClick={() => onSelect(id)}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function WordCorpusPanel({ lexical }: { lexical: JsonRecord | null }) {
  const terms = asRecordArray(lexical?.topTerms);
  const itemCount = numberValue(lexical?.itemCount);
  const uniqueTerms = numberValue(lexical?.uniqueTerms);
  const maxFrequency = terms.reduce((max, term) => Math.max(max, numberValue(term.frequency)), 0);
  return (
    <div className="grid gap-7">
      <section>
        <p className="text-xs font-semibold uppercase tracking-[0.12em] text-muted">{itemCount > 1 ? "Corpus view" : "Lexical view"}</p>
        <h4 className="mt-1 text-xl font-semibold text-ink">{itemCount > 1 ? "Word corpus" : "Word profile"}</h4>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-muted">
          {itemCount > 1
            ? "Corpus-wide normalized term frequencies from Rust across all supplied documents."
            : "Normalized term frequencies for the current document. Upload multiple documents together for corpus-wide lexical evidence."}
          {" "}Bars compare relative frequency within this result; they do not introduce a second scoring algorithm.
        </p>
        <dl className="mt-4 grid gap-x-8 gap-y-3 text-sm sm:grid-cols-2 lg:grid-cols-4">
          <Fact label="Items" value={formatInteger(lexical?.itemCount)} />
          <Fact label="Words" value={formatInteger(lexical?.wordCount)} />
          <Fact label="Unique terms" value={formatInteger(lexical?.uniqueTerms)} />
          <Fact label="Lexical diversity" value={formatNumber(lexical?.lexicalDiversity)} />
        </dl>
        {uniqueTerms > terms.length ? (
          <p className="mt-3 text-xs text-muted">Showing the {terms.length.toLocaleString()} highest-frequency terms of {uniqueTerms.toLocaleString()} unique terms.</p>
        ) : null}
      </section>

      <section>
        <h4 className="text-lg font-semibold text-ink">Ranked terms</h4>
        {terms.length ? (
          <div className="mt-3 max-h-[38rem] overflow-auto rounded-md border border-line bg-surface">
            <table className="w-full min-w-[36rem] border-collapse text-left text-sm">
              <thead className="sticky top-0 z-10 bg-surface">
                <tr className="border-b border-line text-muted">
                  <th className="px-3 py-2 font-medium">#</th>
                  <th className="px-3 py-2 font-medium">Term</th>
                  <th className="px-3 py-2 text-right font-medium">Count</th>
                  <th className="px-3 py-2 text-right font-medium">Frequency</th>
                </tr>
              </thead>
              <tbody>
                {terms.map((term, index) => {
                  const relative = maxFrequency > 0 ? numberValue(term.frequency) / maxFrequency : 0;
                  return (
                    <tr key={`${stringValue(term.term)}-${index}`} className="border-b border-line/70 last:border-0">
                      <td className="px-3 py-2 text-muted">{index + 1}</td>
                      <td className="px-3 py-2">
                        <span className="font-medium text-ink">{stringValue(term.term)}</span>
                        <span className="mt-1 block h-1.5 overflow-hidden rounded-full bg-zinc-100" aria-hidden="true">
                          <span className="block h-full rounded-full bg-accent" style={{ width: `${Math.max(2, relative * 100)}%` }} />
                        </span>
                      </td>
                      <td className="px-3 py-2 text-right text-ink">{formatInteger(term.count)}</td>
                      <td className="px-3 py-2 text-right text-muted">{formatPercent(term.frequency)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : <p className="mt-3 text-sm text-muted">No terms were produced.</p>}
      </section>
    </div>
  );
}

function SemanticCorpusPanel({ report }: { report: JsonRecord | null }) {
  const concepts = asRecordArray(report?.concepts);
  const sources = asRecordArray(report?.sources);
  const semantic = asRecord(report?.semantic);
  const timeline = asRecordArray(semantic?.timeline);
  const embeddingModel = asRecord(semantic?.embeddingModel);
  const itemCount = numberValue(report?.itemCount);
  const nonConceptUnitCount = numberValue(report?.nonConceptUnitCount);
  const maxMembers = concepts.reduce((max, concept) => Math.max(max, numberValue(concept.memberUnitCount)), 0);
  const modelName = stringValue(embeddingModel?.modelName, "unknown embedding backend");
  const dimensions = numberValue(embeddingModel?.dimensions);

  if (itemCount < 2) {
    return (
      <div className="grid gap-6">
        <section>
          <p className="text-xs font-semibold uppercase tracking-[0.12em] text-muted">Corpus view</p>
          <h4 className="mt-1 text-xl font-semibold text-ink">Corpus themes</h4>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-muted">
            Corpus themes require at least two supplied documents. This analysis contains one item, so cross-source theme evidence is intentionally not presented here. Use Semantic map for structure inside the current document, or select/drop multiple files together.
          </p>
          <dl className="mt-4 grid gap-x-8 gap-y-3 text-sm sm:grid-cols-2 lg:grid-cols-4">
            <Fact label="Corpus items" value={formatInteger(report?.itemCount)} />
            <Fact label="Sources" value={String(sources.length)} />
            <Fact label="Semantic units" value={String(timeline.length)} />
            <Fact label="Embedding" value={modelName} />
          </dl>
        </section>
        <p className="rounded-md border border-line bg-surface px-4 py-3 text-sm text-muted">
          The raw one-item corpus operation remains available in Technical for inspection, but the product surface does not label within-document clusters as corpus themes.
        </p>
      </div>
    );
  }

  return (
    <div className="grid gap-8">
      <section>
        <p className="text-xs font-semibold uppercase tracking-[0.12em] text-muted">Corpus view</p>
        <h4 className="mt-1 text-xl font-semibold text-ink">Corpus themes</h4>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-muted">
          Recurring, cohesion-preserving themes across the supplied sources. A cluster must have at least two supporting sentence units before it is promoted as a theme; one-off evidence remains available in Technical instead of becoming a fake concept.
        </p>
        <p className="mt-2 max-w-3xl text-xs leading-5 text-muted">
          Embedding evidence: {modelName}{dimensions > 0 ? ` · ${dimensions} dimensions` : ""}. The built-in hashed TF-IDF backend is a deterministic local baseline, not a learned sentence model.
        </p>
        <dl className="mt-4 grid gap-x-8 gap-y-3 text-sm sm:grid-cols-2 lg:grid-cols-5">
          <Fact label="Corpus items" value={formatInteger(report?.itemCount)} />
          <Fact label="Sources" value={String(sources.length)} />
          <Fact label="Supported themes" value={String(concepts.length)} />
          <Fact label="Low-support units" value={formatInteger(nonConceptUnitCount)} />
          <Fact label="Semantic units" value={String(timeline.length)} />
        </dl>
      </section>

      <section>
        <h4 className="text-lg font-semibold text-ink">Theme evidence</h4>
        {concepts.length ? (
          <div className="mt-3 grid gap-4 lg:grid-cols-2">
            {concepts.map((concept) => {
              const representative = asRecord(concept.representative);
              const relativeSize = maxMembers > 0 ? numberValue(concept.memberUnitCount) / maxMembers : 0;
              const keyTerms = asArray(concept.keyTerms).map((term) => stringValue(term, "")).filter(Boolean);
              return (
                <article key={stringValue(concept.clusterId)} className="rounded-lg border border-line bg-surface p-4">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div>
                      <h5 className="text-sm font-semibold text-ink">{stringValue(concept.label, stringValue(concept.clusterId))}</h5>
                      <span className="mt-1 block text-xs text-accent">{stringValue(concept.clusterId)}</span>
                    </div>
                    <span className="text-right text-xs text-muted">
                      {formatInteger(concept.memberUnitCount)} units · {formatInteger(concept.sourceItemCount)} sources<br />
                      coherence {formatNumber(concept.coherence)}
                    </span>
                  </div>
                  {keyTerms.length ? (
                    <div className="mt-3 flex flex-wrap gap-1.5">
                      {keyTerms.map((term) => <span key={term} className="rounded-full bg-zinc-100 px-2 py-1 text-xs text-muted">{term}</span>)}
                    </div>
                  ) : null}
                  <p className="mt-3 text-sm font-semibold leading-6 text-ink">{stringValue(representative?.text)}</p>
                  <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-zinc-100" aria-hidden="true">
                    <div className="h-full rounded-full bg-accent" style={{ width: `${Math.max(3, relativeSize * 100)}%` }} />
                  </div>
                  <dl className="mt-3 grid gap-2 text-xs text-muted">
                    <div className="flex justify-between gap-3"><dt>Representative source</dt><dd className="text-right">{stringValue(representative?.source, stringValue(representative?.sourceId))}</dd></div>
                    {asArray(concept.authors).length ? <div className="flex justify-between gap-3"><dt>Authors</dt><dd className="text-right">{stringValue(concept.authors)}</dd></div> : null}
                  </dl>
                </article>
              );
            })}
          </div>
        ) : (
          <p className="mt-3 rounded-md border border-line bg-surface px-4 py-3 text-sm text-muted">
            No recurring theme met the minimum support and coherence thresholds. This is preferable to promoting unrelated or one-off sentences as concepts.
          </p>
        )}
      </section>
    </div>
  );
}

function OverviewPanel({ documentReport, core, lexical, enrichedStats, summary, clusters }: {
  documentReport: JsonRecord;
  core: JsonRecord | null;
  lexical: JsonRecord | null;
  enrichedStats: JsonRecord | null;
  summary: JsonRecord[];
  clusters: JsonRecord[];
}) {
  const scriptProfile = asRecord(core?.scriptProfile);
  const readability = asRecord(lexical?.readability);
  const sentiment = asRecord(lexical?.sentiment);
  return (
    <div className="grid gap-8">
      <section>
        <p className="text-xs font-semibold uppercase tracking-[0.12em] text-muted">Document view</p>
        <h4 className="mt-1 text-xl font-semibold text-ink">Extractive summary</h4>
        {summary.length ? (
          <ol className="mt-3 grid gap-3">
            {summary.map((item, index) => (
              <li key={`${stringValue(item.index)}-${index}`} className="border-l-2 border-line pl-4 text-sm leading-6 text-ink">{stringValue(item.text)}</li>
            ))}
          </ol>
        ) : <p className="mt-2 text-sm text-muted">No summary sentences were produced.</p>}
      </section>
      <section>
        <h4 className="text-lg font-semibold text-ink">Document facts</h4>
        <dl className="mt-3 grid gap-x-8 gap-y-3 text-sm sm:grid-cols-2 lg:grid-cols-3">
          <Fact label="Language" value={stringValue(documentReport.language, "undetermined")} />
          <Fact label="Dominant script" value={stringValue(scriptProfile?.dominantScript, "undetermined")} />
          <Fact label="Tokens" value={String(asArray(core?.tokens).length)} />
          <Fact label="Sentences" value={String(asArray(core?.sentences).length)} />
          <Fact label="Paragraphs" value={String(asArray(core?.paragraphs).length)} />
          <Fact label="Lexical density" value={formatNumber(enrichedStats?.lexicalDensity)} />
          <Fact label="Shannon entropy" value={formatNumber(enrichedStats?.shannonEntropy)} />
          <Fact label="Average sentence words" value={formatNumber(readability?.averageSentenceWords)} />
          <Fact label="Average word characters" value={formatNumber(readability?.averageWordChars)} />
        </dl>
      </section>
      <section className="grid gap-6 lg:grid-cols-2">
        <div><h4 className="text-lg font-semibold text-ink">Sentiment evidence</h4><JsonTable value={sentiment} empty="No lexical sentiment evidence." /></div>
        <div>
          <h4 className="text-lg font-semibold text-ink">Leading semantic concepts</h4>
          {clusters.length ? (
            <ul className="mt-3 grid gap-3">{clusters.slice(0, 6).map((cluster) => (
              <li key={stringValue(cluster.id)} className="text-sm leading-6"><span className="font-medium text-ink">{stringValue(cluster.representativeText)}</span><span className="ml-2 text-muted">mean similarity {formatNumber(cluster.meanSimilarity)}</span></li>
            ))}</ul>
          ) : <p className="mt-2 text-sm text-muted">No semantic clusters were produced.</p>}
        </div>
      </section>
    </div>
  );
}

function KeywordsPanel({ keywords, phraseKeywords, topTerms }: { keywords: JsonRecord[]; phraseKeywords: JsonRecord[]; topTerms: JsonRecord[] }) {
  return <div className="grid gap-8 lg:grid-cols-3"><RankedTextList title="Keywords" items={keywords} /><RankedTextList title="Phrase keywords" items={phraseKeywords} /><RankedTextList title="Top terms" items={topTerms} /></div>;
}

function RankedTextList({ title, items }: { title: string; items: JsonRecord[] }) {
  return (
    <section>
      <h4 className="text-lg font-semibold text-ink">{title}</h4>
      {items.length ? <ol className="mt-3 grid gap-2">{items.map((item, index) => (
        <li key={`${stringValue(item.text ?? item.term)}-${index}`} className="flex items-baseline justify-between gap-3 border-b border-line py-2 text-sm">
          <span className="font-medium text-ink">{stringValue(item.text ?? item.term ?? item.terms)}</span>
          <span className="shrink-0 text-muted">{typeof item.score === "number" ? formatNumber(item.score) : typeof item.count === "number" ? `${item.count}×` : ""}</span>
        </li>
      ))}</ol> : <p className="mt-2 text-sm text-muted">No items.</p>}
    </section>
  );
}

function EntitiesPanel({ entities }: { entities: JsonRecord[] }) {
  return (
    <section>
      <h4 className="text-lg font-semibold text-ink">Rule-based entity evidence</h4>
      <p className="mt-1 max-w-3xl text-sm leading-6 text-muted">These are deterministic mentions from the lexical analysis, not claims beyond the source text.</p>
      {entities.length ? (
        <div className="mt-4 overflow-x-auto"><table className="w-full min-w-[32rem] border-collapse text-left text-sm">
          <thead><tr className="border-b border-line text-muted"><th className="py-2 pr-4 font-medium">Mention</th><th className="py-2 pr-4 font-medium">Kind</th><th className="py-2 font-medium">Span</th></tr></thead>
          <tbody>{entities.map((entity, index) => (
            <tr key={`${stringValue(entity.text)}-${index}`} className="border-b border-line/70"><td className="py-3 pr-4 font-medium text-ink">{stringValue(entity.text)}</td><td className="py-3 pr-4 text-muted">{stringValue(entity.kind)}</td><td className="py-3 text-muted"><JsonInline value={entity.span} /></td></tr>
          ))}</tbody>
        </table></div>
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
  if (!linguistic) return <p className="text-sm text-muted">No linguistic section was produced.</p>;
  return <div className="grid gap-3">{sections.map(([key, label]) => (
    <details key={key} className="rounded-md border border-line bg-surface px-4 py-3" open={key === "topics" || key === "style" || key === "outline"}><summary className="cursor-pointer text-sm font-semibold text-ink">{label}</summary><JsonBlock value={linguistic[key]} /></details>
  ))}</div>;
}

function SemanticMapPanel({ clusters, timeline, unitsById, semantic }: { clusters: JsonRecord[]; timeline: JsonRecord[]; unitsById: Map<string, JsonRecord>; semantic: JsonRecord | null }) {
  const maxMembers = clusters.reduce((max, cluster) => Math.max(max, asArray(cluster.memberUnitIds).length), 0);
  return (
    <div className="grid gap-8">
      <section>
        <p className="text-xs font-semibold uppercase tracking-[0.12em] text-muted">Document view</p>
        <h4 className="mt-1 text-xl font-semibold text-ink">Semantic map</h4>
        <p className="mt-1 text-sm leading-6 text-muted">Concepts and trajectory for this document only. Upload multiple files together for cross-source Corpus themes.</p>
        <div className="mt-4 grid gap-4 lg:grid-cols-2">{clusters.map((cluster) => {
          const members = asArray(cluster.memberUnitIds).length;
          const relativeSize = maxMembers > 0 ? members / maxMembers : 0;
          return (
            <article key={stringValue(cluster.id)} className="rounded-md border border-line bg-surface p-4">
              <p className="text-sm font-semibold text-ink">{stringValue(cluster.representativeText)}</p>
              <p className="mt-1 text-xs leading-5 text-muted">{stringValue(cluster.id)} · {members} units · mean similarity {formatNumber(cluster.meanSimilarity)}</p>
              <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-zinc-100" aria-hidden="true"><div className="h-full rounded-full bg-accent" style={{ width: `${Math.max(3, relativeSize * 100)}%` }} /></div>
            </article>
          );
        })}</div>
      </section>
      <section>
        <h4 className="text-lg font-semibold text-ink">Semantic trajectory</h4>
        {timeline.length ? <ol className="mt-4 grid gap-3">{timeline.map((point, index) => {
          const unit = unitsById.get(stringValue(point.unitId));
          return <li key={`${stringValue(point.unitId)}-${index}`} className="grid gap-1 border-l-2 border-line pl-4 sm:grid-cols-[minmax(0,1fr)_auto] sm:gap-4"><span className="text-sm leading-6 text-ink">{stringValue(unit?.text, stringValue(point.unitId))}</span><span className="text-xs text-muted">{stringValue(point.clusterId)} · shift {formatNumber(point.semanticShift)}</span></li>;
        })}</ol> : <p className="mt-3 text-sm text-muted">No semantic timeline was produced.</p>}
      </section>
      <section><h4 className="text-lg font-semibold text-ink">Hotspots and neighborhood evidence</h4><JsonBlock value={{ hotspots: semantic?.hotspots, neighbors: semantic?.neighbors }} /></section>
    </div>
  );
}

function TechnicalPanel({ documentReport, semanticReport, corpusReport }: { documentReport: JsonRecord; semanticReport: JsonRecord | null; corpusReport: JsonRecord | null }) {
  return (
    <div className="grid gap-4">
      <details className="rounded-md border border-line bg-surface px-4 py-3" open><summary className="cursor-pointer text-sm font-semibold text-ink">Document analysis JSON</summary><JsonBlock value={documentReport} /></details>
      <details className="rounded-md border border-line bg-surface px-4 py-3"><summary className="cursor-pointer text-sm font-semibold text-ink">Semantic map JSON</summary><JsonBlock value={semanticReport} /></details>
      <details className="rounded-md border border-line bg-surface px-4 py-3"><summary className="cursor-pointer text-sm font-semibold text-ink">Corpus analysis JSON</summary><JsonBlock value={corpusReport} /></details>
    </div>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return <div className="border-b border-line pb-2"><dt className="text-muted">{label}</dt><dd className="mt-1 font-medium text-ink">{value}</dd></div>;
}

function JsonTable({ value, empty }: { value: JsonRecord | null; empty: string }) {
  if (!value || Object.keys(value).length === 0) return <p className="mt-2 text-sm text-muted">{empty}</p>;
  return <dl className="mt-3 grid gap-2 text-sm">{Object.entries(value).map(([key, entry]) => (
    <div key={key} className="flex items-start justify-between gap-4 border-b border-line py-2"><dt className="text-muted">{humanize(key)}</dt><dd className="max-w-[65%] text-right font-medium text-ink"><JsonInline value={entry} /></dd></div>
  ))}</dl>;
}

function JsonInline({ value }: { value: unknown }) {
  if (value == null) return <>—</>;
  if (typeof value === "number") return <>{formatNumber(value)}</>;
  if (typeof value === "string" || typeof value === "boolean") return <>{String(value)}</>;
  return <>{JSON.stringify(value)}</>;
}

function JsonBlock({ value }: { value: unknown }) {
  return <pre className="mt-3 max-h-[36rem] overflow-auto whitespace-pre-wrap break-words rounded-md bg-zinc-950 p-4 text-xs leading-5 text-zinc-100">{JSON.stringify(value ?? null, null, 2)}</pre>;
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

function numberValue(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function formatNumber(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value)
    ? new Intl.NumberFormat(undefined, { maximumFractionDigits: 3 }).format(value)
    : "—";
}

function formatInteger(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value) ? Math.trunc(value).toLocaleString() : "—";
}

function formatPercent(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value)
    ? new Intl.NumberFormat(undefined, { style: "percent", maximumFractionDigits: 2 }).format(value)
    : "—";
}

function humanize(value: string): string {
  return value.replace(/([a-z])([A-Z])/g, "$1 $2").replace(/^./, (letter) => letter.toUpperCase());
}

function documentId(sourceLabel: string): string {
  return sourceLabel.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 64) || "browser-text";
}
