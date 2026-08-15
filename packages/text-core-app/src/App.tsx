import { useState } from "react";
import {
  PackageSurfaceWorkbench,
  type PackageAppConfig,
  type SurfaceResponse,
} from "@moritzbrantner/nlp-app-ui/package-surface";
import * as wasm from "@moritzbrantner/text-core-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "text-core",
  title: "Text Core",
  description: "Shared text documents, tokenization, spans, and statistics for video-analysis.",
  domain: "text",
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/text-core",
    standaloneRoute: "",
  },
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run the main package workflow.",
      operations: ["text.statistics", "text.normalize", "text.tokenize", "text.boundaries"],
    },
    {
      id: "debug",
      label: "Debug",
      description: "Inspect inputs, plans, metadata, and diagnostic helpers.",
      operations: ["describe"],
    },
  ],
  defaultOperation: "text.tokenize",
  defaultPresetId: "tokenize-transcript-notes",
  featuredOperations: ["text.tokenize", "text.statistics", "text.normalize", "text.boundaries", "describe"],
  presets: [
    {
      id: "tokenize-transcript-notes",
      label: "Tokenize transcript notes",
      operation: "text.tokenize",
      description: "Return tokens, spans, normalized forms, and token statistics.",
      input: { text: "Rust analyzes transcripts, captions, and scene notes.", includeStats: true },
    },
    {
      id: "statistics-scene-summary",
      label: "Count document statistics",
      operation: "text.statistics",
      description: "Count bytes, characters, words, lines, and sentence boundaries.",
      input: { text: "Scene one opens in Berlin.\nRust text packages summarize transcript evidence." },
    },
    {
      id: "normalize-noisy-caption",
      label: "Normalize noisy caption",
      operation: "text.normalize",
      description: "Trim, collapse whitespace, and lowercase a caption-like string.",
      input: { text: "  RUST   Text   WORKFLOWS  ", lowercase: true, collapseWhitespace: true, trim: true },
    },
    {
      id: "boundaries-review-copy",
      label: "Find word and sentence boundaries",
      operation: "text.boundaries",
      description: "Split a short review note into word, sentence, paragraph, and grapheme spans.",
      input: { text: "Alice reviewed the tokenizer roadmap. Bob checked transcript retrieval.\n\nThe release note stayed concise." },
    },
  ],
  benchmarkScenarios: [
    {
      id: "tokenize",
      label: "Tokenize",
      operation: "text.tokenize",
      input: { text: "Rust analyzes transcripts, captions, and scene notes.".repeat(24), includeStats: true },
      iterations: 80,
      warmupIterations: 5,
      outputCountPath: ["tokens"],
    },
    {
      id: "boundaries",
      label: "Boundaries",
      operation: "text.boundaries",
      input: { text: "One sentence. Another sentence for browser benchmarks.\n\nA new paragraph follows." },
      iterations: 120,
      warmupIterations: 5,
      outputCountPath: ["sentences"],
    },
    {
      id: "statistics",
      label: "Statistics",
      operation: "text.statistics",
      input: { text: "Statistics over repeated transcript text. ".repeat(64) },
      iterations: 100,
      warmupIterations: 5,
      outputCountPath: ["value"],
    },
  ],
  resultTabs: [
    {
      id: "summary",
      label: "Summary",
      render: (response) => <TextCoreSummary response={response} />,
    },
  ],
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}

function TextCoreSummary({ response }: { response: SurfaceResponse | null }) {
  if (!response) {
    return (
      <div className="mt-4 rounded-md border border-dashed border-zinc-300 bg-zinc-50 px-4 py-6 text-sm text-zinc-600">
        Run a text operation to see the formatted result.
      </div>
    );
  }

  const value = asRecord(response.value);
  const title = stringValue(value.title) ?? titleForOperation(response.operation);
  const message = stringValue(value.message) ?? messageForOperation(response.operation);

  if (response.operation === "text.tokenize") {
    return (
      <SummaryFrame title={title} message={message}>
        <TokenSummary value={value} />
      </SummaryFrame>
    );
  }

  if (response.operation === "text.statistics") {
    return (
      <SummaryFrame title={title} message={message}>
        <StatisticsSummary value={value} />
      </SummaryFrame>
    );
  }

  if (response.operation === "text.normalize") {
    return (
      <SummaryFrame title={title} message={message}>
        <NormalizeSummary value={value} />
      </SummaryFrame>
    );
  }

  if (response.operation === "text.boundaries") {
    return (
      <SummaryFrame title={title} message={message}>
        <BoundariesSummary value={value} />
      </SummaryFrame>
    );
  }

  return (
    <SummaryFrame title={title} message={message}>
      <KeyValuePanel title="Response" value={value} />
    </SummaryFrame>
  );
}

function SummaryFrame({
  title,
  message,
  children,
}: {
  title: string;
  message?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="mt-4 space-y-5">
      <div className="rounded-md border border-zinc-200 bg-zinc-50 p-4">
        <p className="text-xs font-semibold uppercase text-zinc-500">Text Core Result</p>
        <h2 className="mt-1 text-lg font-semibold text-zinc-950">{title}</h2>
        {message ? <p className="mt-2 max-w-3xl text-sm leading-6 text-zinc-600">{message}</p> : null}
      </div>
      {children}
    </div>
  );
}

function TokenSummary({ value }: { value: Record<string, unknown> }) {
  const tokens = arrayValue(value.tokens).map(asRecord);
  const stats = asRecord(value.stats);
  const scriptProfile = asRecord(value.scriptProfile);

  return (
    <div className="space-y-5">
      <StatGrid
        stats={{
          Tokens: tokens.length,
          "Unique Tokens": numberValue(stats.unique_tokens),
          "Dominant Script": stringValue(scriptProfile.dominant_script) ?? stringValue(scriptProfile.dominantScript),
          "Mixed Script": boolLabel(booleanValue(scriptProfile.is_mixed) ?? booleanValue(scriptProfile.isMixed)),
        }}
      />
      <CollapsiblePanel title="Tokens" count={tokens.length}>
        {tokens.length > 0 ? (
          <ol className="divide-y divide-zinc-100">
            {tokens.map((token, index) => {
              const span = asRecord(token.span);
              return (
                <li key={`${stringValue(token.text) ?? "token"}-${index}`} className="grid gap-2 px-4 py-3 sm:grid-cols-[minmax(0,1fr)_9rem_10rem]">
                  <div className="min-w-0">
                    <p className="break-words text-sm font-medium text-zinc-950">{stringValue(token.text) ?? "(empty)"}</p>
                    <p className="mt-1 break-words text-xs text-zinc-500">Normalized: {stringValue(token.normalized) ?? "n/a"}</p>
                  </div>
                  <p className="text-sm text-zinc-700">{stringValue(token.kind) ?? "Token"}</p>
                  <p className="font-mono text-xs text-zinc-500">
                    chars {formatSpanNumber(span.char_start, span.charStart)}-{formatSpanNumber(span.char_end, span.charEnd)}
                  </p>
                </li>
              );
            })}
          </ol>
        ) : (
          <p className="px-4 py-5 text-sm text-zinc-600">No tokens returned.</p>
        )}
      </CollapsiblePanel>
      <KeyValuePanel title="Text Statistics" value={stats} />
    </div>
  );
}

function StatisticsSummary({ value }: { value: Record<string, unknown> }) {
  const stats = asRecord(value.value);
  return (
    <div className="space-y-5">
      <StatGrid
        stats={{
          Bytes: numberValue(stats.byteCount),
          Characters: numberValue(stats.characterCount),
          Words: numberValue(stats.wordCount),
          Lines: numberValue(stats.lineCount),
          Sentences: numberValue(stats.sentenceCount),
        }}
      />
      <KeyValuePanel title="Statistics Value" value={stats} />
    </div>
  );
}

function NormalizeSummary({ value }: { value: Record<string, unknown> }) {
  const normalized = stringValue(value.text) ?? "";
  const before = asRecord(value.before);
  const after = asRecord(value.after);

  return (
    <div className="space-y-5">
      <section className="rounded-md border border-emerald-200 bg-emerald-50 p-4">
        <p className="text-xs font-semibold uppercase text-emerald-700">Normalized Text</p>
        <p className="mt-2 whitespace-pre-wrap break-words text-base font-medium text-zinc-950">{normalized || "(empty)"}</p>
      </section>
      <div className="grid gap-4 xl:grid-cols-2">
        <TextStatsPanel title="Before" value={before} />
        <TextStatsPanel title="After" value={after} />
      </div>
    </div>
  );
}

function BoundariesSummary({ value }: { value: Record<string, unknown> }) {
  const words = arrayValue(value.words);
  const sentences = arrayValue(value.sentences);
  const paragraphs = arrayValue(value.paragraphs);
  const graphemes = arrayValue(value.graphemes);

  return (
    <div className="space-y-5">
      <StatGrid
        stats={{
          Words: words.length,
          Sentences: sentences.length,
          Paragraphs: paragraphs.length,
          Graphemes: graphemes.length,
        }}
      />
      <ListPanel title="Words" values={words} />
      <ListPanel title="Sentences" values={sentences} />
      <ListPanel title="Paragraphs" values={paragraphs} />
    </div>
  );
}

function TextStatsPanel({ title, value }: { title: string; value: Record<string, unknown> }) {
  return (
    <section className="rounded-md border border-zinc-200 bg-white p-4">
      <h3 className="text-sm font-semibold text-zinc-950">{title}</h3>
      <KeyValuePanel title="Basic" value={asRecord(value.basic)} flush />
      <StatGrid
        stats={{
          Paragraphs: numberValue(value.paragraphs),
          Tokens: numberValue(value.tokens),
          "Unique Tokens": numberValue(value.unique_tokens),
          "Words / Sentence": numberValue(value.average_words_per_sentence),
          "Chars / Word": numberValue(value.average_chars_per_word),
        }}
        compact
      />
    </section>
  );
}

function KeyValuePanel({
  title,
  value,
  flush = false,
}: {
  title: string;
  value: Record<string, unknown>;
  flush?: boolean;
}) {
  const entries = Object.entries(value).filter(([, entryValue]) => isScalar(entryValue));
  if (entries.length === 0) {
    return null;
  }

  return (
    <section className={flush ? "mt-4" : "rounded-md border border-zinc-200 bg-white"}>
      <div className={flush ? "mb-2" : "border-b border-zinc-200 px-4 py-3"}>
        <h3 className="text-sm font-semibold text-zinc-950">{title}</h3>
      </div>
      <dl className={flush ? "grid gap-2" : "divide-y divide-zinc-100"}>
        {entries.map(([key, entryValue]) => (
          <div key={key} className={flush ? "flex justify-between gap-4 text-sm" : "grid gap-1 px-4 py-3 sm:grid-cols-[12rem_minmax(0,1fr)]"}>
            <dt className="font-medium text-zinc-600">{humanizeKey(key)}</dt>
            <dd className="break-words text-zinc-950">{formatValue(entryValue)}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

function ListPanel({ title, values }: { title: string; values: unknown[] }) {
  return (
    <CollapsiblePanel title={title} count={values.length}>
      {values.length > 0 ? (
        <ul className="divide-y divide-zinc-100">
          {values.slice(0, 24).map((value, index) => (
            <li key={index} className="px-4 py-3 text-sm text-zinc-800">
              {formatBoundary(value)}
            </li>
          ))}
        </ul>
      ) : (
        <p className="px-4 py-5 text-sm text-zinc-600">No {title.toLowerCase()} returned.</p>
      )}
    </CollapsiblePanel>
  );
}

function CollapsiblePanel({
  title,
  count,
  children,
}: {
  title: string;
  count: number;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(true);
  const contentId = `text-core-list-${title.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`;

  return (
    <section className="rounded-md border border-zinc-200 bg-white">
      <button
        aria-controls={contentId}
        aria-expanded={open}
        className="flex w-full items-center justify-between gap-3 border-b border-zinc-200 px-4 py-3 text-left"
        type="button"
        onClick={() => setOpen((current) => !current)}
      >
        <span className="min-w-0">
          <span className="block text-sm font-semibold text-zinc-950">{title}</span>
          <span className="mt-0.5 block text-xs text-zinc-500">
            {count} {count === 1 ? "item" : "items"}
          </span>
        </span>
        <span className="shrink-0 rounded-md border border-zinc-300 px-2 py-1 text-xs font-medium text-zinc-700">
          {open ? "Collapse" : "Expand"}
        </span>
      </button>
      {open ? <div id={contentId}>{children}</div> : null}
    </section>
  );
}

function StatGrid({
  stats,
  compact = false,
}: {
  stats: Record<string, unknown>;
  compact?: boolean;
}) {
  return (
    <div className={compact ? "mt-4 grid gap-3 sm:grid-cols-2" : "grid gap-3 sm:grid-cols-2 xl:grid-cols-4"}>
      {Object.entries(stats).map(([label, value]) => (
        <div key={label} className="rounded-md border border-zinc-200 bg-white p-3">
          <p className="text-xs font-medium uppercase text-zinc-500">{label}</p>
          <p className="mt-1 break-words text-xl font-semibold text-zinc-950">{formatValue(value)}</p>
        </div>
      ))}
    </div>
  );
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

function arrayValue(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function numberValue(value: unknown): number | undefined {
  return typeof value === "number" ? value : undefined;
}

function booleanValue(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}

function boolLabel(value: boolean | undefined): string {
  if (value == null) {
    return "n/a";
  }
  return value ? "Yes" : "No";
}

function isScalar(value: unknown): boolean {
  return value == null || ["string", "number", "boolean"].includes(typeof value);
}

function formatValue(value: unknown): string {
  if (value == null) {
    return "n/a";
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(2);
  }
  if (typeof value === "boolean") {
    return value ? "Yes" : "No";
  }
  return String(value);
}

function formatBoundary(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  const record = asRecord(value);
  return (
    stringValue(record.text) ??
    stringValue(record.normalized) ??
    Object.entries(record)
      .filter(([, entryValue]) => isScalar(entryValue))
      .map(([key, entryValue]) => `${humanizeKey(key)}: ${formatValue(entryValue)}`)
      .join(", ")
  );
}

function formatSpanNumber(snakeCase: unknown, camelCase: unknown): string {
  return formatValue(snakeCase ?? camelCase);
}

function humanizeKey(key: string): string {
  return key
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function titleForOperation(operation: string): string {
  switch (operation) {
    case "text.tokenize":
      return "Tokenized Text";
    case "text.statistics":
      return "Text Statistics";
    case "text.normalize":
      return "Normalized Text";
    case "text.boundaries":
      return "Text Boundaries";
    case "describe":
      return "Package Surface Metadata";
    default:
      return humanizeKey(operation);
  }
}

function messageForOperation(operation: string): string | undefined {
  switch (operation) {
    case "text.tokenize":
      return "Tokenized the supplied text with token spans, normalized values, script profile, and text statistics.";
    case "text.statistics":
      return "Counted bytes, characters, words, lines, and sentences.";
    case "text.normalize":
      return "Normalized the supplied text and compared before and after statistics.";
    case "text.boundaries":
      return "Extracted word, sentence, paragraph, and grapheme boundaries.";
    default:
      return undefined;
  }
}
