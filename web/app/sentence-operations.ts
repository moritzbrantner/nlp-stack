// Per-sentence operation rows for the Sentences view: the Rust routing record
// (`linguistic.sentences`) plus the browser NER model's per-sentence report.

type JsonRecord = Record<string, unknown>;

export type OperationStatus = "supported" | "partial" | "unsupported" | "unknown" | "rust-only";

export interface OperationRow {
  operation: string;
  label: string;
  status: OperationStatus;
  producer: string | null;
  detail: string;
}

export interface SentenceView {
  index: number;
  text: string;
  byteStart: number;
  byteEnd: number;
  language: string | null;
  detection: string;
  rows: OperationRow[];
  reporting: ReportingView;
  scope: ScopeView;
  syntax: SyntaxView | null;
}

/** One syntactic word from a syntax backend (`text_linguistics::SyntaxWord`). */
export interface SyntaxWordView {
  index: number;
  form: string;
  /** The surface token's text when it expands to several words (`zum` for `zu`). */
  token: string | null;
  lemma: string | null;
  upos: string | null;
  feats: string;
  /** The head word's form, or `null` for the root. */
  head: string | null;
  relation: string;
}

/** Word-level syntax of one sentence with the backend and model versions that produced it. */
export interface SyntaxView {
  producer: string;
  models: string;
  words: SyntaxWordView[];
}

/** A reporting construction an operator refers to, by its source and cue. */
export interface ReportingRef {
  id: number;
  label: string;
}

/** One negation/uncertainty/hypothetical operator (`text_linguistics::ScopeOperator`) with its nested children. */
export interface ScopeNode {
  id: number;
  kind: "negation" | "uncertainty" | "hypothetical";
  resolved: boolean;
  unresolvedReason: string | null;
  cue: PassageView;
  cueRelation: string;
  head: PassageView | null;
  scope: PassageView | null;
  consequent: PassageView | null;
  reportingTarget: ReportingRef | null;
  reportedWithin: ReportingRef | null;
  confidence: number | null;
  children: ScopeNode[];
}

/** The sentence's scope evidence: support status plus its outermost operators. */
export interface ScopeView {
  status: OperationStatus;
  producer: string | null;
  detail: string;
  roots: ScopeNode[];
}

/** A passage of the source text with its UTF-8 half-open byte span. */
export interface PassageView {
  text: string;
  byteStart: number;
  byteEnd: number;
}

/** One reporting/belief construction (`text_linguistics::ReportingConstruction`) with its nested children. */
export interface ReportingNode {
  id: number;
  kind: "speech" | "belief";
  form: "indirect" | "direct";
  source: PassageView | null;
  cue: PassageView;
  cueLemma: string;
  content: PassageView;
  confidence: number;
  children: ReportingNode[];
}

/** The sentence's reporting evidence: support status plus its outermost constructions. */
export interface ReportingView {
  status: OperationStatus;
  producer: string | null;
  detail: string;
  roots: ReportingNode[];
}

const operationLabels: [string, string][] = [
  ["languageDetection", "Language detection"],
  ["sentenceSegmentation", "Sentence segmentation"],
  ["tokenization", "Tokenization"],
  ["namedEntities", "Named entities"],
  ["morphology", "Lemma and morphology"],
  ["partOfSpeech", "Part of speech"],
  ["dependencySyntax", "Dependency syntax"],
];

const languageNames: Record<string, string> = {
  en: "English",
  de: "German",
  es: "Spanish",
  fr: "French",
  it: "Italian",
  pt: "Portuguese",
  nl: "Dutch",
};

/**
 * Languages whose lemma/POS/dependency models exist only in the native Rust library.
 * The browser runs no heuristic stand-in for them (#186), so these layers show the
 * model they need instead.
 */
const rustOnlyModelLanguages = new Set(["en", "de", "es"]);

export function languageName(language: string | null): string {
  if (!language) return "Undecided";
  return `${languageNames[language] ?? language} (${language})`;
}

function record(value: unknown): JsonRecord | null {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as JsonRecord) : null;
}

function records(value: unknown): JsonRecord[] {
  return Array.isArray(value) ? value.map(record).filter((item): item is JsonRecord => item !== null) : [];
}

function text(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function detectionSummary(detection: JsonRecord | null, pinned: boolean): string {
  if (pinned) return "language given by the caller";
  const candidates = records(detection?.candidates)
    .map((candidate) => `${text(candidate.language) ?? "?"} ${Number(candidate.confidence ?? 0).toFixed(2)}`)
    .join(", ");
  switch (text(detection?.decision)) {
    case "detected":
      return `detected (${candidates})`;
    case "ambiguous":
      return `ambiguous, no language leads clearly (${candidates})`;
    case "tooShort":
      return "too few words to decide a language";
    case "unsupportedScript":
      return text(records(detection?.candidates)[0]?.reason) ?? "script not covered by the detector";
    default:
      return "no detection record";
  }
}

function browserNerRow(
  label: string,
  language: string | null,
  report: JsonRecord | null,
  model: JsonRecord | null,
): OperationRow {
  const modelName = `${text(model?.name) ?? "browser NER model"}@${(text(model?.revision) ?? "").slice(0, 8)}`;
  const supported = Array.isArray(model?.supportedLanguages) ? (model.supportedLanguages as string[]).join(", ") : "en, de, es";
  switch (text(report?.status)) {
    case "completed": {
      const count = Number(report?.entityCount ?? 0);
      const truncated = report?.truncated ? "; the model input was truncated, trailing text is unlabelled" : "";
      return { operation: "namedEntities", label, status: "supported", producer: `${modelName} (browser)`, detail: `${count} ${count === 1 ? "entity" : "entities"}${truncated}` };
    }
    case "unsupported-language":
      return { operation: "namedEntities", label, status: "unsupported", producer: null, detail: `the browser model supports ${supported}, not ${language ?? "this language"}` };
    default:
      return { operation: "namedEntities", label, status: "unknown", producer: null, detail: "sentence language undecided, so the model was not run" };
  }
}

function rustRow(label: string, operation: string, language: string | null, report: JsonRecord | null, tokenCount: number): OperationRow {
  const support = text(report?.support) ?? "unknown";
  const producer = text(report?.producer);
  const reason = text(report?.reason) ?? "";
  const linguistic = operation === "morphology" || operation === "partOfSpeech" || operation === "dependencySyntax";
  if (linguistic && support === "unsupported" && !language) {
    return { operation, label, status: "unknown", producer: null, detail: "sentence language undecided, so no model was chosen" };
  }
  if (linguistic && support === "unsupported" && language && rustOnlyModelLanguages.has(language)) {
    return {
      operation,
      label,
      status: "rust-only",
      producer: null,
      detail: `not available in the browser: the ${languageNames[language]} UD model (COMBO on XLM-RoBERTa base, about 1.1 GB) runs only in the native Rust library; connect a local Stanza sidecar to see this layer here`,
    };
  }
  if (linguistic && support === "unsupported" && language) {
    return { operation, label, status: "unsupported", producer: null, detail: `no model for \`${language}\`: UD models exist for en, de and es only` };
  }
  const detail = operation === "tokenization" ? `${tokenCount} ${tokenCount === 1 ? "token" : "tokens"}; ${reason}` : reason;
  const status = (["supported", "partial", "unsupported", "unknown"] as const).find((value) => value === support) ?? "unknown";
  return { operation, label, status, producer, detail };
}

function passage(value: unknown): PassageView | null {
  const item = record(value);
  const span = record(item?.span);
  if (!item || !span) return null;
  return { text: text(item.text) ?? "", byteStart: Number(span.byte_start ?? 0), byteEnd: Number(span.byte_end ?? 0) };
}

function reportingNode(item: JsonRecord, byId: Map<number, JsonRecord>, depth = 0): ReportingNode {
  const children = depth > 16
    ? []
    : (Array.isArray(item.children) ? item.children : [])
      .map((id) => byId.get(Number(id)))
      .filter((child): child is JsonRecord => child !== undefined)
      .map((child) => reportingNode(child, byId, depth + 1));
  return {
    id: Number(item.id),
    kind: item.kind === "belief" ? "belief" : "speech",
    form: item.form === "direct" ? "direct" : "indirect",
    source: passage(item.source),
    cue: passage(item.cue) ?? { text: "", byteStart: 0, byteEnd: 0 },
    cueLemma: text(item.cueLemma) ?? "",
    content: passage(item.content) ?? { text: "", byteStart: 0, byteEnd: 0 },
    confidence: Number(item.confidence ?? 0),
    children,
  };
}

/** The Rust reporting evidence (`linguistic.reporting`) for one linguistic sentence. */
function reportingView(reporting: JsonRecord | null, sentenceIndex: number | null): ReportingView {
  const report = records(reporting?.sentences).find((item) => Number(item.sentenceIndex) === sentenceIndex) ?? null;
  if (!report) {
    return { status: "unknown", producer: null, detail: "no reporting evidence for this sentence", roots: [] };
  }
  const constructions = records(reporting?.constructions).filter((item) => Number(item.sentenceIndex) === sentenceIndex);
  const byId = new Map(constructions.map((item) => [Number(item.id), item]));
  const roots = constructions
    .filter((item) => item.parent === null || item.parent === undefined)
    .map((item) => reportingNode(item, byId));
  return { status: supportStatus(report.support), producer: text(report.producer), detail: text(report.reason) ?? "", roots };
}

function supportStatus(value: unknown): OperationStatus {
  const support = text(value) ?? "unknown";
  return (["supported", "partial", "unsupported", "unknown"] as const).find((item) => item === support) ?? "unknown";
}

function reportingRef(id: unknown, reporting: JsonRecord | null): ReportingRef | null {
  if (id === null || id === undefined) return null;
  const construction = records(reporting?.constructions).find((item) => Number(item.id) === Number(id));
  if (!construction) return null;
  const source = passage(construction.source)?.text ?? "no stated source";
  return { id: Number(id), label: `${source} — ${passage(construction.cue)?.text ?? ""}` };
}

function scopeNode(item: JsonRecord, byId: Map<number, JsonRecord>, reporting: JsonRecord | null, depth = 0): ScopeNode {
  const children = depth > 16
    ? []
    : (Array.isArray(item.children) ? item.children : [])
      .map((id) => byId.get(Number(id)))
      .filter((child): child is JsonRecord => child !== undefined)
      .map((child) => scopeNode(child, byId, reporting, depth + 1));
  const kind = item.kind === "uncertainty" || item.kind === "hypothetical" ? item.kind : "negation";
  return {
    id: Number(item.id),
    kind,
    resolved: item.resolution === "resolved",
    unresolvedReason: text(item.unresolvedReason),
    cue: passage(item.cue) ?? { text: "", byteStart: 0, byteEnd: 0 },
    cueRelation: text(item.cueRelation) ?? "",
    head: passage(item.head),
    scope: passage(item.scope),
    consequent: passage(item.consequent),
    reportingTarget: reportingRef(item.reportingTarget, reporting),
    reportedWithin: reportingRef(item.reportedWithin, reporting),
    confidence: typeof item.confidence === "number" ? item.confidence : null,
    children,
  };
}

/** The Rust scope evidence (`linguistic.scope`) for one linguistic sentence. */
function scopeView(scope: JsonRecord | null, reporting: JsonRecord | null, sentenceIndex: number | null): ScopeView {
  const report = records(scope?.sentences).find((item) => Number(item.sentenceIndex) === sentenceIndex) ?? null;
  if (!report) {
    return { status: "unknown", producer: null, detail: "no scope evidence for this sentence", roots: [] };
  }
  const operators = records(scope?.operators).filter((item) => Number(item.sentenceIndex) === sentenceIndex);
  const byId = new Map(operators.map((item) => [Number(item.id), item]));
  const roots = operators
    .filter((item) => item.parent === null || item.parent === undefined)
    .map((item) => scopeNode(item, byId, reporting));
  return { status: supportStatus(report.support), producer: text(report.producer), detail: text(report.reason) ?? "", roots };
}

/** One view per core sentence, in source order. */
function syntaxView(sentence: JsonRecord | null): SyntaxView | null {
  if (!sentence) return null;
  const provenance = record(sentence.provenance);
  const tokens = records(sentence.tokens);
  const words = records(sentence.words);
  const models = records(provenance?.models);
  const depparse = models.find((model) => model.stage === "depparse") ?? models[0] ?? null;
  const language = text(sentence.language) ?? "?";
  const producer = `${text(provenance?.backend) ?? "backend"} ${text(provenance?.backendVersion) ?? "?"}`;
  const modelLabel = depparse
    ? `${text(depparse.source) ?? "?"}@${(text(depparse.revision) ?? "").slice(0, 8)} · ${language} ${models
        .map((model) => `${text(model.stage)}=${text(model.package)}`)
        .join(", ")}`
    : "no model versions reported";
  return {
    producer,
    models: modelLabel,
    words: words.map((word, index) => {
      const token = tokens[Number(word.token)] ?? null;
      const feats = Object.entries(record(word.feats) ?? {})
        .map(([name, value]) => `${name}=${String(value)}`)
        .join("|");
      const head = typeof word.head === "number" ? words[word.head] ?? null : null;
      return {
        index,
        form: text(word.form) ?? "",
        token: token && Number(token.wordCount) > 1 ? text(token.text) : null,
        lemma: text(word.lemma),
        upos: text(word.upos),
        feats: feats || "_",
        head: head ? text(head.form) : null,
        relation: [text(word.relation) ?? "dep", text(word.subtype)].filter(Boolean).join(":"),
      };
    }),
  };
}

export function sentenceViews(documentReport: JsonRecord | null): SentenceView[] {
  const core = record(documentReport?.core);
  const routing = records(record(documentReport?.linguistic)?.sentences);
  const reporting = record(record(documentReport?.linguistic)?.reporting);
  const scope = record(record(documentReport?.linguistic)?.scope);
  const syntax = records(record(documentReport?.linguistic)?.syntax);
  const model = record(documentReport?.namedEntityModel);
  const nerReports = records(model?.sentences);
  return records(core?.sentences).map((sentence, index) => {
    const span = record(sentence.span);
    const byteStart = Number(span?.byte_start ?? 0);
    const byteEnd = Number(span?.byte_end ?? 0);
    const routed = routing.find((item) => Number(record(item.span)?.byte_start) === byteStart) ?? null;
    const language = text(routed?.language);
    const operations = records(routed?.operations);
    const ner = nerReports.find((item) => Number(item.sentenceIndex) === index) ?? null;
    const rows = operationLabels.map(([operation, label]) => {
      if (operation === "namedEntities") return browserNerRow(label, language, ner, model);
      const report = operations.find((item) => item.operation === operation) ?? null;
      return rustRow(label, operation, language, report, Number(sentence.token_count ?? 0));
    });
    return {
      index,
      text: text(sentence.text) ?? "",
      byteStart,
      byteEnd,
      language,
      detection: detectionSummary(record(routed?.detection), routed?.languagePinned === true),
      rows,
      reporting: reportingView(reporting, routed ? Number(routed.sentenceIndex) : null),
      scope: scopeView(scope, reporting, routed ? Number(routed.sentenceIndex) : null),
      syntax: syntaxView(
        routed ? syntax.find((item) => Number(item.sentenceIndex) === Number(routed.sentenceIndex)) ?? null : null,
      ),
    };
  });
}
