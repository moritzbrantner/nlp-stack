import { Badge, EmptyState } from "../shared/primitives";
import { SemanticMapPanel } from "./SemanticMapPanels";
import type { ResultTabDefinition, SurfaceResponse } from "./types";

interface TermFrequency {
  term: string;
  count: number;
  frequency: number;
}

interface AuthorProfile {
  author: string;
  itemCount: number;
  semanticUnitCount: number;
  wordCount: number;
  uniqueTerms: number;
  conceptCount: number;
}

interface ConceptEvidence {
  clusterId: string;
  memberUnitCount: number;
  sourceItemCount: number;
  authorCount: number;
  representativeText: string;
  representativeAuthor?: string;
  representativeSource?: string;
  representativeSourceId: string;
}

export function createSemanticCorpusResultTabs(
  operation = "analysis.semantic-corpus",
): ResultTabDefinition[] {
  return [
    {
      id: "semantic-corpus",
      label: "Corpus",
      render: (response) => <SemanticCorpusPanel operation={operation} response={response} />,
    },
    {
      id: "semantic-corpus-map",
      label: "Map",
      render: (response) => <SemanticMapPanel operation={operation} response={response} />,
    },
  ];
}

export function SemanticCorpusPanel({
  operation = "analysis.semantic-corpus",
  response,
}: {
  operation?: string;
  response: SurfaceResponse | null;
}) {
  if (!response || response.operation !== operation) {
    return (
      <div className="mt-4">
        <EmptyState>Run semantic corpus analysis to inspect vocabulary, authors, and concept evidence.</EmptyState>
      </div>
    );
  }

  const report = corpusPayload(response.value);
  const lexical = asRecord(report.lexical);
  const terms = termFrequencies(lexical.topTerms);
  const authors = authorProfiles(report.authors);
  const concepts = conceptEvidence(report.concepts);
  const itemCount = numberValue(report.itemCount);
  const authorCount = numberValue(report.authorCount);
  const wordCount = numberValue(lexical.wordCount);
  const uniqueTerms = numberValue(lexical.uniqueTerms);

  return (
    <div className="mt-4 space-y-5">
      <section className="rounded-md border border-zinc-200 bg-zinc-50 p-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-teal-700">Semantic corpus</p>
            <h2 className="mt-1 text-lg font-semibold text-zinc-950">Language and meaning across sources</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-zinc-600">
              Aggregate lexical usage and deterministic concept structure across attributed texts while retaining the passages and sources behind every concept.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Badge>{itemCount} items</Badge>
            <Badge>{authorCount} authors</Badge>
            <Badge>{wordCount} words</Badge>
            <Badge>{concepts.length} concepts</Badge>
          </div>
        </div>
      </section>

      <section className="overflow-hidden rounded-md border border-zinc-200 bg-white">
        <SectionHeader
          title="Vocabulary profile"
          detail={`${uniqueTerms} unique normalized terms across ${wordCount} word occurrences.`}
        />
        {terms.length === 0 ? (
          <p className="px-4 py-5 text-sm text-zinc-500">No term-frequency evidence was returned.</p>
        ) : (
          <div className="divide-y divide-zinc-100">
            {terms.slice(0, 16).map((term) => (
              <div className="grid gap-3 px-4 py-3 sm:grid-cols-[minmax(8rem,1fr)_7rem_minmax(10rem,2fr)]" key={term.term}>
                <span className="truncate text-sm font-medium text-zinc-900">{term.term}</span>
                <span className="text-sm tabular-nums text-zinc-600">{term.count} uses</span>
                <div className="flex items-center gap-3">
                  <div className="h-2 flex-1 overflow-hidden rounded-full bg-zinc-100">
                    <div
                      className="h-full rounded-full bg-teal-600"
                      style={{ width: `${Math.max(2, Math.min(100, term.frequency * 100))}%` }}
                    />
                  </div>
                  <span className="w-14 text-right text-xs tabular-nums text-zinc-500">
                    {(term.frequency * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="overflow-hidden rounded-md border border-zinc-200 bg-white">
        <SectionHeader
          title="Author profiles"
          detail="Compare how much source material and semantic evidence each attributed author contributes."
        />
        {authors.length === 0 ? (
          <p className="px-4 py-5 text-sm text-zinc-500">No attributed authors were supplied.</p>
        ) : (
          <div className="divide-y divide-zinc-100">
            {authors.map((author) => (
              <div className="grid gap-3 px-4 py-3 md:grid-cols-[minmax(10rem,1.4fr)_repeat(4,minmax(6rem,1fr))]" key={author.author}>
                <span className="truncate text-sm font-semibold text-zinc-950">{author.author}</span>
                <Metric label="items" value={author.itemCount} />
                <Metric label="semantic units" value={author.semanticUnitCount} />
                <Metric label="words" value={author.wordCount} />
                <Metric label="concepts" value={author.conceptCount} />
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="overflow-hidden rounded-md border border-zinc-200 bg-white">
        <SectionHeader
          title="Concept evidence"
          detail="Each corpus concept keeps its deterministic representative passage and source provenance."
        />
        {concepts.length === 0 ? (
          <p className="px-4 py-5 text-sm text-zinc-500">No concept clusters were returned.</p>
        ) : (
          <div className="divide-y divide-zinc-100">
            {concepts.slice(0, 12).map((concept) => (
              <article className="grid gap-3 px-4 py-4 lg:grid-cols-[10rem_minmax(0,1fr)_15rem]" key={concept.clusterId}>
                <div>
                  <p className="text-sm font-semibold text-zinc-950">{concept.clusterId}</p>
                  <p className="mt-1 text-xs text-zinc-500">
                    {concept.memberUnitCount} units · {concept.sourceItemCount} sources · {concept.authorCount} authors
                  </p>
                </div>
                <blockquote className="text-sm leading-6 text-zinc-700">{concept.representativeText}</blockquote>
                <div className="min-w-0 text-xs leading-5 text-zinc-500">
                  <p className="truncate" title={concept.representativeSource ?? concept.representativeSourceId}>
                    {concept.representativeSource ?? concept.representativeSourceId}
                  </p>
                  {concept.representativeAuthor ? <p>{concept.representativeAuthor}</p> : null}
                </div>
              </article>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

function SectionHeader({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="border-b border-zinc-200 px-4 py-3">
      <h3 className="text-sm font-semibold text-zinc-950">{title}</h3>
      <p className="mt-1 text-xs leading-5 text-zinc-500">{detail}</p>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <p className="text-sm font-medium tabular-nums text-zinc-900">{value}</p>
      <p className="text-xs text-zinc-500">{label}</p>
    </div>
  );
}

function corpusPayload(value: unknown): Record<string, unknown> {
  const root = asRecord(value);
  const result = asRecord(root.result);
  return Object.keys(result).length > 0 ? result : root;
}

function termFrequencies(value: unknown): TermFrequency[] {
  return asArray(value).flatMap((entry) => {
    const record = asRecord(entry);
    const term = stringValue(record.term);
    if (!term) return [];
    return [
      {
        term,
        count: numberValue(record.count),
        frequency: numberValue(record.frequency),
      },
    ];
  });
}

function authorProfiles(value: unknown): AuthorProfile[] {
  return asArray(value).flatMap((entry) => {
    const record = asRecord(entry);
    const author = stringValue(record.author);
    if (!author) return [];
    const lexical = asRecord(record.lexical);
    return [
      {
        author,
        itemCount: numberValue(record.itemCount),
        semanticUnitCount: numberValue(record.semanticUnitCount),
        wordCount: numberValue(lexical.wordCount),
        uniqueTerms: numberValue(lexical.uniqueTerms),
        conceptCount: asArray(record.concepts).length,
      },
    ];
  });
}

function conceptEvidence(value: unknown): ConceptEvidence[] {
  return asArray(value).flatMap((entry) => {
    const record = asRecord(entry);
    const clusterId = stringValue(record.clusterId);
    if (!clusterId) return [];
    const representative = asRecord(record.representative);
    return [
      {
        clusterId,
        memberUnitCount: numberValue(record.memberUnitCount),
        sourceItemCount: numberValue(record.sourceItemCount),
        authorCount: numberValue(record.authorCount),
        representativeText: stringValue(representative.text),
        representativeAuthor: optionalString(representative.author),
        representativeSource: optionalString(representative.source),
        representativeSourceId: stringValue(representative.sourceId),
      },
    ];
  });
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function stringValue(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function optionalString(value: unknown): string | undefined {
  const valueString = stringValue(value);
  return valueString || undefined;
}

function numberValue(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}
