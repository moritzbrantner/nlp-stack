export const corpusBaselineSchemaVersion = 1 as const;
export const corpusBaselineKind = "nlp-stack.semantic-corpus-baseline" as const;

export type CorpusBaselineDocument = {
  id: string;
  source: string;
  text: string;
  ingestion: {
    method: string;
    pageCount?: number;
    ocrPageCount?: number;
  };
};

export type SemanticCorpusBaselineOptions = {
  topTerms: number;
  minConceptUnits: number;
  neighborsPerUnit: number;
  neighborThreshold: number;
  clusterThreshold: number;
};

export type CorpusBaseline = {
  schemaVersion: typeof corpusBaselineSchemaVersion;
  kind: typeof corpusBaselineKind;
  label: string;
  documents: CorpusBaselineDocument[];
  analysis: {
    operation: "analysis.semantic-corpus";
    options: SemanticCorpusBaselineOptions;
    result: Record<string, unknown>;
  };
};

export function createCorpusBaseline(input: {
  label: string;
  documents: CorpusBaselineDocument[];
  options: SemanticCorpusBaselineOptions;
  result: Record<string, unknown>;
}): CorpusBaseline {
  if (input.documents.length === 0) {
    throw new Error("A corpus baseline requires at least one document.");
  }

  const documents = input.documents.map((document) => ({
    ...document,
    text: document.text.trim(),
    ingestion: { ...document.ingestion },
  }));

  if (documents.some((document) => !document.text)) {
    throw new Error("Corpus baseline documents must contain text.");
  }

  return {
    schemaVersion: corpusBaselineSchemaVersion,
    kind: corpusBaselineKind,
    label: input.label.trim() || "Corpus baseline",
    documents,
    analysis: {
      operation: "analysis.semantic-corpus",
      options: { ...input.options },
      result: input.result,
    },
  };
}

export function serializeCorpusBaseline(baseline: CorpusBaseline): string {
  return `${JSON.stringify(baseline, null, 2)}\n`;
}

export function corpusBaselineFilename(label: string): string {
  const slug = label
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 80);

  return `${slug || "corpus-baseline"}.nlp-corpus.json`;
}
