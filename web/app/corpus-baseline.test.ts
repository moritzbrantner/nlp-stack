import { describe, expect, it } from "vitest";

import {
  corpusBaselineFilename,
  createCorpusBaseline,
  serializeCorpusBaseline,
} from "./corpus-baseline";

const options = {
  topTerms: 128,
  minConceptUnits: 2,
  neighborsPerUnit: 4,
  neighborThreshold: 0.25,
  clusterThreshold: 0.6,
};

describe("corpus baseline export", () => {
  it("keeps source text, provenance, analysis options, and Rust semantic evidence together", () => {
    const baseline = createCorpusBaseline({
      label: "Reference corpus",
      documents: [
        {
          id: "doc-a-1",
          source: "doc-a.md",
          text: "  Alpha discusses durable semantic evidence.  ",
          ingestion: { method: "text" },
        },
        {
          id: "doc-b-2",
          source: "scan.pdf",
          text: "Beta repeats the semantic evidence across another source.",
          ingestion: { method: "pdf+ocr", pageCount: 2, ocrPageCount: 1 },
        },
      ],
      options,
      result: {
        itemCount: 2,
        concepts: [{ clusterId: "concept-1", sourceItemCount: 2 }],
      },
    });

    expect(baseline).toEqual({
      schemaVersion: 1,
      kind: "nlp-stack.semantic-corpus-baseline",
      label: "Reference corpus",
      documents: [
        {
          id: "doc-a-1",
          source: "doc-a.md",
          text: "Alpha discusses durable semantic evidence.",
          ingestion: { method: "text" },
        },
        {
          id: "doc-b-2",
          source: "scan.pdf",
          text: "Beta repeats the semantic evidence across another source.",
          ingestion: { method: "pdf+ocr", pageCount: 2, ocrPageCount: 1 },
        },
      ],
      analysis: {
        operation: "analysis.semantic-corpus",
        options,
        result: {
          itemCount: 2,
          concepts: [{ clusterId: "concept-1", sourceItemCount: 2 }],
        },
      },
    });
  });

  it("serializes deterministically without adding a timestamp", () => {
    const baseline = createCorpusBaseline({
      label: "Stable baseline",
      documents: [
        {
          id: "doc-1",
          source: "notes.txt",
          text: "Stable text.",
          ingestion: { method: "text" },
        },
      ],
      options,
      result: { itemCount: 1 },
    });

    const first = serializeCorpusBaseline(baseline);
    const second = serializeCorpusBaseline(baseline);

    expect(first).toBe(second);
    expect(first).not.toContain("createdAt");
    expect(first.endsWith("\n")).toBe(true);
  });

  it("uses a portable, predictable export filename", () => {
    expect(corpusBaselineFilename("  My Semantic Corpus 2026! ")).toBe(
      "my-semantic-corpus-2026.nlp-corpus.json",
    );
    expect(corpusBaselineFilename("///")).toBe("corpus-baseline.nlp-corpus.json");
  });

  it("rejects empty baseline inputs", () => {
    expect(() => createCorpusBaseline({
      label: "Empty",
      documents: [],
      options,
      result: {},
    })).toThrow(/at least one document/i);
  });
});
