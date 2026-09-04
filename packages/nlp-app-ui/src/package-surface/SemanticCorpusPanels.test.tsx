import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { createSemanticCorpusResultTabs, SemanticCorpusPanel } from "./SemanticCorpusPanels";
import type { SurfaceResponse } from "./types";

const corpusResponse: SurfaceResponse = {
  operation: "analysis.semantic-corpus",
  diagnostics: [],
  artifacts: [],
  value: {
    title: "Semantic corpus profile",
    result: {
      itemCount: 3,
      authorCount: 2,
      lexical: {
        wordCount: 18,
        uniqueTerms: 12,
        lexicalDiversity: 0.67,
        topTerms: [
          { term: "semantic", count: 4, frequency: 0.22 },
          { term: "retrieval", count: 3, frequency: 0.17 },
        ],
      },
      authors: [
        {
          author: "Alice",
          itemCount: 2,
          semanticUnitCount: 4,
          lexical: { wordCount: 12, uniqueTerms: 8 },
          concepts: [{ clusterId: "concept-1", unitCount: 4, share: 1 }],
        },
        {
          author: "Bob",
          itemCount: 1,
          semanticUnitCount: 2,
          lexical: { wordCount: 6, uniqueTerms: 5 },
          concepts: [{ clusterId: "concept-2", unitCount: 2, share: 1 }],
        },
      ],
      concepts: [
        {
          clusterId: "concept-1",
          memberUnitCount: 4,
          sourceItemCount: 2,
          authorCount: 1,
          representative: {
            sourceId: "alice-1",
            author: "Alice",
            source: "letters/alice-1.txt",
            text: "Semantic search improves retrieval.",
          },
        },
      ],
      semantic: {
        timeline: [
          {
            unitId: "alice-1:sentence:0",
            sequenceIndex: 0,
            clusterId: "concept-1",
            semanticShift: 0,
            clusterActivation: 1,
          },
        ],
        clusters: [
          {
            id: "concept-1",
            representativeText: "Semantic search improves retrieval.",
            memberUnitIds: ["alice-1:sentence:0"],
            meanSimilarity: 1,
          },
        ],
        hotspots: [],
      },
    },
  },
};

describe("SemanticCorpusPanel", () => {
  it("renders vocabulary, author profiles, and source-backed concept evidence", () => {
    render(<SemanticCorpusPanel response={corpusResponse} />);

    expect(screen.getByText("Language and meaning across sources")).toBeTruthy();
    expect(screen.getByText("Vocabulary profile")).toBeTruthy();
    expect(screen.getByText("Author profiles")).toBeTruthy();
    expect(screen.getByText("Concept evidence")).toBeTruthy();
    expect(screen.getByText("semantic")).toBeTruthy();
    expect(screen.getByText("Alice")).toBeTruthy();
    expect(screen.getByText("Semantic search improves retrieval.")).toBeTruthy();
    expect(screen.getByText("letters/alice-1.txt")).toBeTruthy();
  });

  it("does not render another operation as a corpus profile", () => {
    render(
      <SemanticCorpusPanel
        response={{ ...corpusResponse, operation: "analysis.document" }}
      />,
    );

    expect(
      screen.getByText(
        "Run semantic corpus analysis to inspect vocabulary, authors, and concept evidence.",
      ),
    ).toBeTruthy();
  });

  it("provides dedicated corpus and semantic map tabs", () => {
    const tabs = createSemanticCorpusResultTabs();
    expect(tabs).toHaveLength(2);
    expect(tabs[0]?.id).toBe("semantic-corpus");
    expect(tabs[1]?.id).toBe("semantic-corpus-map");
  });
});
