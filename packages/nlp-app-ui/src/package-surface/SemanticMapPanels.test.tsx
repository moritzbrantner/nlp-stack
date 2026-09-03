import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { createSemanticMapResultTabs, SemanticMapPanel } from "./SemanticMapPanels";
import type { SurfaceResponse } from "./types";

const semanticResponse: SurfaceResponse = {
  operation: "analysis.semantic-map",
  diagnostics: [],
  artifacts: [],
  value: {
    title: "Semantic map result",
    result: {
      semantic: {
        timeline: [
          {
            unitId: "sentence-1",
            sequenceIndex: 0,
            clusterId: "concept-1",
            semanticShift: 0,
            clusterActivation: 1,
          },
          {
            unitId: "sentence-2",
            sequenceIndex: 1,
            clusterId: "concept-1",
            semanticShift: 0.12,
            clusterActivation: 0.91,
          },
          {
            unitId: "sentence-3",
            sequenceIndex: 2,
            clusterId: "concept-2",
            semanticShift: 0.84,
            clusterActivation: 1,
          },
        ],
        clusters: [
          {
            id: "concept-1",
            representativeText: "Semantic retrieval uses embeddings.",
            representativeUnitId: "sentence-1",
            memberUnitIds: ["sentence-1", "sentence-2"],
            meanSimilarity: 0.91,
          },
          {
            id: "concept-2",
            representativeText: "Tomatoes grow in garden soil.",
            representativeUnitId: "sentence-3",
            memberUnitIds: ["sentence-3"],
            meanSimilarity: 1,
          },
        ],
        hotspots: [
          {
            clusterId: "concept-1",
            coverage: 0.67,
            persistence: 0.67,
            meanActivation: 0.95,
            peakSequenceIndex: 0,
          },
        ],
      },
      linguisticGraph: {
        nodes: [
          { id: "concept:concept-1", kind: "concept", label: "Semantic retrieval uses embeddings." },
          { id: "sentence-1", kind: "unit", label: "Semantic retrieval uses embeddings.", sequenceIndex: 0 },
          { id: "mention:berlin", kind: "entityMention", label: "Berlin", sequenceIndex: 0, confidence: 0.9 },
        ],
        edges: [
          { sourceId: "sentence-1", targetId: "concept:concept-1", kind: "conceptMembership" },
          { sourceId: "sentence-1", targetId: "mention:berlin", kind: "unitContainsMention", label: "Location", weight: 0.9 },
        ],
      },
    },
  },
};

describe("SemanticMapPanel", () => {
  it("renders trajectory, hotspots, and the linguistic graph from a wrapped surface result", () => {
    render(<SemanticMapPanel response={semanticResponse} />);

    expect(screen.getByText("Meaning over sequence")).toBeTruthy();
    expect(screen.getByText("Semantic trajectory")).toBeTruthy();
    expect(screen.getByText("Concept hotspots")).toBeTruthy();
    expect(screen.getByText("Concept graph")).toBeTruthy();
    expect(screen.getByText("67% coverage")).toBeTruthy();
    expect(screen.getByRole("img", { name: "Semantic trajectory chart" })).toBeTruthy();
    expect(screen.getByRole("img", { name: "Semantic linguistic graph" })).toBeTruthy();
  });

  it("does not pretend another operation is a semantic map", () => {
    render(
      <SemanticMapPanel
        response={{ ...semanticResponse, operation: "analysis.document" }}
      />,
    );

    expect(screen.getByText("Run the semantic-map operation to explore its trajectory and graph.")).toBeTruthy();
  });

  it("provides a reusable result tab", () => {
    const tabs = createSemanticMapResultTabs();
    expect(tabs).toHaveLength(1);
    expect(tabs[0]?.id).toBe("semantic-map");
    expect(tabs[0]?.label).toBe("Map");
  });
});
