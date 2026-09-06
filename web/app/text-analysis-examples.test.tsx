import { describe, expect, it } from "vitest";

import { defaultTextAnalysisExample, textAnalysisExamples } from "./text-analysis-examples";

describe("text analysis examples", () => {
  it("keeps a varied, selectable example catalog", () => {
    expect(textAnalysisExamples.length).toBeGreaterThanOrEqual(5);
    expect(new Set(textAnalysisExamples.map((example) => example.id)).size).toBe(textAnalysisExamples.length);
    expect(new Set(textAnalysisExamples.map((example) => example.category)).size).toBeGreaterThanOrEqual(4);
    expect(textAnalysisExamples.some((example) => example.focus === "semantic-corpus")).toBe(true);
    expect(textAnalysisExamples.some((example) => example.focus === "linguistics")).toBe(true);
    expect(textAnalysisExamples.every((example) => example.text.trim().length > 100)).toBe(true);
  });

  it("uses a real catalog entry as the default", () => {
    expect(textAnalysisExamples).toContain(defaultTextAnalysisExample);
  });
});
