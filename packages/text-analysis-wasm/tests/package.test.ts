import { expect, test } from "bun:test";

import { analyzeDocument, compareTexts, packageSurface, runOperation } from "../index.js";

test("packageSurface lists document analysis", async () => {
  const surface = await packageSurface();
  expect(surface.operations.map((operation) => operation.id)).toContain("analysis.document");
});

test("analyzeDocument returns stats, keywords, and fingerprints", async () => {
  const report = await analyzeDocument({
    id: "doc-1",
    text: "Rust crates analyze text. Rust text analysis is deterministic.",
  });
  const result = report.result ?? report;
  expect(result.core.stats.basic.words).toBeGreaterThan(0);
  expect(result.lexical.keywords.length).toBeGreaterThan(0);
  expect(result.similarity.tokenShingleSimhash).toBeDefined();
});

test("compareTexts returns token shingle jaccard", async () => {
  const report = await compareTexts({
    left: "scene transitions follow motion",
    right: "scene transitions follow dialogue",
    n: 2,
    mode: "token",
  });
  const result = report.result ?? report;
  expect(result.similarity.jaccard).toBeGreaterThan(0);
});

test("runOperation executes document analysis with structured output", async () => {
  const result = assertStructuredResponse(
    await runOperation({
      operation: "analysis.document",
      input: {
        id: "doc-1",
        text: "Rust text analysis is deterministic.",
      },
    }),
    "analysis.document",
  );
  expect(result.core.stats.basic.words).toBeGreaterThan(0);
});

function assertStructuredResponse(response: any, operation: string) {
  expect(response.operation).toBe(operation);
  expect(response.value.operation).toBe(operation);
  expect(typeof response.value.title).toBe("string");
  expect(typeof response.value.message).toBe("string");
  expect(response.value.summary.status).toBe("ok");
  expect(response.value.result).toBeDefined();
  return response.value.result;
}
