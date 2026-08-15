import { expect, test } from "bun:test";

test("text-lexical-wasm package exports stable entrypoints", async () => {
  const entry = await import("../index.js");
  expect(typeof entry.init).toBe("function");
  expect(typeof entry.packageSurface).toBe("function");
  expect(typeof entry.runOperation).toBe("function");
});

test("runOperation reports corpus stats with structured output", async () => {
  const entry = await import("../index.js");
  const result = assertStructuredResponse(
    await entry.runOperation({
      operation: "lexical.corpusStats",
      input: {
        documents: [
          { id: "doc-1", text: "rust text analysis" },
          { id: "doc-2", text: "video scene analysis" },
        ],
        documentId: "doc-1",
        limit: 8,
      },
    }),
    "lexical.corpusStats",
  );
  expect(result.stats.documents).toBe(2);
  expect(result.terms.length).toBeGreaterThan(0);
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
