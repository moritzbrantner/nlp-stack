import { expect, test } from "bun:test";

test("text-linguistics-wasm package exports stable entrypoints", async () => {
  const entry = await import("../index.js");
  expect(typeof entry.init).toBe("function");
  expect(typeof entry.packageSurface).toBe("function");
  expect(typeof entry.runOperation).toBe("function");
});

test("runOperation detects language with structured output", async () => {
  const entry = await import("../index.js");
  const result = assertStructuredResponse(
    await entry.runOperation({
      operation: "linguistics.language",
      input: {
        text: "This is a simple English sentence.",
        sentenceLevel: true,
        maxAlternatives: 3,
      },
    }),
    "linguistics.language",
  );
  expect(result.primary.language).toBe("en");
});

test("runOperation reports model required for analysis in the browser build", async () => {
  const entry = await import("../index.js");
  await expect(
    entry.runOperation({
      operation: "linguistics.analyze",
      input: { text: "Alice presented the roadmap in Berlin.", profile: "fast" },
    }),
  ).rejects.toThrow(/missing_dependency/);
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
