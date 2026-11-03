import { expect, test } from "bun:test";

test("text-generation-linguistics-wasm package exports stable entrypoints", async () => {
  const entry = await import("../index.js");
  expect(typeof entry.init).toBe("function");
  expect(typeof entry.packageSurface).toBe("function");
  expect(typeof entry.runOperation).toBe("function");
});

test("runOperation requires the linguistic models in the browser build", async () => {
  const entry = await import("../index.js");
  await expect(
    entry.runOperation({
      operation: "generationLinguistics.analysisTerms",
      input: { text: "Alice presented the tokenizer roadmap in Berlin." },
    }),
  ).rejects.toThrow(/missing_dependency/);
});
