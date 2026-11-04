import { expect, test } from "bun:test";

test("text-retrieval-wasm package exports stable entrypoints", async () => {
  const entry = await import("../index.js");
  expect(typeof entry.init).toBe("function");
  expect(typeof entry.packageSurface).toBe("function");
  expect(typeof entry.runOperation).toBe("function");
});

test("runOperation requires an embedding model for index-backed search", async () => {
  const entry = await import("../index.js");
  await expect(
    entry.runOperation({
      operation: "retrieval.search",
      input: {
        documents: [{ id: "doc-1", body: "rust retrieval search" }],
        query: "rust",
        mode: "full_text",
      },
    }),
  ).rejects.toThrow(/missing_dependency/);
});
