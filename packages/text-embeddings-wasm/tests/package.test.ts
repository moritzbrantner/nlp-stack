import { expect, test } from "bun:test";

test("text-embeddings-wasm package exports stable entrypoints", async () => {
  const entry = await import("../index.js");
  expect(typeof entry.init).toBe("function");
  expect(typeof entry.packageSurface).toBe("function");
  expect(typeof entry.runOperation).toBe("function");
});

test("runOperation requires an embedding model instead of hashing", async () => {
  const entry = await import("../index.js");
  await expect(
    entry.runOperation({
      operation: "embeddings.similarity",
      input: { left: "rust text", right: "rust language" },
    }),
  ).rejects.toThrow(/missing_dependency/);
});

test("runOperation exposes backend catalog with structured output", async () => {
  const entry = await import("../index.js");
  const result = assertStructuredResponse(
    await entry.runOperation({
      operation: "embeddings.backends",
      input: {},
    }),
    "embeddings.backends",
  );
  expect(result.defaultBackend ?? null).toBeNull();
  expect(result.backends.every((backend: any) => backend.loadable === false)).toBe(true);
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
