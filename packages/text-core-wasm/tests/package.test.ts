import { beforeAll, expect, test } from "bun:test";

let entry: any;

beforeAll(async () => {
  entry = await import("../index.js");
  await entry.init();
});

test("text-core-wasm package exports stable entrypoints", () => {
  expect(typeof entry.init).toBe("function");
  expect(typeof entry.packageSurface).toBe("function");
  expect(typeof entry.runOperation).toBe("function");
});

test("runOperation tokenizes text with structured output", () => {
  const result = assertStructuredResponse(
    entry.runOperation({
      operation: "text.tokenize",
      input: { text: "Rust text crates.", includePunctuation: false },
    }),
    "text.tokenize",
  );
  expect(result.tokens.length).toBeGreaterThan(0);
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
