import { expect, test } from "bun:test";

test("text-transcripts-wasm package exports stable entrypoints", async () => {
  const entry = await import("../index.js");
  expect(typeof entry.init).toBe("function");
  expect(typeof entry.packageSurface).toBe("function");
  expect(typeof entry.runOperation).toBe("function");
});

test("runOperation parses transcripts with structured output", async () => {
  const entry = await import("../index.js");
  const result = assertStructuredResponse(
    await entry.runOperation({
      operation: "transcripts.parse",
      input: {
        format: "srt",
        content: "1\n00:00:01,000 --> 00:00:02,000\nHello.\n",
      },
    }),
    "transcripts.parse",
  );
  expect(result.segments.length).toBeGreaterThan(0);
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
