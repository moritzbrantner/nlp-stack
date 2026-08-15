import { expect, test } from "bun:test";

test("text-question-answering-wasm package exports stable entrypoints", async () => {
  const entry = await import("../index.js");
  expect(typeof entry.init).toBe("function");
  expect(typeof entry.packageSurface).toBe("function");
  expect(typeof entry.runOperation).toBe("function");
});

test("runOperation answers from imported spans with structured output", async () => {
  const entry = await import("../index.js");
  const result = assertStructuredResponse(
    await entry.runOperation({
      operation: "qa.answer",
      input: {
        question: "What is reliable?",
        context: "Rust is reliable.",
        importedPredictions: [{ text: "Rust", score: 0.9 }],
      },
    }),
    "qa.answer",
  );
  expect(result.answers.length).toBeGreaterThan(0);
});

test("runOperation answers from an indexed document with structured output", async () => {
  const entry = await import("../index.js");
  const result = assertStructuredResponse(
    await entry.runOperation({
      operation: "qa.answerWithIndex",
      input: {
        question: "What language has ownership?",
        documents: [
          {
            id: "doc-rust",
            body: "Rust has ownership and deterministic package workflows.",
            language: "en",
          },
        ],
        indexOptions: {
          chunkingStrategy: "tokenWindow",
          chunkTokens: 16,
          chunkOverlapTokens: 0,
          storeRawText: true,
        },
        topKChunks: 2,
        topKAnswers: 1,
        localModel: { autoDownload: false },
        fallbackPolicy: "heuristicIfUnavailable",
      },
    }),
    "qa.answerWithIndex",
  );
  expect(result.answers.length).toBeGreaterThan(0);
  expect(result.answers[0].citations.length).toBeGreaterThan(0);
});

test("packageSurface marks retrieval QA as deprecated support", async () => {
  const entry = await import("../index.js");
  const surface = await entry.packageSurface();
  const operation = surface.operations.find((operation: any) => operation.id === "qa.answerWithRetrieval");
  expect(operation.inputSchema.xOperationCategory).toBe("support");
  expect(operation.inputSchema.xDeprecated).toBe(true);
  expect(operation.inputSchema.xReplacementOperation).toBe("qa.answerWithIndex");
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
