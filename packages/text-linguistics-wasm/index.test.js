import { beforeAll, expect, test } from "bun:test";

import init, {
  analyzeTextLinguistics,
  postprocessClassification,
  postprocessEmbeddings,
  postprocessSentiment,
  postprocessZeroShot,
  rerankFromImportedScores,
  summarizeLexical,
} from "./index.js";

beforeAll(async () => {
  await init();
});

test("analyzes linguistics through wasm heuristic mode", () => {
  const analysis = analyzeTextLinguistics("Alice works at OpenAI in Berlin.", {
    profile: "balanced",
    entityRecognition: "heuristic",
  });

  expect(analysis.model.entityRecognition).toBe("heuristic");
  expect(analysis.summary.tokenCount).toBeGreaterThan(0);
  expect(analysis.summary.entityCount).toBeGreaterThan(0);
});

test("accepts BERT-NER token predictions through wasm", () => {
  const analysis = analyzeTextLinguistics("Alice works at OpenAI.", {
    profile: "balanced",
    bertNerPredictions: [
      {
        kind: "token",
        label: "B-PER",
        text: "Alice",
        score: 0.99,
        attributes: { byte_start: "0", byte_end: "5", token_index: "1" },
      },
      {
        kind: "token",
        label: "B-ORG",
        text: "OpenAI",
        score: 0.98,
        attributes: { byte_start: "15", byte_end: "21", token_index: "4" },
      },
    ],
  });

  expect(analysis.model).toMatchObject({
    entityRecognition: "client-wasm-predictions",
    entityModel: "bert-base-ner",
  });
  expect(analysis.entities.map((entity) => entity.text)).toContain("Alice");
  expect(analysis.entities.map((entity) => entity.text)).toContain("OpenAI");
});

test("postprocesses shared NLP prediction payloads through wasm", () => {
  const classification = postprocessClassification("Rust is useful.", [
    { label: "technology", score: 0.8 },
    { label: "sports", score: 0.2 },
  ]);
  expect(classification.operation).toBe("classify");
  expect(classification.predictions[0].label).toBe("technology");

  const sentiment = postprocessSentiment("Great work.", [
    { label: "positive", score: 0.9 },
    { label: "negative", score: 0.1 },
  ]);
  expect(sentiment.operation).toBe("sentiment");
  expect(sentiment.label).toBe("positive");

  const embeddings = postprocessEmbeddings([[1, 0, 0]]);
  expect(embeddings.dimensions).toBe(3);

  const zeroShot = postprocessZeroShot("Rust API", ["technology", "music"], [
    { label: "technology", score: 0.9 },
  ]);
  expect(zeroShot.hypotheses[0]).toBe("This example is about technology.");
});

test("summarizes and reranks with browser-safe wasm helpers", () => {
  const summary = summarizeLexical(
    "Rust enables reliable systems. The API exposes task-specific NLP endpoints. Music theory is separate.",
    2,
  );
  expect(summary.operation).toBe("summarize");
  expect(summary.sentences.length).toBeGreaterThan(0);

  const reranked = rerankFromImportedScores("rust", ["music", "rust api"], [0.1, 0.9]);
  expect(reranked.results[0].document).toBe("rust api");
});
