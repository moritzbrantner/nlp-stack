// Deterministic embedding protocol fixture; only the E2E server serves this file.
self.addEventListener("message", ({ data: { id, texts } }) => {
  self.postMessage({
    id,
    modelName: "fixture/browser-semantic-model",
    dimensions: 2,
    vectors: texts.map(() => [1, 0]),
  });
});
