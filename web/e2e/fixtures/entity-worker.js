// Deterministic browser NER worker fixture; only the E2E server serves this file in place
// of /ner/browser-ner-worker.js. It replaces the model with fixed WordPiece predictions and
// runs them through the canonical projection from the copied package module.
import { projectBrowserNerSegments } from "/ner/browser-ner-core.js";

const labels = new Map([
  ["Maya", "B-PER"],
  ["Jonas", "B-PER"],
  ["Lina", "B-PER"],
  ["Sam", "B-PER"],
  ["Berlin", "B-LOC"],
]);

function predictions(text) {
  return Array.from(text.matchAll(/\p{L}+|\S/gu), ([word]) => ({
    word,
    entity: labels.get(word) ?? "O",
    score: 0.99,
  }));
}

self.addEventListener("message", ({ data: { type, id, segments } }) => {
  if (type === "prepare") {
    self.postMessage({ type: "ready", id });
    return;
  }
  const result = projectBrowserNerSegments(segments, segments.map((segment) => predictions(segment.text)));
  self.postMessage({ type: "result", id, ...result });
});
