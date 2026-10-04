// Deterministic NER protocol fixture; only the E2E server serves this file.
self.addEventListener("message", ({ data: { id, segments } }) => {
  const entities = [];
  for (let sentenceIndex = 0; sentenceIndex < segments.length; sentenceIndex += 1) {
    const segment = segments[sentenceIndex];
    const matches = [
      ["Maya", "PER"],
      ["Jonas", "PER"],
      ["Lina", "PER"],
      ["Sam", "PER"],
      ["Berlin", "LOC"],
    ];
    for (const [text, kind] of matches) {
      const index = segment.text.indexOf(text);
      if (index < 0) continue;
      const before = new TextEncoder().encode(segment.text.slice(0, index)).length;
      const bytes = new TextEncoder().encode(text).length;
      entities.push({
        text,
        kind,
        score: 0.99,
        span: { byte_start: segment.byteStart + before, byte_end: segment.byteStart + before + bytes },
        sentenceIndex,
      });
    }
  }
  self.postMessage({ id, modelName: "fixture/browser-ner-model", entities });
});
