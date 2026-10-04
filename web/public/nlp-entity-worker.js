const transformersModuleUrl = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";
const namedEntityModel = "onnx-community/distilbert-NER-ONNX";
let classifierPromise = null;

async function classifier() {
  if (!classifierPromise) {
    classifierPromise = import(transformersModuleUrl)
      .then(({ pipeline }) => pipeline("token-classification", namedEntityModel, { dtype: "q4" }))
      .catch((error) => {
        classifierPromise = null;
        throw error;
      });
  }
  return classifierPromise;
}

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}

function byteLength(value) {
  return new TextEncoder().encode(value).length;
}

self.addEventListener("message", async (event) => {
  const id = event.data?.id;
  const segments = event.data?.segments;
  if (!Number.isInteger(id) || !Array.isArray(segments) || segments.length === 0) {
    return;
  }

  try {
    const ner = await classifier();
    const texts = segments.map((segment) => segment.text);
    const raw = await ner(texts, { aggregation_strategy: "simple" });
    const grouped = segments.length === 1 && Array.isArray(raw) && !Array.isArray(raw[0])
      ? [raw]
      : raw;
    if (!Array.isArray(grouped) || grouped.length !== segments.length) {
      throw new Error(`Hugging Face NER model ${namedEntityModel} returned an unexpected result shape.`);
    }

    const entities = [];
    grouped.forEach((predictions, segmentIndex) => {
      const segment = segments[segmentIndex];
      if (!Array.isArray(predictions)) {
        throw new Error(`Hugging Face NER model ${namedEntityModel} returned an unexpected segment result.`);
      }
      for (const prediction of predictions) {
        const start = prediction?.start;
        const end = prediction?.end;
        const group = prediction?.entity_group ?? prediction?.entity;
        if (!Number.isInteger(start) || !Number.isInteger(end) || end <= start || typeof group !== "string") {
          continue;
        }
        const localText = segment.text.slice(start, end);
        const byteStart = segment.byteStart + byteLength(segment.text.slice(0, start));
        const byteEnd = byteStart + byteLength(localText);
        entities.push({
          text: localText,
          kind: group.replace(/^[BI]-/, ""),
          score: Number(prediction.score ?? 0),
          span: { byte_start: byteStart, byte_end: byteEnd },
          sentenceIndex: segmentIndex,
        });
      }
    });

    self.postMessage({ id, modelName: namedEntityModel, entities });
  } catch (error) {
    self.postMessage({ id, error: errorMessage(error) });
  }
});
