const transformersModuleUrl = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";
const semanticEmbeddingModel = "onnx-community/all-MiniLM-L6-v2-ONNX";
let semanticExtractorPromise = null;

async function semanticExtractor() {
  if (!semanticExtractorPromise) {
    semanticExtractorPromise = import(transformersModuleUrl)
      .then(({ pipeline }) => pipeline("feature-extraction", semanticEmbeddingModel, { dtype: "q4" }))
      .catch((error) => {
        semanticExtractorPromise = null;
        throw error;
      });
  }
  return semanticExtractorPromise;
}

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}

self.addEventListener("message", async (event) => {
  const id = event.data?.id;
  const texts = event.data?.texts;
  if (!Number.isInteger(id) || !Array.isArray(texts) || texts.length === 0) {
    return;
  }

  try {
    const extractor = await semanticExtractor();
    const output = await extractor(texts, { pooling: "mean", normalize: true });
    const vectors = output.tolist();
    if (!Array.isArray(vectors) || vectors.length !== texts.length || !Array.isArray(vectors[0])) {
      throw new Error(`Hugging Face semantic model ${semanticEmbeddingModel} returned an unexpected embedding shape.`);
    }

    self.postMessage({
      id,
      modelName: semanticEmbeddingModel,
      dimensions: vectors[0].length,
      vectors,
    });
  } catch (error) {
    self.postMessage({ id, error: errorMessage(error) });
  }
});
