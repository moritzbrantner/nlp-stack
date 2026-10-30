// Canonical browser NER contract shared by the adapter, its worker and tests.
// Pure module: importing it performs no network or model requests.

export const BROWSER_NER_PRODUCER = Object.freeze({
  package: "@moritzbrantner/text-linguistics-wasm",
  module: "browser-ner",
  contractVersion: 1,
});

export const BROWSER_NER_MODEL = Object.freeze({
  id: "onnx-community/distilbert-NER-ONNX",
  // Pinned Hugging Face commit; never follow the moving `main` revision.
  revision: "3a19fe9404a4469d91aa3d551558a97f68872f67",
  baseModel: "dslim/distilbert-NER",
  dtype: "q4",
  labels: Object.freeze(["PER", "ORG", "LOC", "MISC"]),
  // distilbert-NER is trained on English CoNLL-2003 only (#117 owns language parity).
  supportedLanguages: Object.freeze(["en"]),
});

export const BROWSER_NER_TRANSFORMERS_MODULE_URL =
  "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

export function browserNerCapabilities() {
  return {
    producer: { ...BROWSER_NER_PRODUCER },
    model: {
      ...BROWSER_NER_MODEL,
      labels: [...BROWSER_NER_MODEL.labels],
      supportedLanguages: [...BROWSER_NER_MODEL.supportedLanguages],
    },
    runtime: { transformersModuleUrl: BROWSER_NER_TRANSFORMERS_MODULE_URL },
  };
}

/** Returns the primary subtag of a BCP 47 tag (`en-US` -> `en`), or `null`. */
export function primaryLanguage(language) {
  if (typeof language !== "string") return null;
  const primary = language.trim().toLowerCase().split(/[-_]/)[0];
  return primary.length > 0 ? primary : null;
}

export function supportsBrowserNerLanguage(language) {
  const primary = primaryLanguage(language);
  return primary !== null && BROWSER_NER_MODEL.supportedLanguages.includes(primary);
}

/** Validates caller segments; returns normalized `{ id, text, byteStart }` copies. */
export function normalizeBrowserNerSegments(segments) {
  if (!Array.isArray(segments)) {
    throw new TypeError("Browser NER segments must be an array.");
  }
  const seen = new Set();
  return segments.map((segment, index) => {
    const id = segment?.id ?? index;
    if (typeof id !== "string" && !Number.isInteger(id)) {
      throw new TypeError(`Browser NER segment ${index} id must be a string or integer.`);
    }
    if (seen.has(id)) {
      throw new TypeError(`Browser NER segment id ${JSON.stringify(id)} is duplicated.`);
    }
    seen.add(id);
    if (typeof segment?.text !== "string") {
      throw new TypeError(`Browser NER segment ${JSON.stringify(id)} text must be a string.`);
    }
    const byteStart = segment.byteStart ?? 0;
    if (!Number.isInteger(byteStart) || byteStart < 0) {
      throw new TypeError(`Browser NER segment ${JSON.stringify(id)} byteStart must be a non-negative integer.`);
    }
    return { id, text: segment.text, byteStart };
  });
}

const encoder = new TextEncoder();

function utf8Length(value) {
  return encoder.encode(value).length;
}

function parseLabel(label) {
  if (typeof label !== "string") return null;
  const upper = label.trim().toUpperCase();
  if (upper === "O" || upper === "") return null;
  const match = /^([BIES])-(.+)$/.exec(upper);
  return match ? { prefix: match[1], kind: match[2] } : { prefix: "B", kind: upper };
}

/**
 * Aligns raw (unaggregated) WordPiece predictions to `text` and aggregates them into
 * whole-word entity spans: the first sub-word's label applies to the whole word (HF
 * `aggregation_strategy="first"`), then words are grouped by BIO prefix.
 *
 * `predictions` are Transformers.js token-classification items produced with
 * `ignore_labels: []`, in token order: `{ word, entity, score }`. Pieces that cannot be
 * located in `text` (for example `[UNK]`) end the current word without inventing offsets.
 *
 * Returns `{ entities, alignedThrough }`; `alignedThrough` is the UTF-16 offset of the
 * last aligned character, so callers can tell when model truncation left text unlabeled.
 */
export function entitiesFromTokenPredictions(text, predictions) {
  const words = [];
  let cursor = 0;
  let current = null;
  for (const prediction of predictions) {
    const raw = typeof prediction?.word === "string" ? prediction.word : "";
    const continuation = raw.startsWith("##");
    const piece = continuation ? raw.slice(2) : raw;
    const start = piece.length > 0 ? text.indexOf(piece, cursor) : -1;
    if (start < 0 || (continuation && (!current || start !== cursor))) {
      // Unalignable piece ([UNK], or a detached continuation): end the current word
      // rather than inventing offsets.
      current = null;
      continue;
    }
    const end = start + piece.length;
    if (continuation) {
      current.end = end;
    } else {
      current = {
        start,
        end,
        label: parseLabel(prediction.entity),
        score: Number(prediction.score ?? 0),
      };
      words.push(current);
    }
    cursor = end;
  }

  const entities = [];
  let group = null;
  for (const word of words) {
    const label = word.label;
    const continuesGroup =
      group && label && label.kind === group.kind && (label.prefix === "I" || label.prefix === "E");
    if (continuesGroup) {
      group.end = word.end;
      group.scores.push(word.score);
      continue;
    }
    if (group) entities.push(group);
    group = label ? { kind: label.kind, start: word.start, end: word.end, scores: [word.score] } : null;
  }
  if (group) entities.push(group);

  return {
    alignedThrough: cursor,
    entities: entities.map(({ kind, start, end, scores }) => ({
      label: kind,
      text: text.slice(start, end),
      start,
      end,
      score: scores.reduce((sum, score) => sum + score, 0) / scores.length,
    })),
  };
}

/** Projects per-segment predictions into source-span evidence with absolute UTF-8 offsets. */
export function projectBrowserNerSegments(segments, predictionsBySegment) {
  if (!Array.isArray(predictionsBySegment) || predictionsBySegment.length !== segments.length) {
    throw new Error(`Browser NER model ${BROWSER_NER_MODEL.id} returned an unexpected result shape.`);
  }
  const entities = [];
  const segmentReports = segments.map((segment, index) => {
    const predictions = predictionsBySegment[index];
    if (!Array.isArray(predictions)) {
      throw new Error(`Browser NER model ${BROWSER_NER_MODEL.id} returned an unexpected segment result.`);
    }
    const { entities: local, alignedThrough } = entitiesFromTokenPredictions(segment.text, predictions);
    for (const entity of local) {
      const byteStart = segment.byteStart + utf8Length(segment.text.slice(0, entity.start));
      entities.push({
        segmentId: segment.id,
        label: entity.label,
        text: entity.text,
        score: entity.score,
        start: entity.start,
        end: entity.end,
        byteStart,
        byteEnd: byteStart + utf8Length(entity.text),
      });
    }
    const unlabeledTail = segment.text.slice(alignedThrough);
    return {
      id: segment.id,
      entityCount: local.length,
      truncated: /\S/.test(unlabeledTail),
    };
  });
  return { entities, segments: segmentReports };
}
