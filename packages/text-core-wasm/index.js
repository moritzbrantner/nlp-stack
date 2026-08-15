import initWasm, { initSync } from "./pkg/moenarch_text_core_wasm.js";
import * as wasmModule from "./pkg/moenarch_text_core_wasm.js";

export async function init() {
  return wasmModule;
}

if (typeof process !== "undefined" && process.versions?.node) {
  const { readFileSync } = await import("node:fs");
  const wasmUrl = new URL("./pkg/moenarch_text_core_wasm_bg.wasm", import.meta.url);
  const wasmPath =
    wasmUrl.protocol === "file:"
      ? wasmUrl
      : decodeURIComponent(wasmUrl.pathname.replace(/^\/@fs/, ""));
  initSync({ module: readFileSync(wasmPath) });
} else {
  await initWasm();
}

export function packageSurface() {
  return fromWasmValue(wasmModule.packageSurface());
}

export function runOperation(request) {
  return fromWasmValue(wasmModule.runOperation(request));
}

export function extractWordTexts(text) {
  const response = runLoadedOperation("text.boundaries", { text });
  return response.words.filter(isTextSegment).map((word) => word.text);
}

export function splitSentences(text) {
  const response = runLoadedOperation("text.boundaries", { text });
  return response.sentences.map((sentence) => sentence.text);
}

export function segmentTextDocument(
  text,
  keepApostrophes = true,
  includePunctuation = false,
  includeTokens = true,
) {
  const boundaries = runLoadedOperation("text.boundaries", {
    keepApostrophes,
    text,
  });
  const tokenized = includeTokens
    ? runLoadedOperation("text.tokenize", {
        includePunctuation,
        text,
      })
    : { tokens: [] };

  return {
    paragraphs: boundaries.paragraphs.map((paragraph) => toSpan(paragraph, text)),
    sentences: boundaries.sentences.map((sentence) => toSpan(sentence, text)),
    tokens: tokenized.tokens.map((token) => toToken(token, text)),
  };
}

export function analyzeTextDocument(text, options = {}) {
  const includePunctuation = Boolean(options.includePunctuation);
  const includeTokens = options.includeTokens !== false;
  const tokenized = runLoadedOperation("text.tokenize", {
    includePunctuation,
    text,
  });
  const segmented = segmentTextDocument(text, true, includePunctuation, includeTokens);

  return {
    ...segmented,
    scriptProfile: toScriptProfile(tokenized.scriptProfile),
    stats: toStats(tokenized.stats),
    tokens: includeTokens ? tokenized.tokens.map((token) => toToken(token, text)) : [],
  };
}

function runLoadedOperation(operation, input) {
  return runOperation({ input, operation }).value;
}

function fromWasmValue(value) {
  if (value instanceof Map) {
    return Object.fromEntries(
      Array.from(value.entries(), ([key, entry]) => [key, fromWasmValue(entry)]),
    );
  }

  if (Array.isArray(value)) {
    return value.map(fromWasmValue);
  }

  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, entry]) => [key, fromWasmValue(entry)]),
    );
  }

  return value;
}

function toSpan(value, sourceText = "") {
  const span = value.span ?? {};

  return {
    end: toUtf16Offset(sourceText, span.byte_end ?? span.byteEnd, span.char_end ?? span.charEnd),
    start: toUtf16Offset(
      sourceText,
      span.byte_start ?? span.byteStart,
      span.char_start ?? span.charStart,
    ),
    text: value.text ?? "",
  };
}

function toToken(value, sourceText = "") {
  return {
    ...toSpan(value, sourceText),
    kind: toKind(value.kind),
    normalized: value.normalized,
  };
}

function toUtf16Offset(text, byteOffset, fallback = 0) {
  if (!text || byteOffset === undefined) {
    return fallback ?? 0;
  }

  let currentByteOffset = 0;
  let currentUtf16Offset = 0;

  for (const character of text) {
    if (currentByteOffset >= byteOffset) {
      return currentUtf16Offset;
    }

    currentByteOffset += new TextEncoder().encode(character).length;
    currentUtf16Offset += character.length;
  }

  return currentUtf16Offset;
}

function isTextSegment(segment) {
  return /[\p{L}\p{N}]/u.test(segment.text);
}

function toKind(kind) {
  return String(kind)
    .replace(/([a-z0-9])([A-Z])/g, "$1-$2")
    .toLowerCase();
}

function toScriptProfile(profile) {
  return {
    digits: profile.digits,
    dominantScript: profile.dominant_script ?? profile.dominantScript ?? null,
    isMixed: profile.is_mixed ?? profile.isMixed ?? false,
    other: profile.other,
    punctuation: profile.punctuation,
    scripts: profile.scripts ?? {},
    whitespace: profile.whitespace,
  };
}

function toStats(stats) {
  const basic = stats.basic ?? {};

  return {
    averageCharsPerWord: stats.average_chars_per_word ?? stats.averageCharsPerWord ?? 0,
    averageWordsPerSentence:
      stats.average_words_per_sentence ?? stats.averageWordsPerSentence ?? 0,
    basic,
    paragraphs: stats.paragraphs ?? 0,
    sentences: basic.sentences ?? 0,
    tokens: stats.tokens ?? 0,
    uniqueTokens: stats.unique_tokens ?? stats.uniqueTokens ?? 0,
  };
}

export default init;
