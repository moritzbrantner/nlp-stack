import { beforeAll, expect, test } from "bun:test";

import init, {
  analyzeTextDocument,
  extractWordTexts,
  segmentTextDocument,
  splitSentences,
} from "./index.js";

beforeAll(async () => {
  await init();
});

test("segments text through the packaged wasm bindings", () => {
  expect(extractWordTexts("Rust tests, shipped.")).toEqual(["Rust", "tests", "shipped"]);
  expect(splitSentences("First sentence. Second sentence.")).toEqual([
    "First sentence.",
    "Second sentence.",
  ]);

  const document = segmentTextDocument("Hello from wasm.", true, false, true);
  expect(document.sentences[0].text).toBe("Hello from wasm.");
  expect(document.tokens.map((token) => token.text)).toEqual(["Hello", "from", "wasm"]);
});

test("analyzes document stats, scripts, and normalized tokens", () => {
  const document = analyzeTextDocument("Hello 東京!\n\nCafe\u0301 time.", {
    includePunctuation: true,
  });

  expect(document.stats.paragraphs).toBe(2);
  expect(document.stats.sentences).toBe(2);
  expect(document.stats.tokens).toBeGreaterThan(0);
  expect(document.stats.uniqueTokens).toBeGreaterThan(0);
  expect(document.scriptProfile.scripts.Latin).toBe(13);
  expect(document.scriptProfile.scripts.Han).toBe(2);
  expect(document.scriptProfile.isMixed).toBe(true);
  expect(document.tokens[0]).toMatchObject({
    start: 0,
    end: 5,
    text: "Hello",
    normalized: "hello",
    kind: "word",
  });
});

test("analyzeTextDocument honors punctuation and token output options", () => {
  const withPunctuation = analyzeTextDocument("Hello, world.", {
    includePunctuation: true,
  });
  const withoutPunctuation = analyzeTextDocument("Hello, world.", {
    includePunctuation: false,
  });
  const withoutTokens = analyzeTextDocument("Hello, world.", {
    includePunctuation: true,
    includeTokens: false,
  });

  expect(withPunctuation.tokens.map((token) => token.text)).toEqual(["Hello", ",", "world", "."]);
  expect(withoutPunctuation.tokens.map((token) => token.text)).toEqual(["Hello", "world"]);
  expect(withoutTokens.tokens).toEqual([]);
  expect(withoutTokens.stats.tokens).toBe(4);
});

test("analyzeTextDocument reports UTF-16 offsets for browser selections", () => {
  const document = analyzeTextDocument("A😀B.\n\nCafe\u0301 time.", {
    includePunctuation: true,
  });

  expect(document.paragraphs[0]).toMatchObject({
    start: 0,
    end: 5,
    text: "A😀B.",
  });
  expect(document.paragraphs[1]).toMatchObject({
    start: 7,
    end: 18,
    text: "Cafe\u0301 time.",
  });
  expect(document.tokens[1]).toMatchObject({
    start: 1,
    end: 3,
    text: "😀",
    kind: "other",
  });
  expect(document.tokens[5]).toMatchObject({
    start: 11,
    end: 12,
    text: "\u0301",
    kind: "other",
  });
});
