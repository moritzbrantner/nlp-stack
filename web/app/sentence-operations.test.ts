import { describe, expect, it } from "vitest";

import { sentenceViews } from "./sentence-operations";

const text = "Sarah reports that John believes the river is polluted.";
const span = (byteStart: number, byteEnd: number) => ({ byte_start: byteStart, byte_end: byteEnd });
const passage = (byteStart: number, byteEnd: number) => ({ text: text.slice(byteStart, byteEnd), span: span(byteStart, byteEnd) });

function documentReport(reporting: unknown) {
  return {
    core: { sentences: [{ text, span: span(0, text.length), token_count: 10 }] },
    linguistic: {
      sentences: [{ sentenceIndex: 0, span: span(0, text.length), language: "en", operations: [] }],
      reporting,
    },
  };
}

describe("reporting evidence view", () => {
  it("nests the belief construction inside the report", () => {
    const [view] = sentenceViews(
      documentReport({
        constructions: [
          { id: 0, parent: null, children: [1], sentenceIndex: 0, kind: "speech", form: "indirect", cue: passage(6, 13), cueLemma: "report", source: passage(0, 5), content: passage(19, 54), confidence: 0.99 },
          { id: 1, parent: 0, children: [], sentenceIndex: 0, kind: "belief", form: "indirect", cue: passage(24, 32), cueLemma: "believe", source: passage(19, 23), content: passage(33, 54), confidence: 0.98 },
        ],
        sentences: [{ sentenceIndex: 0, support: "supported", producer: "ud-model:combo-english-ewt-ud2.17", reason: "", constructionCount: 2 }],
      }),
    );
    expect(view!.reporting.status).toBe("supported");
    expect(view!.reporting.roots).toHaveLength(1);
    const [outer] = view!.reporting.roots;
    expect([outer!.kind, outer!.source?.text, outer!.cue.text, outer!.content.text]).toEqual([
      "speech",
      "Sarah",
      "reports",
      "John believes the river is polluted",
    ]);
    const [inner] = outer!.children;
    expect([inner!.kind, inner!.source?.text, inner!.cue.text, inner!.content.text]).toEqual([
      "belief",
      "John",
      "believes",
      "the river is polluted",
    ]);
    expect([inner!.content.byteStart, inner!.content.byteEnd]).toEqual([33, 54]);
  });

  it("keeps an unsupported sentence's reason and shows no construction", () => {
    const [view] = sentenceViews(
      documentReport({
        constructions: [],
        sentences: [{ sentenceIndex: 0, support: "unsupported", producer: null, reason: "no dependency tree: not requested (DependencyParsingMode::Disabled); no heuristic fallback", constructionCount: 0 }],
      }),
    );
    expect(view!.reporting).toEqual({
      status: "unsupported",
      producer: null,
      detail: "no dependency tree: not requested (DependencyParsingMode::Disabled); no heuristic fallback",
      roots: [],
    });
  });
});

describe("scope evidence view", () => {
  const negated = "Sarah does not say John believes it.";
  const at = (byteStart: number, byteEnd: number) => ({ text: negated.slice(byteStart, byteEnd), span: span(byteStart, byteEnd) });
  const report = (scope: unknown) => ({
    core: { sentences: [{ text: negated, span: span(0, negated.length), token_count: 8 }] },
    linguistic: {
      sentences: [{ sentenceIndex: 0, span: span(0, negated.length), language: "en", operations: [] }],
      reporting: {
        constructions: [
          { id: 0, parent: null, children: [1], sentenceIndex: 0, kind: "speech", form: "indirect", cue: at(15, 18), cueLemma: "say", source: at(0, 5), content: at(19, 35), confidence: 0.99 },
          { id: 1, parent: 0, children: [], sentenceIndex: 0, kind: "belief", form: "indirect", cue: at(24, 32), cueLemma: "believe", source: at(19, 23), content: at(33, 35), confidence: 0.98 },
        ],
        sentences: [{ sentenceIndex: 0, support: "supported", producer: "ud-model:x", reason: "", constructionCount: 2 }],
      },
      scope,
    },
  });

  it("links a negated report to the report, not the embedded belief", () => {
    const [view] = sentenceViews(
      report({
        operators: [
          { id: 0, parent: null, children: [], sentenceIndex: 0, kind: "negation", resolution: "resolved", unresolvedReason: null, cue: at(11, 14), cueRelation: "advmod", head: at(15, 18), scope: at(0, 35), consequent: null, reportingTarget: 0, reportedWithin: null, confidence: 0.97 },
        ],
        sentences: [{ sentenceIndex: 0, support: "supported", producer: "ud-model:x", reason: "", operatorCount: 1 }],
      }),
    );
    const [negation] = view!.scope.roots;
    expect(view!.scope.status).toBe("supported");
    expect([negation!.kind, negation!.cue.text, negation!.head?.text, negation!.reportingTarget?.label]).toEqual([
      "negation",
      "not",
      "say",
      "Sarah — say",
    ]);
    expect(negation!.reportedWithin).toBeNull();
    expect([negation!.cue.byteStart, negation!.cue.byteEnd]).toEqual([11, 14]);
  });

  it("keeps an unresolved operator without inventing a scope", () => {
    const [view] = sentenceViews(
      report({
        operators: [
          { id: 0, parent: null, children: [], sentenceIndex: 0, kind: "negation", resolution: "unresolved", unresolvedReason: "the cue attaches as `conj`", cue: at(11, 14), cueRelation: "conj", head: null, scope: null, consequent: null, reportingTarget: null, reportedWithin: null, confidence: null },
        ],
        sentences: [{ sentenceIndex: 0, support: "supported", producer: "ud-model:x", reason: "", operatorCount: 1 }],
      }),
    );
    const [negation] = view!.scope.roots;
    expect([negation!.resolved, negation!.scope, negation!.unresolvedReason]).toEqual([false, null, "the cue attaches as `conj`"]);
  });

  it("reports unknown when the analysis has no scope section", () => {
    const [view] = sentenceViews(report(undefined));
    expect(view!.scope.status).toBe("unknown");
    expect(view!.scope.roots).toEqual([]);
  });
});

describe("syntax view", () => {
  it("lists the backend's syntactic words with versions and multi-word tokens", () => {
    const sentence = "Er geht zum Bahnhof.";
    const word = (token: number, form: string, lemma: string, upos: string, head: number | null, relation: string, feats = {}) => ({
      token, form, lemma, upos, xpos: null, feats, head, relation, subtype: null,
    });
    const report = {
      core: { sentences: [{ text: sentence, span: span(0, sentence.length), token_count: 5 }] },
      linguistic: {
        sentences: [{ sentenceIndex: 0, span: span(0, sentence.length), language: "de", operations: [] }],
        syntax: [{
          sentenceIndex: 0,
          language: "de",
          provenance: {
            backend: "stanza",
            backendVersion: "1.15.0",
            models: [
              { stage: "depparse", package: "combined_charlm", source: "stanfordnlp/stanza-de", revision: "e7acb8e5f0ff683032c9c60fe95c2a003c672a16" },
              { stage: "tokenize", package: "combined_nocharlm", source: "stanfordnlp/stanza-de", revision: "e7acb8e5f0ff683032c9c60fe95c2a003c672a16" },
            ],
          },
          tokens: [
            { span: span(0, 2), text: "Er", firstWord: 0, wordCount: 1 },
            { span: span(3, 7), text: "geht", firstWord: 1, wordCount: 1 },
            { span: span(8, 11), text: "zum", firstWord: 2, wordCount: 2 },
            { span: span(12, 19), text: "Bahnhof", firstWord: 4, wordCount: 1 },
            { span: span(19, 20), text: ".", firstWord: 5, wordCount: 1 },
          ],
          words: [
            word(0, "Er", "er", "PRON", 1, "nsubj", { Case: "Nom", Person: "3" }),
            word(1, "geht", "gehen", "VERB", null, "root"),
            word(2, "zu", "zu", "ADP", 4, "case"),
            word(2, "dem", "der", "DET", 4, "det"),
            word(3, "Bahnhof", "Bahnhof", "NOUN", 1, "obl"),
            word(4, ".", ".", "PUNCT", 1, "punct"),
          ],
        }],
      },
    };
    const [view] = sentenceViews(report);
    expect(view!.syntax?.producer).toBe("stanza 1.15.0");
    expect(view!.syntax?.models).toBe("stanfordnlp/stanza-de@e7acb8e5 · de depparse=combined_charlm, tokenize=combined_nocharlm");
    const words = view!.syntax!.words;
    expect(words.map((item) => [item.form, item.token, item.upos, item.head, item.relation])).toEqual([
      ["Er", null, "PRON", "geht", "nsubj"],
      ["geht", null, "VERB", null, "root"],
      ["zu", "zum", "ADP", "Bahnhof", "case"],
      ["dem", "zum", "DET", "Bahnhof", "det"],
      ["Bahnhof", null, "NOUN", "geht", "obl"],
      [".", null, "PUNCT", "geht", "punct"],
    ]);
    expect(words[0]!.feats).toBe("Case=Nom|Person=3");
    expect(words[1]!.feats).toBe("_");
  });

  it("has no syntax section without a backend", () => {
    const [view] = sentenceViews(documentReport({ constructions: [], sentences: [] }));
    expect(view!.syntax).toBeNull();
  });
});
