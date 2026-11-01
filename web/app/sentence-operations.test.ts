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
        sentences: [{ sentenceIndex: 0, support: "unsupported", producer: null, reason: "dependency trees come from heuristic:english-rules", constructionCount: 0 }],
      }),
    );
    expect(view!.reporting).toEqual({
      status: "unsupported",
      producer: null,
      detail: "dependency trees come from heuristic:english-rules",
      roots: [],
    });
  });
});
