export type ExampleResultView =
  | "word-corpus"
  | "semantic-corpus"
  | "overview"
  | "keywords"
  | "linguistics"
  | "semantic-map";

export interface TextAnalysisExample {
  id: string;
  label: string;
  category: string;
  description: string;
  demonstrates: string;
  focus: ExampleResultView;
  text: string;
}

export const textAnalysisExamples: TextAnalysisExample[] = [
  {
    id: "technical-release",
    label: "Technical release",
    category: "Technical prose",
    description: "A compact engineering update with repeated domain vocabulary and named places.",
    demonstrates: "word profile, keywords, entities",
    focus: "word-corpus",
    text:
      "Alice presented the semantic search roadmap in Berlin during the release review. The Rust text-analysis package extracts keywords, entities, linguistic evidence, and deterministic semantic structure. The team moved corpus statistics into the shared analysis surface and kept browser decoding outside the semantic core. Bob asked how retrieval scales to larger corpora. Alice explained that exact similarity remains the deterministic baseline while indexed search is measured separately. The release keeps source provenance attached to representative passages so editors can inspect why a concept was selected.",
  },
  {
    id: "meeting-dialogue",
    label: "Meeting dialogue",
    category: "Conversation",
    description: "Alternating speakers revisit concepts, disagree, and converge on a decision.",
    demonstrates: "semantic map, topic shifts",
    focus: "semantic-map",
    text:
      "Maya: We should launch the new search experience with semantic ranking enabled by default.\n\nJonas: I disagree. The exact lexical ranking is easier to audit, and our current evaluation set is still small.\n\nMaya: The semantic model finds related passages that lexical search misses. We can keep the exact score visible as evidence.\n\nJonas: Then the decision should depend on measurable retrieval quality, not novelty. We need a baseline, a semantic candidate, and the same queries for both.\n\nMaya: Agreed. Let us ship the comparison first, preserve source passages, and only change the default after the evaluation is reproducible.\n\nJonas: That works. The audit trail is the important part.",
  },
  {
    id: "short-story",
    label: "Short story",
    category: "Narrative",
    description: "A small narrative with people, places, chronology, and recurring motifs.",
    demonstrates: "summary, entities, semantic map",
    focus: "semantic-map",
    text:
      "Elena arrived in Freiburg just before the evening rain. She had promised her brother Tomas that she would return the blue notebook to the old library before closing time. At the station, a violinist played beneath the departures board while commuters hurried toward the trams. Elena crossed the square, passed the cathedral, and found the library doors still open. The librarian recognized the notebook immediately: it contained field notes from a botanist who had worked in the Black Forest forty years earlier. Elena stayed to read one page about a rare orchid, then handed the notebook over. On the walk back to the station, the rain had stopped, and the violinist was playing the same melody again.",
  },
  {
    id: "multilingual-note",
    label: "Multilingual note",
    category: "Mixed language",
    description: "English, German, and Spanish appear in one source with Unicode punctuation and accents.",
    demonstrates: "script profile, tokenization, linguistics",
    focus: "linguistics",
    text:
      "Project note: The browser analysis stays local and deterministic. In Stuttgart besprechen wir morgen die nächsten Schritte für die Suche und die Dokumentanalyse. Lucía añade: «La evidencia debe permanecer vinculada a la fuente, especialmente cuando resumimos documentos largos.» Danach vergleichen wir lexical search, semantische Nachbarschaften und die Qualität der extrahierten Begriffe. The final decision should remain reproducible across languages and runtimes.",
  },
  {
    id: "ocr-like-report",
    label: "OCR-like report",
    category: "Noisy document",
    description: "Line breaks, spacing errors, numbers, and split words resemble imperfect document extraction.",
    demonstrates: "word profile, document facts, robustness",
    focus: "overview",
    text:
      "FIELD REPORT 17 / BERLIN\n\nCollected: 2026-08-14   Operator: A. Keller\n\nThe north plat-form sensor recorded  42 events.\nSeveral entries were dupli-\ncated after the power interruption.  The recovery process restored the local index, but two timestamps remained incomplete.\n\nRecommendation: compare the recovered records against the source log; preserve the original identifiers; do not infer missing times.\n\nStatus: usable with review. Confidence is high for event counts and lower for the damaged timestamp sequence.",
  },
];

export const defaultTextAnalysisExample = textAnalysisExamples[0]!;
