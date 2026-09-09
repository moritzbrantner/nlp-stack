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
    description: "A substantial engineering release note with repeated concepts, named places, tradeoffs, and implementation boundaries.",
    demonstrates: "word profile, keywords, entities",
    focus: "word-corpus",
    text:
      "Alice opened the September release review in Berlin with a summary of the semantic search roadmap. The Rust text-analysis package now exposes document statistics, keywords, entities, linguistic evidence, embeddings, and semantic-map structure through one package surface. The browser still owns document decoding because PDF extraction and OCR are transport concerns rather than semantic-analysis responsibilities. Corpus statistics remain in the shared analysis layer so command-line tools, native applications, and the browser can compare the same evidence.\n\nBob asked whether semantic retrieval would replace exact lexical search. Alice said no: exact matching remains valuable when a user knows the wording or needs an auditable baseline. The semantic path instead targets paraphrases, conceptually related passages, and vocabulary mismatches that lexical ranking cannot recover well. The team evaluates both methods against the same queries and source passages, then records which method retrieved each relevant result. Engineers also keep model identity and vector dimensions with the output so an embedding change cannot masquerade as an equivalent run.\n\nThe release also changes how larger corpora are handled. Exact cosine similarity remains useful for small evaluations, while indexed search is measured separately once the corpus grows. Representative passages always retain their source identifiers, and derived concepts point back to the sentences that formed them. That provenance matters during review because editors need to understand why a concept was selected rather than accept an unexplained score. The next release will focus on model-backed semantic quality, better topic grouping, and realistic long-form examples instead of adding another parallel analysis surface.",
  },
  {
    id: "meeting-dialogue",
    label: "Product review meeting",
    category: "Conversation",
    description: "A longer design review that leaves and revisits retrieval quality, latency, evidence, and launch policy.",
    demonstrates: "semantic map, recurring topics, topic shifts",
    focus: "semantic-map",
    text:
      "Maya: We should launch the new search experience with semantic ranking enabled for a controlled group because paraphrased questions are still failing lexical retrieval. Jonas: I am not convinced that a better-looking demo proves retrieval quality. Exact lexical ranking is easy to audit, and our judged evaluation set is still too small. Lina: The latest evaluation contains support questions where users describe the right concept with completely different vocabulary. On those queries, the embedding model recovers passages that share almost no important words with the question. Maya: That is precisely the gap I want the semantic candidate to close. Jonas: Then we need the lexical baseline, the semantic candidate, and identical queries so the comparison measures retrieval rather than presentation.\n\nSam: Before we discuss the default, we also need to talk about latency. The browser model has to load before it can embed a new document, and older phones may wait noticeably longer than desktop machines. Maya: The model can be cached after the first download, so repeated analysis should be much faster. Sam: Cached weights help startup, but embedding thirty sentences still costs more than hashing thirty sentences. Lina: We should report model-loading time separately from per-document inference so the two costs are not confused. Jonas: Good, because a quality improvement is not useful if the interface feels broken on the machines our customers actually use.\n\nMaya: There is another requirement that matters as much as speed: every semantic concept must remain connected to its source sentences. Jonas: Yes, I do not want a topic label that cannot be traced back to the passages that caused it. Lina: The evaluation report should store the model name, dimensions, similarity scores, and representative passages. Sam: That also lets us tell whether a later model upgrade changed the neighborhood graph rather than blaming the clustering code. Maya: We can show the semantic map as evidence, not as an oracle.\n\nJonas: Returning to the original launch question, I would support the experiment if the measured retrieval gain survives a larger evaluation and the provenance remains visible. Maya: I agree; semantic ranking should not become the default merely because embeddings are fashionable. Lina: I will expand the judged queries with paraphrases and ambiguous wording, then rerun both retrieval paths. Sam: I will capture cold-load and warm-run performance on desktop and mobile. Jonas: Once those results are reproducible, we can change the default with evidence instead of intuition. Maya: Then the decision is settled: compare first, preserve the audit trail, and only promote the semantic path when both quality and runtime behavior justify it.",
  },
  {
    id: "climate-research-memo",
    label: "Climate adaptation memo",
    category: "Research memo",
    description: "A multi-theme policy memo with semantically related passages deliberately separated across the document.",
    demonstrates: "semantic map, concept recurrence, semantic neighborhoods",
    focus: "semantic-map",
    text:
      "The city climate office compared five neighborhoods during the July heat wave. Street-level sensors showed that blocks with mature tree canopy were several degrees cooler in late afternoon than nearby blocks dominated by asphalt and exposed parking. Residents described shaded sidewalks as usable even when unshaded plazas felt intolerable. The forestry team therefore proposed protecting mature trees before expanding ornamental planting, because large existing crowns provide cooling that new saplings cannot replace quickly.\n\nPublic-health staff focused on a different problem: nighttime heat remained high inside older apartments. Emergency visits rose most sharply in districts with many top-floor flats, limited ventilation, and low household income. Cooling centers helped during the hottest hours, but interviews showed that elderly residents were reluctant to travel across town after dark. Health officials recommended smaller neighborhood cooling rooms in libraries, clinics, and community halls rather than relying on one central facility.\n\nTransport planners initially treated the heat program as unrelated to mobility. Their passenger data showed otherwise. Bus stops without shelter produced the highest number of heat complaints, and long transfers forced riders to stand on exposed pavement. Adding shade structures at major transfer points reduced radiant heat even when air temperature barely changed. The planners also found that reliable evening service mattered because residents using public cooling spaces needed a safe way home after sunset.\n\nThe energy team examined apartment retrofits. Exterior shutters, roof insulation, and night ventilation reduced indoor temperature without requiring continuous air conditioning. Heat pumps could provide efficient cooling, but rapid adoption would increase summer electricity demand if building envelopes remained poor. The team recommended pairing mechanical cooling with insulation and solar shading so the grid would not carry avoidable peak load. That recommendation echoed the health finding that the most dangerous heat exposure happens indoors, especially overnight.\n\nWhen the groups reconvened, they realized that several interventions served more than one goal. Protecting a large street tree cools pedestrians, shades bus passengers, and lowers solar gain on nearby façades. A neighborhood library used as a cooling room works better when evening buses still run and the walking route is shaded. Apartment retrofits reduce health risk while also limiting pressure on the electrical grid. The strongest proposals therefore connected public health, transport, buildings, and urban forestry instead of treating each department as an isolated program.\n\nThe final memo returned to measurement. Officials did not want success defined by the number of trees planted, cooling centers announced, or retrofit grants issued. They proposed tracking actual canopy survival, indoor nighttime temperature, access time to a cool public space, and heat-related medical incidents. Researchers would preserve sensor locations and sampling dates so year-to-year comparisons remained interpretable. The city council could then distinguish visible activity from interventions that measurably reduce heat exposure. The central conclusion was that heat adaptation is a network problem: shade, mobility, buildings, public space, and health reinforce one another, and the evaluation must capture those relationships.",
  },
  {
    id: "short-story",
    label: "The blue notebook",
    category: "Narrative",
    description: "A longer narrative with recurring motifs, separated memories, people, places, and a return to earlier meanings.",
    demonstrates: "summary, entities, semantic map, narrative recurrence",
    focus: "semantic-map",
    text:
      "Elena arrived in Freiburg just before the evening rain, carrying a blue notebook wrapped in brown paper. She had promised her brother Tomas that she would return it to the old library before closing time. At the station, a violinist played a slow melody beneath the departures board while commuters hurried toward the trams. The tune reminded Elena of summer visits to the Black Forest when their father used to whistle while identifying plants along the trail.\n\nShe crossed the square and passed the cathedral as the rain became heavier. Inside a café she opened the parcel for the first time. The notebook contained field notes from a botanist named Marta Weiss, who had catalogued orchids and wetland plants forty years earlier. Several pages described a rare orchid growing near a spring above the Dreisam valley, but the final location was written only as a sketch of a footpath and three old beech trees. Elena understood why Tomas had been reluctant to give the notebook away: their father had copied that same sketch into one of his own hiking journals.\n\nThe library doors were still open when she arrived. Herr Bauer, the evening librarian, recognized the notebook immediately and said it had disappeared during an archive move many years ago. He showed Elena a catalogue card bearing Marta Weiss's name and a note that her field collection had never been completely indexed. Elena asked whether the orchid site had ever been found again. Herr Bauer said a university survey had searched for it twice, but the landscape had changed and the old path names no longer matched modern maps.\n\nFor a moment Elena considered photographing every page before returning the book. Instead she photographed only the catalogue card and the sketch her father had copied, then handed the original to the librarian. Herr Bauer promised that the notebook would be digitized with its provenance rather than separated into anonymous images. That mattered to Elena because the handwriting, dates, and sequence of observations told a story that isolated photographs would lose.\n\nWhen she stepped outside, the rain had stopped. The violinist at the station was playing the same melody as before, and Elena finally remembered its name. She called Tomas and told him that returning the notebook had not erased their connection to their father; it had placed that memory where other people could follow it. Tomas was silent for a while, then asked whether they could hike the old spring path together in October. Elena looked at the photograph of the three beech trees and said yes. The notebook was now behind library glass, but the unfinished search it contained had become theirs.",
  },
  {
    id: "multilingual-note",
    label: "Multilingual project note",
    category: "Mixed language",
    description: "English, German, and Spanish appear in one source with Unicode punctuation, accents, and repeated cross-language concepts.",
    demonstrates: "script profile, tokenization, linguistics",
    focus: "linguistics",
    text:
      "Project note: The browser analysis should keep source evidence attached to every derived result. In Stuttgart besprechen wir morgen die nächsten Schritte für die Suche, die Dokumentanalyse und die Qualität der Beispiele. Die Testdokumente sollen länger werden, damit wiederkehrende Themen und echte Themenwechsel sichtbar sind. Lucía añade: «La evidencia debe permanecer vinculada a la fuente, especialmente cuando resumimos documentos largos o agrupamos ideas parecidas.» También necesitamos distinguir una traducción real de dos frases que solamente comparten algunas palabras. Danach vergleichen wir lexical search, semantische Nachbarschaften und die Qualität der extrahierten Begriffe. The final decision should be understandable across languages: model-derived similarities may assist discovery, but the original passage remains the evidence a reviewer can inspect.",
  },
  {
    id: "ocr-like-report",
    label: "OCR-like field report",
    category: "Noisy document",
    description: "A longer imperfect extraction with line breaks, spacing errors, numbers, duplicated text, and damaged timestamps.",
    demonstrates: "word profile, document facts, robustness",
    focus: "overview",
    text:
      "FIELD REPORT 17 / BERLIN\n\nCollected: 2026-08-14   Operator: A. Keller\nStation: NORTH PLAT-FORM / SENSOR ARRAY B\n\nThe north plat-form sensor recorded  42 events between 18:00 and 22:00. Several entries were dupli-\ncated after the power interruption at approximately 19:4?. The recovery process restored the local index, but two timestamps remained incomplete and one device identifier was split across lines.\n\nEVENT 031  door-open     19:31:12\nEVENT 032  motion        19:33:08\nEVENT 032  motion        19:33:08   [duplicate?]\nEVENT 033  temperature   19:4?:51\nDEVICE B-17-\n04 resumed transmission after restart.\n\nRecommendation: compare recovered records against the source log and preserve the original identifiers before correcting display text. Do not infer missing times from neighboring rows. Duplicate detection may flag EVENT 032, but the raw record must remain available for review.\n\nStatus: usable with review. Confidence is high for the event count, moderate for device continuity, and lower for the damaged timestamp sequence. A second export should be compared before any irreversible cleanup.",
  },
];

export const defaultTextAnalysisExample = textAnalysisExamples[0]!;
