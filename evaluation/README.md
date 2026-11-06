# NLP evaluation

This directory owns repository-level quality evaluation for `nlp-stack`. It is development and evidence infrastructure, not a runtime package and not a publication surface.

## Contract

Evaluation inputs are versioned JSONL corpora under `evaluation/corpora/`. Predictions use task-specific JSONL records with the same case IDs. Reports use `schemaVersion: 1` and record the suite, system, aggregate metrics, and per-case evidence when useful.

The initial sentence-boundary suite uses UTF-8 half-open byte ends so it evaluates the same span coordinate system exposed by `text-core`. The committed corpus is deliberately small: it is a deterministic smoke/regression corpus covering ordinary English, decimals and built-in abbreviations, caller-supplied citation abbreviations, internal-period abbreviations, Unicode case folding, and CJK terminators. It is not presented as a production-quality benchmark.

`evaluation.metrics` provides reusable deterministic metrics for later suites:

- precision / recall / F1 over exact events or spans;
- accuracy and macro F1 for classification-style tasks;
- reciprocal rank, MRR, Recall@k, and nDCG@k for retrieval;
- Spearman correlation for semantic similarity scores.

## Sentence-boundary baseline

Run the live `text-core` baseline with:

```bash
bash scripts/check-nlp-evaluation.sh
```

The script generates predictions through the public `text-core` sentence-boundary API, evaluates them against `sentence-boundaries-v1`, writes a report under `.artifacts/nlp-eval/`, and requires F1 to remain at least `NLP_EVAL_MIN_BOUNDARY_F1` (default `1.0`).

The committed prediction and aggregate report files under `evaluation/baselines/` are reproducible fixtures for the current baseline. The Python test suite verifies that the aggregate report is derivable from those committed inputs. The live script is the proof that current Rust behavior still produces predictions through the public capability seam.

## Semantic-analysis smoke suites

`semantic_similarity_smoke_v1.jsonl` and `topic_shift_smoke_v1.jsonl` establish the first versioned H7 fixtures. They are deliberately small contract/evaluator fixtures, not semantic-quality benchmarks.

The evaluator exposes two report-only commands:

```bash
python3 -m evaluation.runner semantic-similarity \
  --gold evaluation/corpora/semantic_similarity_smoke_v1.jsonl \
  --predictions path/to/predictions.jsonl \
  --suite semantic-similarity-smoke-v1 \
  --system your-system

python3 -m evaluation.runner topic-shifts \
  --gold evaluation/corpora/topic_shift_smoke_v1.jsonl \
  --predictions path/to/predictions.jsonl \
  --suite topic-shift-smoke-v1 \
  --system your-system
```

Semantic similarity reports Spearman rank correlation and mean absolute error. Topic shifts use exact index precision/recall/F1. No minimum semantic-quality threshold is enforced yet: hashed embeddings are useful deterministic interoperability fixtures, but they are not a meaningful quality target for semantic similarity or topic segmentation.

### Grouped and multilingual evidence

Semantic-similarity and topic-shift gold cases may include an optional non-empty `group` string. Group metadata is owned by the gold corpus: it does not participate in case identity, prediction matching, or score calculation. When at least one gold case is grouped, the report adds a deterministic `groups` section with per-group case counts and the same metrics used by the aggregate report. Ungrouped cases continue to contribute to aggregate metrics but are intentionally absent from the grouped breakdown.

`semantic_similarity_multilingual_v1.jsonl` adds small English, German, Spanish, and cross-language pairs. Its groups separate monolingual language pairs from English–German, English–Spanish, and German–Spanish comparisons. `topic_shift_multilingual_v1.jsonl` adds monolingual topic trajectories plus language-switch cases, including a language switch that keeps the same topic so systems can expose language sensitivity rather than silently treating every language change as a semantic topic boundary.

These fixtures are still calibration evidence, not production benchmarks. They make language-specific regressions visible without introducing a preferred embedding model or a minimum quality threshold. Larger labelled corpora and model-backed comparisons should be added with explicit provenance before semantic quality gates are considered.

## Named-entity gold fixtures

`named-entities-multilingual-v1.jsonl` holds 36 short, self-written sentences (12 each in English, German and Spanish) with hand-labelled PER/ORG/LOC spans as UTF-8 half-open byte offsets (`byteStart`/`byteEnd`) plus the span text. Articles are outside the span ("the United Nations" -> `United Nations`). The opt-in Rust test `crates/text/text-linguistics/tests/model_loading_external.rs` scores every local NER preset on it with exact-span micro F1 and fails below 0.8 for a language the preset declares. The browser NER test checks that the corpus spans are exact and that its languages match the browser module's declared languages. It is a small regression and model-selection fixture, not a production benchmark.

## Morphology gold fixtures

`morphology-multilingual-v1.jsonl` holds 30 short, self-written sentences (10 each in English, German and Spanish). Each token, as `text-core` splits it, has a hand-assigned UPOS, lemma and UD FEATS string. The annotation follows the conventions of UD_English-EWT, UD_German-GSD and UD_Spanish-AnCora, checked against how those treebanks' training splits annotate the same forms. One exception: lemmas use current spelling (`Fluss`, where GSD writes `Fluß`). Contractions (`im`, `al`) are one token: they get the adposition's UPOS and lemma and the article's features. English clitic contractions (`doesn't`, `can't`, `it's`, `John's`, `I'm`, `we'll`) are one token too and get the host word's UPOS, lemma and features (`can't`: AUX `can` `VerbForm=Fin`). The opt-in Rust test `crates/text/text-linguistics/tests/ud_model_external.rs` runs the pipeline with the UD morphology presets on these sentences, each routed by its detected language, and fails if a sentence is routed to the wrong language. It also interleaves the English, German and Spanish sentences in one document and checks that each still reaches its own model with the same scores. The same test also scores the presets on the UD 2.17 test splits, using gold word segmentation (`UD_TREEBANK_DIR`). Like the NER fixtures, this is a small regression fixture, not a production benchmark.

## Dependency gold fixtures

`dependencies-multilingual-v1.jsonl` holds the same 30 sentences as the morphology fixtures. Each token has a hand-assigned head (1-based token position, 0 for the root) and UD DEPREL. The trees follow UD_English-EWT, UD_German-GSD and UD_Spanish-AnCora conventions, checked against those treebanks. Bare German datives, Spanish `a`-datives and dative clitics are `obl:arg`, since neither treebank uses `iobj`. Bare English temporal nouns (`yesterday`, `next summer`) are `obl:unmarked`, as in EWT 2.17. A comma before a subordinate or coordinated clause, or after a fronted one, attaches to that clause's head. A contraction token (`im`, `al`) carries the adposition's attachment (`case`). An English clitic contraction carries the attachment of its word nearest the root: the possessor for `John's` (`nmod:poss`), the host where both words attach to the same head (`doesn't`: `aux`; `it's`, `we'll`: `nsubj`). The opt-in Rust test `crates/text/text-linguistics/tests/ud_model_external.rs` parses them through the full pipeline, routed by detected language, and reports UAS and LAS. LAS compares universal relations and ignores subtypes, as the CoNLL 2018 scorer does. The same test reports UAS/LAS on the UD 2.17 test splits. This is a small regression fixture, not a production benchmark.

## Sentence-language gold fixtures

`sentence-languages-multilingual-v1.jsonl` holds 12 short, self-written documents of two or three sentences each. Most mix English, German and Spanish; some pair a target language with French, Portuguese or Dutch, an undecidable one-word sentence (`Thanks.`) or a Cyrillic sentence. Each sentence has its text, its UTF-8 half-open byte span in the document (`byteStart`/`byteEnd`) and a hand-assigned ISO 639-1 language, or `null` where no language should be decided. The first two documents are the #116 samples. The Rust unit test `never_misroutes_a_gold_fixture_sentence` (`crates/text/text-linguistics/src/tests.rs`) checks that the sentence splitter reproduces the spans and that no sentence of this corpus or of the NER, morphology and dependency corpora is routed to the wrong language; a labelled sentence that comes back undecided is counted, not failed. The Pages Playwright journey `web/e2e/sentence-languages.spec.ts` uses the same kind of mixed document. It is a regression fixture, not a production benchmark.

## External gold sets and model-default gates

The hand-written fixtures above are regression fixtures. Model defaults are gated
on existing, licensed, human-labelled test sets instead. `evaluation/gold/manifest.json`
pins each one by URL at an exact upstream commit, plus SHA-256 and size, and records its
licence. The corpora are not committed. `python3 -m evaluation.gold fetch` downloads
them to `NLP_GOLD_DIR` (default `.test-corpora/gold`, ignored) and keeps a file only
if its digest matches.

| Corpus | Task | Lang | Size | Licence |
| --- | --- | --- | --- | --- |
| UD_English-EWT r2.17 test | syntax | en | 2,077 sentences, 25,094 words | CC BY-SA 4.0 |
| UD_German-GSD r2.17 test | syntax | de | 977 sentences, 16,499 words | CC BY-SA 4.0 |
| UD_Spanish-AnCora r2.17 test | syntax | es | 1,721 sentences, 53,622 words | CC BY 4.0 |
| Europarl NER test (ixa-ehu, Agerri et al. 2018) | entities | en, de, es | 799 parallel sentences each; ~520 PER/ORG/LOC each | Europarl terms ("no known copyright restrictions") |
| UNER_English-EWT test (Universal NER v1) | entities | en | 2,077 sentences, 1,088 entities | CC BY-SA 4.0 |
| UNER_German-PUD (Universal NER v1) | entities | de | 1,000 sentences, 1,039 entities | CC BY-SA 3.0 |
| XQuAD | retrieval | en, de, es | 240 parallel paragraphs, 1,190 questions each | CC BY-SA 4.0 |

Excluded on licence grounds: CoNLL-2002 Spanish (research use within the shared
task only, no commercial use), CoNLL-2003 (Reuters/Frankfurter Rundschau text
not redistributable), WikiANN (no stated licence and silver labels), WikiNEuRal and
MultiNERD (non-commercial), and STS-b multilingual (mixed upstream licences).
Universal NER has no Spanish set, so Spanish NER rests on Europarl alone.
`evaluation/gold/fixtures/` holds a few unmodified sentences of these corpora for
the offline unit tests (`scripts/test_gold_evaluation.py`); their sources and
licences are listed there.

### Harness

`python3 -m evaluation.gold export --out DIR` writes the raw-text inputs a backend
adapter needs (`{id, language, text}`, no labels). The adapter writes predictions
and a `{language: {backend, model, revision, details}}` systems file; the scorers
then write a score report that records the task, language, every corpus's id,
upstream revision and SHA-256, the system's model id and revision, the metrics and
the measurement date:

- `score-syntax` takes CoNLL-U predictions. Words are aligned to the gold words by
  character spans and, inside multi-word tokens, by form, as the CoNLL 2018 scorer
  does. It reports F1 for tokens, words, UPOS, UFeats (universal features only),
  lemma, UAS and LAS (universal relation, subtypes ignored), plus accuracy over
  aligned words. Inputs are gold sentences; the system tokenizes them itself.
- `score-entities` takes `{id, entities: [{byteStart, byteEnd, label}]}` lines and
  reports exact span+type micro F1 and per-type scores over PER/ORG/LOC. Gold MISC
  and system labels outside those three are ignored on both sides. Europarl text is
  the corpus tokens joined by single spaces (the corpus has no untokenized text);
  UNER uses the original sentence text.
- `score-retrieval` takes `{id, ranking: [passage ids]}` lines for queries of one
  language against the passages of another (or the same) language, and reports
  MRR@10, Recall@1, Recall@10 and nDCG@10. XQuAD paragraphs are parallel, so all
  nine en/de/es pairs share the same relevance judgements.
- `table REPORT ...` prints the reports as a per-language Markdown table.
- `gates` checks `evaluation/gold/gates.json`.

Adapters, all opt-in and local (CI stays offline):

- Stanza oracle (#190): `python3 -m evaluation.backends.stanza_sidecar --corpus ID ... --out DIR`
  against a running `sidecars/stanza/serve` (`NLP_STANZA_URL`). It also writes
  `stanza.systems.json` from the sidecar's `/v1/health` pins.
- NER and embedding presets: not yet wired; they need opt-in Rust writers that read
  the exported inputs and write the prediction shapes above.

### Gates

`gates.json` lists, per capability and backend, the pinned model per language,
the threshold status (`proposed` until the owner agrees, then `agreed`) and one
requirement per language, corpus and metric. Each requirement names the committed
report in `evaluation/gold/reports/` that holds its measurement. `python3 -m
evaluation.gold gates` fails unless every report exists, was measured on the
currently pinned corpus revision with the gated model and revision, and meets its
threshold. A model may become a default only through a passing gate. Thresholds
are proposed from the first measurements, a little below them, never invented
beforehand.

## Adding a capability suite

Keep each suite independently runnable and independently interpretable. Add a versioned corpus, a prediction shape, a deterministic evaluator using the reusable metrics where possible, a baseline report that states which system produced it, and the narrowest adapter needed to obtain predictions through the capability's stable public seam.

Do not mix model downloads, training, or hosted inference into the deterministic smoke suites. Larger external datasets and model-backed evaluations may be opt-in, but their provenance and setup requirements must be explicit.
