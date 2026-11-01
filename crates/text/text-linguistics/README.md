# text-linguistics

Heuristic-first, local-first linguistic analysis pipeline for
`moritzbrantner-video-analysis`.

## Highlights

- Per-sentence language detection (lingua) and routing, with typed supported/partial/unsupported/unknown reports per operation
- Tokenizer routing for word, subword, and mixed analysis modes
- Surface-to-subword alignment
- Lemmatization, morphology, POS tagging, chunking, and dependency parsing
- Explicit local model-backed named entities through `CandleTokenClassifier`
- Heuristic rule extraction remains available through `LinguisticAnalysisOptions::heuristic()`
- Coreference, events, discourse, topics, and style analysis
- Deterministic historical temporal normalization (`normalize_temporal_expressions`): typed dates, ranges, decades and centuries with byte spans, English and German

## Stable contract

The stable surface is the deterministic analysis pipeline, its request options,
serializable analysis records, tokenizer policy types, and transcript-contract
adapters behind the `transcripts` feature. Default constructors use heuristic
entity recognition, avoid model-bundle tokenizer alignment, and do not create or
download model bundles.

## Quality and limits

Language, POS, entity, coreference, event, discourse, topic, and style outputs
are heuristic-first and best-effort. Their data shapes are intended to remain
stable; their labels and confidence scores should not be treated as
production-grade NLP accuracy claims.

## Example

```rust,no_run
use text_linguistics::{TextNlpPipeline, TextNlpConfig, LinguisticAnalysisOptions};

let pipeline = TextNlpPipeline::new(TextNlpConfig {
    options: LinguisticAnalysisOptions::heuristic(),
    ..TextNlpConfig::fast()
});
let analysis = pipeline
    .analyze_text("Alice visited Berlin and presented the roadmap.")
    .unwrap();

assert!(!analysis.entities.is_empty());
assert_eq!(analysis.graph.tokens.len(), analysis.tokens.len());
```

When the `candle` feature is enabled, callers can use a local
token-classification NER model by setting
`EntityRecognitionOptions::local_model()` or
`EntityRecognitionOptions::local_model_with_downloads()`. The no-download
constructor expects a local bundle to already exist; the downloads constructor
is the explicit opt-in path for materializing a missing bundle. No OpenAI,
Claude, or hosted LLM token is required. Transcript-specific analysis is
available behind the `transcripts` feature. Use
`TextNlpConfig::rich_with_model_backends()` when tokenizer/model-backed rich
analysis is explicitly desired.

`CandleTokenClassifier::from_bundle` (or `from_bundle_on` for an explicit device)
loads the weights and tokenizer once; every later `classify` call reuses them,
and clones share them across threads. `classify_batch` pads up to 16 texts into
one forward pass: labels and offsets match per-text `classify`, scores can differ
in the last float bit.

## Package surface

- Primary workflow: `linguistics.analyze` runs the deterministic linguistic
  pipeline and returns tokens, language, entities, topics, and style signals.
- Workflow operations: `linguistics.analyze`, `linguistics.entities`, and
  `linguistics.language`.
- Debug operations: `describe` inspects package metadata and operation support.
- Runtime support: deterministic package-surface operations are pure Rust and
  available through library, CLI, server, and WASM wrappers; richer model-backed
  library APIs remain feature-gated.
- Sample output includes `title`, `message`, `summary`, `result`, and
  operation-specific fields such as `tokens`, `language`, `entities`,
  `relations`, or `events`.
- The package surface does not download models or execute hosted LLMs.

## Model Loading

The deterministic heuristic pipeline is always available. Local NER runs a
Candle token classifier from a bundle under `EntityRecognitionOptions::bundle_dir`
(default `.model-runtime`) when the crate is built with `candle,model-bundles`.
`ModelPreset` selects the model:

| Preset | Model (pinned) | Labels | Languages |
| --- | --- | --- | --- |
| `DistilbertMultilingualNerHrl` (default) | `Davlan/distilbert-base-multilingual-cased-ner-hrl` @ `d421f57d` (AFL-3.0) | PER, ORG, LOC, DATE | en, de, es |
| `BertBaseNer` | `dslim/bert-base-NER` @ `main` (MIT) | PER, ORG, LOC, MISC | en |

`ModelPreset::supports_language` reports whether a preset is evaluated for a
language. The multilingual preset became the default because it matches
`bert-base-ner` on the English gold fixtures and is far better on German and
Spanish. Exact-span micro F1 on PER/ORG/LOC over
`evaluation/corpora/named-entities-multilingual-v1.jsonl` (36 hand-written
sentences, 12 per language), measured 2026-11-01:

| Preset | en | de | es |
| --- | --- | --- | --- |
| `DistilbertMultilingualNerHrl` | 0.971 | 0.985 | 0.969 |
| `BertBaseNer` | 0.971 | 0.761 (unsupported) | 0.710 (unsupported) |

```bash
TEXT_LINGUISTICS_MODEL_DIR=.model-runtime cargo test -p moenarch-text-linguistics \
  --features external-tests --test model_loading_external --release -- --ignored --nocapture
```

The test downloads missing bundles from Hugging Face and fails when a preset
scores below 0.8 F1 for a language it declares.

## UD morphology (en/de/es)

With `candle,model-bundles`, setting
`LinguisticAnalysisOptions::morphology` to `MorphologyOptions::local_model()` (or
`local_model_with_downloads()`) replaces the heuristic lemma, POS and morphology
layers for English, German and Spanish. The layers come from:

| Preset | Model (pinned) | Training data |
| --- | --- | --- |
| `UdModelPreset::ComboEnglishEwt` | `clarin-pl/combo-nlp-xlm-roberta-base-english-ewt-ud2.17` @ `2c0ea96a` | UD_English-EWT 2.17 (CC BY-SA 4.0) |
| `UdModelPreset::ComboGermanGsd` | `clarin-pl/combo-nlp-xlm-roberta-base-german-gsd-ud2.17` @ `55234be1` | UD_German-GSD 2.17 (CC BY-SA 4.0) |
| `UdModelPreset::ComboSpanishAncora` | `clarin-pl/combo-nlp-xlm-roberta-base-spanish-ancora-ud2.17` @ `3f5885ee` | UD_Spanish-AnCora 2.17 (CC BY 4.0) |

`UdModel` runs the checkpoint's XLM-RoBERTa encoder and morphological head
in Candle. It uses first-subword pooling, per-category UD feature heads with
COMBO's refinement layer, and COMBO's nearest-valid-feature constraint per UPOS.
The tokenizer is `FacebookAI/xlm-roberta-base` @ `e73636d4` (MIT). The same
pass also runs the checkpoint's biaffine dependency head (see below). Lemmas come
from `UdLemmatizer`, not from the model. It applies suffix edit rules per UPOS
plus an exception list, both built from the treebank training splits by
`scripts/build_ud_lemma_tables.py` (`data/lemmas/*.tsv`; see `docs/PROVENANCE.md`
for licences). Contractions (`zum`, `im`, `del`, `al`) are tagged as two
syntactic words. The token keeps the adposition's UPOS and lemma and the
article's features. English clitics are split as EWT splits them (`don't` ->
`do n't`, `can't` -> `ca n't`, `John's` -> `John 's`, `I'm`, `we'll`, `cannot`);
the token keeps the host word's UPOS, lemma and features (`can't`: AUX `can`).
Sentences in other languages report the layers as unsupported.

Scores, measured 2026-11-01. The UD test splits use gold word segmentation, and
lemma accuracy uses the predicted UPOS. The fixtures are
`evaluation/corpora/morphology-multilingual-v1.jsonl`, run through the full
pipeline with each sentence routed by its detected language:

| | UPOS | UFeats | Lemma | COMBO published UPOS / UFeats / Lemma |
| --- | --- | --- | --- | --- |
| en, UD_English-EWT test (25,094 words) | 0.981 | 0.978 | 0.976 | 0.981 / 0.982 / 0.982 |
| de, UD_German-GSD test (16,499 words) | 0.973 | 0.920 | 0.972 | |
| es, UD_Spanish-AnCora test (53,622 words) | 0.991 | 0.985 | 0.992 | |
| en, gold fixtures (83 tokens) | 1.000 | 1.000 | 1.000 | |
| de, gold fixtures (87 tokens) | 1.000 | 1.000 | 0.989 | |
| es, gold fixtures (83 tokens) | 0.988 | 0.988 | 1.000 | |

The English published numbers are the EWT card's "aligned accuracy" row. COMBO's
lemmas come from its own lemmatizer head, which this crate does not run; the
UD-derived table is 0.6 points behind it.

```bash
UD_TREEBANK_DIR=path/to/ud-2.17-test-splits TEXT_LINGUISTICS_MODEL_DIR=.model-runtime \
  cargo test -p moenarch-text-linguistics --features external-tests \
  --test ud_model_external --release -- --ignored --nocapture --test-threads 1
```

`UD_TREEBANK_DIR` must hold `en_ewt-ud-test.conllu`, `de_gsd-ud-test.conllu`
and `es_ancora-ud-test.conllu` from the UD 2.17 releases. The test downloads
about 3.4 GB of checkpoints. On the UD test splits it fails below 0.95 UPOS, 0.85
UFeats or 0.90 lemma accuracy. On the fixtures it fails below 0.90 UPOS, 0.75
UFeats or 0.90 lemma accuracy.

## UD dependency syntax (en/de/es)

Setting `LinguisticAnalysisOptions::dependency_parsing` to
`DependencyParsingOptions::local_model()` (or `local_model_with_downloads()`)
replaces the heuristic dependency trees for English, German and Spanish with the
checkpoint's biaffine dependency head. The head uses the same presets and the
same `UdModel` as morphology. When both layers select the same model and bundle
directory, each sentence is encoded once.

The head projects each word, plus a learned root embedding, through
`Linear -> LayerNorm -> ReLU` arc and relation MLPs. Biaffine scorers then give
an arc score for every head/dependent pair and a relation score for each chosen
arc. `decode_dependency_tree` runs Chu-Liu-Edmonds to find the maximum spanning
tree. If several words attach to the root, the one with the best root score keeps
it and the others attach to that word, as COMBO's decoder does. The result is
always one tree with one root. Root arcs are labelled `root`, and no other arc is.
Relations are the full UD v2 set (`DependencyRelation`) plus the treebank's
subtype (`DependencyNode::subtype`, for example `pass` in `nsubj:pass`). A
contraction token (`zum`, `del`, `John's`) takes the attachment of its word
nearest the root (the adposition, the possessor), or of its first word when both
attach to the same head (`doesn't`: `aux`), so the token-level result is still a
tree.

Scores, measured 2026-11-01. LAS compares universal relations and ignores
subtypes, as the CoNLL 2018 scorer does. The UD test splits use gold word
segmentation. COMBO's published scores are from the model cards: "aligned
accuracy" (English, German) and full-text F1 (Spanish, whose card has no aligned
row). The cards parse combo-seg's predicted sentences; with gold sentences the
English EWT web text parses noticeably better, hence the higher UAS/LAS here.

| | UAS | LAS | LAS with subtypes | COMBO published UAS / LAS |
| --- | --- | --- | --- | --- |
| en, UD_English-EWT test (25,094 words) | 0.951 | 0.934 | 0.933 | 0.931 / 0.914 |
| de, UD_German-GSD test (16,499 words) | 0.897 | 0.868 | 0.865 | 0.890 / 0.861 |
| es, UD_Spanish-AnCora test (53,622 words) | 0.943 | 0.927 | 0.924 | 0.940 / 0.925 |
| en, gold fixtures (83 tokens) | 1.000 | 1.000 | | |
| de, gold fixtures (87 tokens) | 1.000 | 1.000 | | |
| es, gold fixtures (83 tokens) | 0.988 | 0.976 | | |

The fixtures are `evaluation/corpora/dependencies-multilingual-v1.jsonl`, run
through the full pipeline with each sentence routed by its detected language. The same opt-in test as
above reports these scores. On the UD test splits it fails below 0.85 UAS or 0.82
LAS, and on the fixtures below 0.90 UAS or 0.85 LAS. It also checks that every
predicted tree is well formed. Parsing adds little to the encoder cost:
relation scores are computed only for the decoded arcs.

## Language detection and per-sentence routing

`LanguageDetector` runs [lingua](https://github.com/pemistahl/lingua-rs) 1.8
(Apache-2.0) in high-accuracy mode over English, German, Spanish, French,
Italian, Portuguese and Dutch (`DETECTABLE_LANGUAGES`). The four extra languages
are there so their text is not forced into en/de/es. Each sentence gets a
`LanguageDecision`:

- `Detected`: the best candidate leads the runner-up by at least
  `LanguageDetectionOptions::min_margin` (0.25) in lingua's relative confidence;
- `Ambiguous`: no candidate leads clearly (the candidates are still reported);
- `TooShort`: fewer than two words (`Thanks.`);
- `UnsupportedScript`: a script none of the languages use (Cyrillic, Han, ...).

The analysis pipeline then routes every sentence by its own language.
`LinguisticAnalysis::sentence_analyses` has one `SentenceAnalysis` per sentence:
its UTF-8 span, the language it was analyzed in, the detection record, and an
`OperationReport` for each of the seven core operations. A report gives the
support level, the producer that ran and the reason:

| Support | Meaning | Results in the analysis |
| --- | --- | --- |
| `Supported` | a producer evaluated for this language, or language-neutral rules | kept |
| `Partial` | ran, but unscored for this language (English heuristics, capitalization NER) | kept, labelled |
| `Unsupported` | the language is known but nothing configured supports it | omitted for this sentence |
| `Unknown` | the language is undecided, so nothing language-specific ran | omitted for this sentence |

German sentences go to the German UD model and Spanish sentences to the Spanish
one within the same document, and no relation is drawn across sentences. Lemmas,
morphology, POS, dependency trees and entities are omitted for sentences whose
report does not keep them. Chunks, events, topics and style still see every
sentence. `TextProcessingOptions::language` pins one language for every sentence
and skips detection.

Accuracy, measured 2026-11-01 (`never_misroutes_a_gold_fixture_sentence` in
`src/tests.rs`, and the opt-in `tests/language_detection_ud.rs` over the UD 2.17
test splits, one UD sentence per unit):

| Sample | Sentences | Decided | Decided correctly |
| --- | --- | --- | --- |
| en/de/es gold fixtures (NER, morphology, dependencies, sentence languages) | 122 | 120 | 120 (0 misrouted) |
| en, UD_English-EWT test | 2,077 | 77.5% | 99.4% |
| de, UD_German-GSD test | 977 | 96.7% | 99.7% |
| es, UD_Spanish-AnCora test | 1,721 | 97.2% | 100% |
| fr / it / pt, UD GSD / ISDT / Bosque test | 416 / 482 / 1,167 | 97.1% / 95.9% / 94.2% | 100% / 99.6% / 99.5% |

The two undecided fixture sentences are mostly names ("Angela Merkel met
Emmanuel Macron in Paris."). EWT has many short web fragments, which come back
undecided rather than misrouted. The stopword detector this replaces got 3 of the
10 German and 3 of the 10 Spanish morphology fixture sentences wrong. Before
choosing lingua, `whatlang` 0.18 (MIT) and `whichlang` 0.1 (MIT) were scored on the
same sentences. Both are much smaller (about 0.3 MB) but less accurate on short
sentences, and `whichlang` gives no confidence. lingua's models add about 31 MB
to every binary that links this crate, including the browser Wasm.

## Language support (en/de/es)

Issue #117 asks for the same seven operations in English, German and Spanish.
Current state:

| Operation | en | de | es |
| --- | --- | --- | --- |
| Language detection | **lingua, scored** | **lingua, scored** | **lingua, scored** |
| Sentence segmentation | language-neutral rules | language-neutral rules | language-neutral rules |
| Tokenization | language-neutral rules | language-neutral rules | language-neutral rules |
| Named entities (local model) | **model, gold-scored** | **model, gold-scored** | **model, gold-scored** |
| Lemma / morphology (`MorphologyMode::LocalModel`) | **UD model + UD lemma table, scored** | **UD model + UD lemma table, scored** | **UD model + UD lemma table, scored** |
| Part of speech (`MorphologyMode::LocalModel`) | **UD model, scored** | **UD model, scored** | **UD model, scored** |
| Dependency syntax (`DependencyParsingMode::LocalModel`) | **UD model, scored** | **UD model, scored** | **UD model, scored** |

The `LocalModel` modes route every en/de/es sentence to its language's UD model.
The default heuristic modes stay English rules: English sentences get them as
`Partial` (unscored), and German and Spanish sentences report those three layers
as `Unsupported` instead of receiving English heuristic output.

The browser does not run the UD models. The COMBO checkpoints are custom heads on
XLM-RoBERTa base, which Transformers.js cannot load as-is, and an int8 ONNX
export would be roughly 280 MB per language, about twice the browser NER model.
The Pages workbench's Sentences view shows those layers as "Rust only" for German
and Spanish sentences; English sentences there still get the `Partial` heuristics. In the browser it runs
language detection, segmentation and tokenization in Rust/Wasm and NER with the
browser model, per sentence language.

Heuristic NER (the default `EntityRecognitionMode::Heuristic`) is
capitalization-based and is not language-aware; it reports `Partial` for every
sentence.

Browser package apps expose local benchmark scenarios in the `Benchmarks` tab.
Root scripts `bun run text-native:bench` and `bun run text-wasm:bench:all`
exercise native and browser paths respectively.

## Related crates

- `text-core`
- `text-transcripts` for SRT, WebVTT, Whisper JSON, and plain transcript parsing
- `text-lexical` for TF-IDF and BM25 corpus indexing from text inputs
- `media-core`
