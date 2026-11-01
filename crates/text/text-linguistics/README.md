# text-linguistics

Heuristic-first, local-first linguistic analysis pipeline for
`moritzbrantner-video-analysis`.

## Highlights

- Language detection with script-aware fallbacks
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

## UD morphology (de/es)

With `candle,model-bundles`, setting
`LinguisticAnalysisOptions::morphology` to `MorphologyOptions::local_model()` (or
`local_model_with_downloads()`) replaces the heuristic lemma, POS and morphology
layers for German and Spanish. The layers come from:

| Preset | Model (pinned) | Training data |
| --- | --- | --- |
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
article's features. Other languages, English included, keep the heuristics.

Scores, measured 2026-11-01. The UD test splits use gold word segmentation, and
lemma accuracy uses the predicted UPOS. The fixtures are
`evaluation/corpora/morphology-multilingual-v1.jsonl`, run through the full
pipeline with the language pinned:

| | UPOS | UFeats | Lemma |
| --- | --- | --- | --- |
| de, UD_German-GSD test (16,499 words) | 0.973 | 0.920 | 0.972 |
| es, UD_Spanish-AnCora test (53,622 words) | 0.991 | 0.985 | 0.992 |
| de, gold fixtures (87 tokens) | 1.000 | 1.000 | 0.989 |
| es, gold fixtures (83 tokens) | 0.988 | 0.988 | 1.000 |

```bash
UD_TREEBANK_DIR=path/to/ud-2.17-test-splits TEXT_LINGUISTICS_MODEL_DIR=.model-runtime \
  cargo test -p moenarch-text-linguistics --features external-tests \
  --test ud_model_external --release -- --ignored --nocapture --test-threads 1
```

`UD_TREEBANK_DIR` must hold `de_gsd-ud-test.conllu` and
`es_ancora-ud-test.conllu` from the UD 2.17 releases. The test downloads about
2.3 GB of checkpoints. On the UD test splits it fails below 0.95 UPOS, 0.85
UFeats or 0.90 lemma accuracy. On the fixtures it fails below 0.90 UPOS, 0.75
UFeats or 0.90 lemma accuracy.

## UD dependency syntax (de/es)

Setting `LinguisticAnalysisOptions::dependency_parsing` to
`DependencyParsingOptions::local_model()` (or `local_model_with_downloads()`)
replaces the heuristic dependency trees for German and Spanish with the
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
contraction token (`zum`, `del`) takes the attachment of its word nearest the
root (the adposition), so the token-level result is still a tree.

Scores, measured 2026-11-01. LAS compares universal relations and ignores
subtypes, as the CoNLL 2018 scorer does. The UD test splits use gold word
segmentation. COMBO's published scores are from the model cards: "aligned
accuracy" (German) and full-text F1 (Spanish, whose card has no aligned row).

| | UAS | LAS | LAS with subtypes | COMBO published UAS / LAS |
| --- | --- | --- | --- | --- |
| de, UD_German-GSD test (16,499 words) | 0.897 | 0.868 | 0.865 | 0.890 / 0.861 |
| es, UD_Spanish-AnCora test (53,622 words) | 0.943 | 0.927 | 0.924 | 0.940 / 0.925 |
| de, gold fixtures (87 tokens) | 1.000 | 1.000 | | |
| es, gold fixtures (83 tokens) | 0.988 | 0.976 | | |

The fixtures are `evaluation/corpora/dependencies-multilingual-v1.jsonl`, run
through the full pipeline with the language pinned. The same opt-in test as
above reports these scores. On the UD test splits it fails below 0.85 UAS or 0.82
LAS, and on the fixtures below 0.90 UAS or 0.85 LAS. It also checks that every
predicted tree is well formed. Parsing adds little to the encoder cost:
relation scores are computed only for the decoded arcs.

## Language support (en/de/es)

Issue #117 asks for the same seven operations in English, German and Spanish.
Current state:

| Operation | en | de | es |
| --- | --- | --- | --- |
| Language detection | stopword-lexicon heuristic | stopword-lexicon heuristic | stopword-lexicon heuristic |
| Sentence segmentation | language-neutral rules | language-neutral rules | language-neutral rules |
| Tokenization | language-neutral rules | language-neutral rules | language-neutral rules |
| Named entities (local model) | **model, gold-scored** | **model, gold-scored** | **model, gold-scored** |
| Lemma / morphology (`MorphologyMode::LocalModel`) | English heuristics | **UD model + UD lemma table, scored** | **UD model + UD lemma table, scored** |
| Part of speech (`MorphologyMode::LocalModel`) | English heuristics | **UD model, scored** | **UD model, scored** |
| Dependency syntax (`DependencyParsingMode::LocalModel`) | English heuristics (unscored) | **UD model, scored** | **UD model, scored** |

The browser module does not do de/es POS, morphology or dependency syntax. The COMBO checkpoints are
custom heads on XLM-RoBERTa base, which Transformers.js cannot load as-is, and an
int8 ONNX export would be roughly 280 MB per language, about twice the browser NER
model. Language detection is still the stopword heuristic: it got the language wrong
for 3 of the 10 German and 3 of the 10 Spanish fixture sentences. Set `MorphologyOptions::language` and
`DependencyParsingOptions::language` to skip detection. Per-sentence routing is #162.

Heuristic NER (the default `EntityRecognitionMode::Heuristic`) is
capitalization-based and is not language-aware. Per-sentence language routing
and the end-to-end en/de/es journey are tracked in #162.

Browser package apps expose local benchmark scenarios in the `Benchmarks` tab.
Root scripts `bun run text-native:bench` and `bun run text-wasm:bench:all`
exercise native and browser paths respectively.

## Related crates

- `text-core`
- `text-transcripts` for SRT, WebVTT, Whisper JSON, and plain transcript parsing
- `text-lexical` for TF-IDF and BM25 corpus indexing from text inputs
- `media-core`
