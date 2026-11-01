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

## Language support (en/de/es)

Issue #117 asks for the same seven operations in English, German and Spanish.
Current state:

| Operation | en | de | es |
| --- | --- | --- | --- |
| Language detection | stopword-lexicon heuristic | stopword-lexicon heuristic | stopword-lexicon heuristic |
| Sentence segmentation | language-neutral rules | language-neutral rules | language-neutral rules |
| Tokenization | language-neutral rules | language-neutral rules | language-neutral rules |
| Named entities (local model) | **model, gold-scored** | **model, gold-scored** | **model, gold-scored** |
| Lemma / morphology | English heuristics | not supported (#160) | not supported (#160) |
| Part of speech | English heuristics | not supported (#160) | not supported (#160) |
| Dependency syntax | English heuristics | not supported (#161) | not supported (#161) |

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
