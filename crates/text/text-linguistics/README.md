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
- `TextAnalyzer` adapter for text pipelines

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

When the `candle` feature is enabled, callers can use a local `bert-base-ner`
token-classification model by setting
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

The deterministic heuristic pipeline is always available. `dslim/bert-base-NER`
is the implemented opt-in Candle token-classification path; it becomes loadable
when the local bundle exists under `.model-runtime` and the crate is built with
`candle,model-bundles`.

```bash
scripts/sync_model_bundles.sh
cargo test -p text-linguistics --features external-tests -- --ignored
```

Browser package apps expose local benchmark scenarios in the `Benchmarks` tab.
Root scripts `bun run text-native:bench` and `bun run text-wasm:bench:all`
exercise native and browser paths respectively.

## Related crates

- `text-core`
- `text-transcripts` for SRT, WebVTT, Whisper JSON, and plain transcript parsing
- `text-lexical` for TF-IDF and BM25 corpus indexing from text inputs
- `media-core`
