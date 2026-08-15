# text-transcripts

Transcript document contracts, parsing, normalization, and subtitle formatting.

## Feature flags

- `external-tests`: reserved for downstream format-compatibility smoke tests

## Stable contract

The stable surface is transcript contracts, segment/word normalization,
SRT/WebVTT/plain/Whisper JSON parsing, formatting, conversion to
`TextSegmentContract`, and transcript-specific text pipeline analyzers.

## Quality and limits

Package operations parse and format text only. Audio decoding, VAD, ASR command
execution, model resolution/downloads, and speaker-model execution belong to
the higher-layer `audio-analysis-transcription` adapter.

## Example

```rust,no_run
use text_transcripts::{parse_whisper_json, TranscriptionContract};

let parsed = parse_whisper_json(include_bytes!("../../../../tests/fixtures/whisper-sample.json"))?;
let transcript = TranscriptionContract::from(parsed).normalized()?;

assert!(!transcript.text_or_joined().is_empty());
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Package surface

- Primary workflow: `transcripts.parse` parses plain text, Whisper JSON, SRT, or
  WebVTT into the normalized transcript contract.
- Workflow operations: `transcripts.parse`, `transcripts.normalize`,
  `transcripts.formatSrt`, `transcripts.formatWebVtt`, and
  `transcripts.toTextSegments`.
- Debug operations: `describe` inspects package metadata and operation support.
- Runtime support: pure Rust parsing/formatting package-surface operations are
  available through library, CLI, server, and WASM wrappers.
- Sample output includes `title`, `message`, `summary`, `result`, and
  operation-specific fields such as `segments`, `text`, `srt`, or `webVtt`.
- Package-surface operations do not invoke whisper.cpp or external ASR tools.

Browser benchmarks cover parse, normalize, and SRT formatting workflows through
`bun run text-wasm:bench:all`.

## Related crates

- `text-core`
- `audio-analysis-transcription` (higher-layer execution adapter)
