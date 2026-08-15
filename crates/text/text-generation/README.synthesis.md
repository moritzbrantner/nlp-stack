# text-generation

Deterministic text synthesis from terms and events for `video-analysis`.

## Feature flags

- No optional feature flags today.

## Example

```rust,no_run
use text_linguistics::{analyze_text, LinguisticAnalysisOptions};
use text_generation::{synthesize_from_analysis, TextSynthesisOptions};

let analysis = analyze_text(
    "Alice presented the roadmap in Berlin.",
    &LinguisticAnalysisOptions::default(),
) .unwrap();
let document = synthesize_from_analysis(
    "doc-1",
    &analysis,
    TextSynthesisOptions::default(),
) .unwrap();

assert!(!document.value.text.trim().is_empty());
```

## Related crates

- `data-inversion-core`
- `text-core`
