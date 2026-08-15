# text-model-runtime

Shared tokenizer input types and native text runtime traits for `moritzbrantner-video-analysis`.

Default builds are deterministic and do not execute native model runtimes. Enable
`tokenizers`, `onnx`, or `candle` only where native runtime support is required.

## Stable contract

The stable surface is tokenizer input DTOs, runtime backend traits, raw
prediction records, softmax helpers, model load/run conformance reports, and
bundle validation without downloads.

## Quality and limits

This crate centralizes runtime contracts; it is not a model acquisition policy
or hosted inference client. Feature gates expose native paths, but callers must
explicitly select and configure them.

## Candle Devices

Candle-backed text execution defaults to CPU. Native server binaries can request
CUDA at startup when they are built with the opt-in `cuda` feature:

```bash
cargo run -p text-analysis-server -- --addr 127.0.0.1:3000
cargo run -p text-analysis-server --features cuda -- --cuda --cuda-device-index 0
```

CUDA is native-server only. It requires a CUDA-capable host plus Candle's CUDA
build prerequisites, and startup fails before binding if the requested CUDA
device cannot be initialized. WASM Candle bindings remain CPU-only.

## Package surface

- Primary workflow: `runtime.onnxQaProbe` is the native server/CLI probe for
  the default RoBERTa SQuAD2 ONNX bundle. It resolves an existing bundle from
  `.model-runtime` or auto-downloads it through `moritzbrantner-model-runtime`.
- Workflow operations: `runtime.tokenizeSummary`, `runtime.bundleCheck`, and
  `runtime.tokenizerProbe`.
- Model operations: `runtime.downloadBundle` materializes a preset bundle, and
  `runtime.onnxQaProbe` runs one local extractive-QA inference when built with
  `local-onnx`.
- Support operations: `runtime.softmax` exposes the shared stable softmax helper.
- Debug operations: `describe` inspects package metadata and operation support.
- Runtime support: deterministic/imported helpers are available through
  library, CLI, server, and WASM wrappers. Model-backed operations are
  server/CLI-only and report unsupported/server-only in WASM.
- Sample output includes `title`, `message`, `summary`, `result`, and
  operation-specific fields such as `tokens`, `tokenized`, readiness reports,
  or `probabilities`.
- `runtime.onnxQaProbe` and `runtime.downloadBundle` are explicit side-effectful
  exceptions to the no-download surface policy. Their execution plans declare
  filesystem reads, filesystem writes, and network access.

## Model Conformance

The crate exposes `TextModelLoadReport`, `TextModelRunReport`, and
`TextModelCapability` for text packages that need to show whether a model is
loadable or reference-only. `validate_text_model_bundle` checks local bundle
presence without downloads; `validate_tokenizer_bundle` additionally attempts a
feature-gated tokenizer load and sample tokenization.

Opt-in checks reuse `.model-runtime`:

```bash
scripts/sync_model_bundles.sh
cargo test -p text-model-runtime --features external-tests -- --ignored
```
