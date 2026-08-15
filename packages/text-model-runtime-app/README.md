# text-model-runtime-app

React, TypeScript, TailwindCSS, Bun, and oxfmt frontend for `text-model-runtime`.

Run the server:

```bash
cargo run -p text-model-runtime-server -- --addr 127.0.0.1:3000
```

Run the app:

```bash
bun run --cwd packages/text-model-runtime-app dev
```

The default app operation is `runtime.onnxQaProbe`. Native server builds can
resolve or download the default RoBERTa SQuAD2 ONNX bundle into
`.model-runtime`; WASM/client mode keeps validation and tokenizer helpers and
reports model execution as server-only.
