# text-question-answering-app

React, TypeScript, TailwindCSS, Bun, and oxfmt frontend for `text-question-answering`.

Run the server:

```bash
cargo run -p text-question-answering-server -- --addr 127.0.0.1:3000
```

Run the app:

```bash
bun run --cwd packages/text-question-answering-app dev
```

The default `qa.answer` preset omits imported predictions so native server
builds with `local-onnx` use the local RoBERTa SQuAD2 ONNX path by default.
Imported predictions remain the WASM-compatible alternative.
