# @moritzbrantner/text-core-wasm

WASM package for `text-core`.

```bash
bun run --cwd packages/text-core-wasm build
bun run --cwd packages/text-core-wasm bench:browser
```

Use `bun run text-wasm:bench:all` from the repository root to run the shared browser benchmark suite across the text WASM packages. Results are local to the current browser, machine, and build profile.
