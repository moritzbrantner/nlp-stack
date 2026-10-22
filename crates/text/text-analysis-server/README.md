# text-analysis-server

HTTP API adapter for `moritzbrantner-text-analysis`.

```bash
cargo run -p text-analysis-server -- --addr 127.0.0.1:3000
```

Launches are CPU-only. The adapter no longer sets a process-wide Candle device:
library callers choose a device per model with `on_device`, and adapter-level
execution configuration is tracked in #17. The `cuda` feature still compiles the
CUDA kernels into the wrapped capabilities.
