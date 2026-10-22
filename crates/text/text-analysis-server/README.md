# text-analysis-server

HTTP API adapter for `moritzbrantner-text-analysis`.

```bash
cargo run -p text-analysis-server -- --addr 127.0.0.1:3000
cargo run -p text-analysis-server --features cuda -- --addr 127.0.0.1:3000
```

Candle models use CUDA device 0 when the server is built with `cuda` and a CUDA
device initializes; otherwise they run on the CPU. There is no device flag: the
wrapped capabilities choose CUDA first and fall back to the CPU.
