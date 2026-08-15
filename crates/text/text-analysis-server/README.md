# text-analysis-server

HTTP API adapter for `moritzbrantner-text-analysis`.

```bash
cargo run -p text-analysis-server -- --addr 127.0.0.1:3000
cargo run -p text-analysis-server --features cuda -- --cuda --cuda-device-index 0
```

`--cuda` is available only for native CUDA-capable builds. Default launches are
CPU-only, and WASM package surfaces remain CPU-only.

## Local CUDA Verification

Run CUDA server checks from the repository root so Cargo can find the workspace
manifest. On `castle`, prefer the CUDA 12 toolkit path:

```bash
cd /home/moenarch/moritzbrantner/rust-packages
CUDA_HOME=/usr/local/cuda-12 \
LD_LIBRARY_PATH=/usr/local/cuda-12/targets/x86_64-linux/lib:/lib/x86_64-linux-gnu \
cargo run -p text-analysis-server --features cuda -- --cuda --cuda-device-index 0
```
