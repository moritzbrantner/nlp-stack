# text-question-answering-server

Thin HTTP API adapter for `moritzbrantner-text-question-answering`.

Run:

```bash
cargo run -p text-question-answering-server -- --addr 127.0.0.1:3000
```

Endpoints:

- `GET /health`
- `GET /api/package`
- `GET /api/schema`
- `GET /api/operations`
- `POST /api/run`
- `POST /api/<operation-id>`
