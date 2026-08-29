# Vitest conventions

## VITEST-001 — Separate execution kinds with names and scripts

- Keep tests at their dependency scope; encode kind in filenames such as .unit.test.ts, .integration.test.ts, or .bench.ts.
- Provide one non-interactive script per kind and separate configuration when setup differs materially.
