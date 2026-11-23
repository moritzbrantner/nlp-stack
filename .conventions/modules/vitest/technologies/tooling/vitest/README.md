# Vitest conventions

## VITEST-001 — Separate execution kinds with names and scripts

- Keep tests at their dependency scope; encode kind in filenames such as .unit.test.ts, .integration.test.ts, or .bench.ts.
- Provide one non-interactive script per kind and separate configuration when setup differs materially.

## VITEST-002 — Do not commit focused or silently disabled tests

- Focused tests such as `test.only`/`describe.only` fail deterministic verification.
- Disabled/skipped tests do not remain inside required suites; move known flaky work into an explicit quarantine tier instead of hiding it with `.skip`.
- Commented-out tests are not a durable backlog mechanism; use an actionable TODO or issue instead.
