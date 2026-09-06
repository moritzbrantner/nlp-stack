# Coding-agent readiness

`nlp-stack` consumes shared coding-agent policy live rather than copying it into the repository. Repository-local instructions stay in `AGENTS.md`; `.coding-tooling.json` names stable shared convention IDs whose presence is required for this repository.

## Quick canary

Run before an implementation session:

```bash
bash scripts/check-agent-readiness.sh
```

The canary checks the committed readiness contract, resolves the current `coding-agent-conventions` stack for this repository, validates the current `coding-agent-skills` source, and resolves its `standard` profile. It accepts `coding-tooling` from `PATH` or `CODING_TOOLING_DIR`. It finds `coding-agent-skills` from `CODING_AGENT_SKILLS_ROOT`, a sibling checkout, or the shared Moenarch environment registry.

A failed policy or skill resolution is an environment/readiness failure, not a product-code failure. Fix the checkout/registry setup rather than copying shared rules or skills into this repository.

## Source-mode canary

Before work that needs unreleased `moenarch-foundation` source, run:

```bash
bash scripts/check-agent-readiness.sh --with-source
```

This additionally requires the sibling `../moenarch-foundation` checkout at the exact revision pinned by `.coding-tooling.source-deps.json`. It asks `coding-tooling` to validate local-only source resolution, temporarily activates the managed Cargo patch configuration when necessary, and runs `cargo metadata`. If the canary activated source mode itself, it deactivates it again on completion or failure; a source mode that was already active is left active.

Hosted CI remains repository-local and registry-based. It validates the structural readiness contract through the normal Python test suite, but it does not clone private sibling repositories or add credentials merely to reproduce the local multi-repository development graph.

## Handoff

The readiness canary establishes that an agent can load policy, skills, and—when requested—the exact local source graph. It does not replace repository verification. Implementations still validate the narrowest affected scope first. Exact-head handoff is the `handoff` tier in `.coding-tooling.json`; run `coding-tooling run --tier handoff --strict --json`. The tier delegates to the repository-owned `bun run check` gate, so command ownership is not duplicated in a second agent-loop config.
