# Tooling conventions

## BUN-001 — Use Bun as the default JavaScript toolchain

- Use Bun for packages, scripts, and JavaScript/TypeScript where required tooling supports it.

## TAILWIND-001 — Prefer Tailwind CSS when practical

- Prefer Tailwind for application styling when utility classes preserve ownership near the markup.

## TAILWIND-002 — Use semantic tokens and named variants

- Use semantic tokens and named variants for visual decisions.
- Do not default to arbitrary radii, shadows, gradients, blur, or raw palette colors.

## Child scopes

- [`vite/`](vite/)
- [`storybook/`](storybook/)
- [`playwright/`](playwright/)
- [`lighthouse/`](lighthouse/)
- [`vitest/`](vitest/)
