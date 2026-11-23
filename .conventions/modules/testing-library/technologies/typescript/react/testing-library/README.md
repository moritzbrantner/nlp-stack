# React Testing Library conventions

## RTL-001 — Test observable user behavior

- Interact through user-facing controls and assert observable outcomes, not React internals.
- Tests should survive implementation-preserving refactors; do not assert hook state, private functions, lifecycle details, or child-component internals merely to increase coverage.

## RTL-002 — Use Testing Library's semantic query APIs

- Apply TEST-020 through Testing Library's user-facing query APIs, preferring roles with accessible names, then labels and visible text where appropriate.
- Use `getByTestId`/`findByTestId` or related test-marker queries only when TEST-020 permits a deliberate test identifier because no stable user-facing semantic target exists.

## RTL-003 — Apply DOM testing progressively

- Do not require a React Testing Library test merely because a React component exists.
- Test pure non-DOM logic with ordinary unit tests without rendering React.
- Prefer React Testing Library for meaningful component behavior such as interaction, local state transitions, validation, async UI, dialogs, menus, and keyboard behavior.
- Exercise cross-component DOM workflows as integration tests with real composition when that gives useful confidence.
- Use a real-browser layer such as Playwright for behavior that materially depends on layout, scrolling, pointer geometry, media, browser APIs, navigation, or other browser semantics that DOM emulation cannot prove reliably.

## RTL-004 — Model interactions through user events

- Prefer `userEvent.setup()` and awaited user interactions for normal input, pointer, and keyboard behavior.
- Use lower-level event dispatch only when the behavior specifically depends on an event that cannot be expressed as a normal user interaction.

## RTL-005 — Wait for observable asynchronous state

- Wait for the state the user can observe with semantic async queries or bounded waiting helpers.
- Do not use arbitrary sleeps to make a test pass.
- Do not manually orchestrate React update flushing when an ordinary user interaction and observable wait can express the behavior.

## RTL-006 — Keep component composition real by default

- Prefer rendering real child components and providers over mocking React implementation boundaries.
- A project may expose one focused render helper for established providers such as routing, query clients, themes, or localization instead of duplicating setup in each test.
- Mock external boundaries at their natural seam. For HTTP behavior, prefer the repository's established network-mocking layer; when introducing one, prefer request-level mocking such as MSW over mocking React components or data hooks.

## RTL-007 — Avoid snapshot-only and duplicate confidence

- Prefer explicit behavioral assertions over broad DOM snapshots. Use small snapshots only when the serialized or rendered shape is itself a meaningful contract.
- Do not reproduce the same confidence mechanically across React Testing Library, Storybook, and Playwright.
- Use the cheapest stable layer that proves the behavior, and add a browser-level test when browser-specific risk remains.
