# Issue tracker: GitHub

GitHub Issues in `moritzbrantner/nlp-stack` are the durable work queue. PRDs are
parent issues; implementation slices carry canonical `parent`, `blocked_by`,
and `scope` YAML frontmatter.

Only issues labeled `agent-ready` are eligible for Agent Loop execution.
Unresolved issue references in `blocked_by`, prerequisite/dependency sections,
or explicit `Blocked by ...` lines keep an issue out of the runnable queue.

Release authorization must come from an open issue in this same repository. An
issue in `rust-packages` may authorize extraction work but cannot authorize
publication from this checkout.
