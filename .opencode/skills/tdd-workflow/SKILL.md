---
name: tdd-workflow
description: 'test driven development, behavior contracts, regression tests'
metadata:
  aihub.tags: '["policy:atomic-effects","policy:causal-subprocess","policy:fail-loud","policy:no-fallback","policy:preflight-before-effects","policy:strict-execution","policy:zero-residue","provenance:agents-owned","role:testing","updates:manual","usage:on-demand"]'
---

# TDD Workflow

Activate when implementing or correcting observable behavior through a
red-green-refactor cycle. Before changing behavior, read the
`complete procedure` (skill file) and follow the active project's
public contract and native validation surface. Do not add tests for a prose-only
or configuration-preserving edit with no behavior boundary.
