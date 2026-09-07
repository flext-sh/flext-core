---
name: code-review-expert
description: 'code review, correctness analysis, regression detection'
metadata:
  aihub.tags: '["policy:causal-subprocess","policy:fail-loud","policy:no-fallback","policy:preflight-before-effects","policy:strict-execution","provenance:agents-owned","role:review","updates:manual","usage:on-demand"]'
---

# Code Review Expert

Activate for an evidence-backed review of a concrete diff, patch, staged change,
or commit range. Do not activate when the user requests only implementation of
an already reviewed correction.

Read the `review procedure` (skill file). Review is read-only unless
the user separately authorizes implementation. Missing review targets, owners,
or required context stop before a verdict; never invent findings, approve by
default, or substitute another range.
