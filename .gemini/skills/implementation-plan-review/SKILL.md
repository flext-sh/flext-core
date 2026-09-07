---
name: implementation-plan-review
description: 'implementation plans, decision completeness, adversarial review'
metadata:
  aihub.tags: '["policy:causal-subprocess","policy:fail-loud","policy:no-fallback","policy:preflight-before-effects","policy:strict-execution","provenance:agents-owned","role:planning","updates:manual","usage:on-demand"]'
---

# Implementation Plan Review

Activate to review or promote a concrete implementation plan before coding.
Do not activate for source-code review, product discovery without approved
intent, or implementation of a plan whose review is not requested.

Read the `decision-completeness procedure` (skill file). Ground the
review in current repository owners and pinned dependencies, preserve supported
choices, and ask only for decisions that cannot be derived.

Return an evidence-backed verdict and a complete replacement plan. Review is
read-only unless the operator separately authorizes changes to its destination.
