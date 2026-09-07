---
name: product-capability
description: 'product intent, capability contracts, requirement boundaries'
metadata:
  aihub.tags: '["policy:fail-loud","policy:no-fallback","policy:preflight-before-effects","policy:strict-execution","provenance:agents-owned","role:planning","updates:manual","usage:on-demand"]'
---

# Product Capability

Activate when approved product intent exists but engineering still needs an
explicit capability boundary. Do not activate for product discovery without an
approved decision, implementation detail already owned by architecture, or a
wording-only request.

## Preflight

Read the smallest authoritative set that establishes the actor, visible promise,
authorization boundary, business rules, data ownership, lifecycle, rollout or
migration constraints, non-goals, current consumers, and existing architecture.
Classify each statement as approved product truth, observed implementation,
engineering preference, or unresolved question.

Missing product truth stops the handoff before a capability artifact or
implementation effect. Never promote an inference, default, adjacent feature, or
possible future consumer into approved scope.

## Single contract

Publish the result at the project's declared product-context owner; do not invent
a path or parallel planning stack. Include only:

- capability: actor, surface, and material outcome;
- constraints: fixed rules, invariants, trust and ownership boundaries;
- implementation contract: inputs, outputs, states, transitions, and policy
  implications justified by current evidence;
- non-goals: explicitly excluded behavior;
- open questions: exact owner decisions that block implementation;
- handoff: ready for implementation, architecture review, or product decision.

The contract owns product intent, not schemas, service design, or delivery steps.
Reference those owners instead of copying them. A handoff is implementation-ready
only when every required product decision is resolved; otherwise return the first
blocking question without fallback scope or fabricated policy.
