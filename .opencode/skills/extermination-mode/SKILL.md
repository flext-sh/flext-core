---
name: extermination-mode
description: 'zero residue, contract removal, consumer rewiring'
metadata:
  aihub.tags: '["policy:atomic-effects","policy:causal-subprocess","policy:fail-loud","policy:no-fallback","policy:preflight-before-effects","policy:strict-execution","policy:zero-residue","provenance:agents-owned","role:migration","updates:manual","usage:router"]'
---

# Extermination Mode

Use only after the operator has approved a complete cutover and identified the
superseded contract. This mode removes the old contract; it does not authorize
deleting data, repositories, branches, runtime state, or unrelated work.

1. Inventory the old owner, producers, consumers, loaders, fallbacks, aliases,
   tests, fixtures, templates, generated facades, documentation, and terms.
2. Classify tests by behavior:
   - delete tests whose sole purpose is enforcing the removed contract;
   - rewrite tests that protect behavior still required by the final contract;
   - preserve unrelated tests and concurrent WIP exactly.
3. Rewire useful consumers to the final SSOT before removing the old owner.
   Use structural search/replace for mechanical migrations and review every
   match. Elide every field and call argument equal to a canonical typed default.
   Never add a compatibility alias, dual reader, fallback, or undeclared
   default-on-error behavior to make deletion easier.
4. Delete exact tracked obsolete files with scoped patches. Regenerate managed
   indexes and artifacts through their canonical owner; do not hand-maintain a
   generated facade.
5. Prove the cutover with zero-residue semantic searches, focused behavior
   tests, generation fixed point, static gates, and the repository's full gate.
   A failed gate means the extermination is incomplete, not that the gate or
   generator should be weakened.

Complete owner, consumer, approval, and concurrent-work preflight before the
first effect. Preserve the first gate or subprocess failure unchanged; a failed
cutover publishes nothing and runs no fallback path.

Stop when an apparent obsolete target still has a valid consumer, belongs to
concurrent work, or deletion would cross the operator-approved boundary.
