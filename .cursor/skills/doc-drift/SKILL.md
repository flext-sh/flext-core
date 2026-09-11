---
name: doc-drift
description: 'documentation drift, runtime contracts, authority reconciliation'
metadata:
  aihub.tags: '["policy:atomic-effects","policy:causal-subprocess","policy:fail-loud","policy:no-fallback","policy:preflight-before-effects","policy:strict-execution","policy:zero-residue","provenance:agents-owned","role:documentation","updates:manual","usage:on-demand"]'
---

# Documentation drift

Compare active documentation with code, declared configuration, and canonical
upstream documentation. Treat commands and schemas as executable interfaces.

Before changing documentation, load every affected claim, canonical owner,
consumer, generator, destination, and validation command. The first missing or
conflicting owner stops with zero writes and no inferred replacement.

1. Read repository law and owner configuration.
2. Inventory commands, flags, paths, schemas, versions, and generated files.
3. Verify each claim against owning code or a pinned upstream source.
4. Remove extinct contracts and update all consumers, fixtures, and evals as one
   atomic cutover.
5. Change generated output only through its owner and prove fixed point.
6. Run native documentation, validation, and contradiction gates.

The first causal validation failure stops that invocation without an unchanged
retry, warning, alternate command, or partial convergence. Correct the owner and
rerun invalidated gates in the same cutover. Publish only the complete verified
result and require zero extinct-command, copied-schema, stale-fixture, and
generated-output residue. Tool-specific drift belongs to its owner.
