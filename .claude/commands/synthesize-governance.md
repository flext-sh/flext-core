---
description: "Synthesize an external governance corpus into current canonical owners without copying its structure."
argument-hint: "<source corpus and requested governance outcome>"
disable-model-invocation: true
---

# Synthesize governance

Treat `$ARGUMENTS` as one explicit source corpus and the requested governance
outcome. Refuse an absent, unreadable, symlinked, ownership-ambiguous, or
unbounded source, or a target whose canonical governance owners cannot be
resolved.

1. Read the target instructions, artifact contracts, catalog, ownership map,
   runtime, projections, evaluations, native gates, and current consumers.
2. Inspect the complete relevant source bundles, including references, scripts,
   assets, manifests, provenance, license, generated state, and executable
   effects. Historical presence is evidence, never authority or permission.
3. Derive each behavior's outcome, trigger, non-trigger, inputs, effects,
   failure contract, scope, consumer, and proof. Classify it as already owned,
   an extension of one current owner, a distinct required capability, or
   rejected residue.
4. Place mandatory invariants in rules, conditional procedures in skills,
   explicit invocation grammar in commands, deterministic behavior in typed
   runtime, and provider forms in projections. Split mixed sources and reference
   owners instead of concatenating prose.
5. Change only canonical owners. Rewire every current consumer and update
   semantic evaluations, guarantee mapping, inventory, documentation, and
   generated projections atomically. Remove superseded canonical identities in
   the same cutover; never modify or delete the supplied source corpus.
6. Exercise representative behavior, run every affected native gate, regenerate
   twice to prove fixed point, and search for duplicate owners, aliases, foreign
   runtimes, private paths, compatibility, fallback, retry, and stale consumers.

Return the source and target identities, semantic classifications, selected
owners, rejected behavior with reasons, consumer rewiring, exact runtime and
gate evidence, projection fixed point, and first blocker. Do not report file
counts or non-empty output as semantic success.
