# Inline simplification

## Gate

Resolve the approved behavior, canonical owner, affected consumer, baseline or
acceptance test, generated-file ownership, runtime, and native gate before the
edit. Missing evidence stops with zero code effects.

## Per-unit pass

After implementing the complete behavior:

- remove redundant branches, nesting, temporary state, conversions, wrappers,
  duplicate expressions, and narrative comments;
- omit declarations equal to a canonical typed default while preserving explicit
  overrides;
- reuse established project/library primitives and keep control flow, names, and
  ownership explicit;
- preserve interfaces, error types and causes, ordering, effects, security,
  observability, types, performance contracts, and non-obvious invariants;
- remove placeholders, stubs, hardcoded results, fallback, compatibility, dead
  code, and superseded paths;
- run the narrow public behavior check and retain its causal failure unchanged.

Do not create premature helpers, frameworks, utility bags, hidden control flow,
metaprogramming, or dense code golf. Do not edit a generated projection or weaken
a test to shorten code.

If the graph proves duplicate owners, god responsibilities, repeated cross-unit
work, or atomic consumer rewiring, pass the current evidence once to `dry`.
Otherwise finish locally. Report complete behavior, units simplified, code
removed, upstream decisions, runtime/gates, first failure, and zero residue.
