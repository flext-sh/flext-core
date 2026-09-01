# Search-first procedure

## Preflight

Before technical design or an implementation edit, resolve the repository root,
applicable instructions, architecture authority, public entry point, canonical
owner, current consumers, manifests and lockfiles, installed dependency versions,
generated-file owner, runtime path, and native gates. Local repository evidence
comes first; primary dependency documentation follows only after its pinned
version is known.

Missing or contradictory authority, an unresolved generated owner, or a
hypothetical consumer blocks implementation. A filename, one keyword search, or
historical code never proves absence or ownership.

## Evidence packet

Keep one smallest sufficient packet in the active delivery record:

- project authority and affected public behavior;
- owner, callers, registrations, generators, and projections;
- semantically adjacent implementations and current consumers;
- dependencies, pinned versions, established adapters, runtime, and gates.

Read decisive definitions and callers, not search snippets. Refresh only a field
invalidated by an owner, consumer, dependency, generated, or runtime change; do
not recursively restart discovery.

## Decision order

1. Reuse the existing capability unchanged.
2. Extend its canonical owner and rewire every affected consumer.
3. Use an approved dependency through the established adapter.
4. Implement the smallest missing behavior required by a current consumer.

`yagni` owns necessity. `ssot` selects one writable authority and classifies
projections. `solid` applies only to surviving architectural boundaries.
`simplify` removes local accidental complexity during each edit. Invoke `dry`
only for proven structural duplication in the authorized graph; it consumes the
same packet and does not restart this skill.

The bounded route is:

`search-first -> yagni -> ssot -> optional solid -> implement + simplify -> optional dry -> recheck yagni/ssot/solid -> simplify -> runtime -> gates`

Do not add packages, helpers, abstractions, adapters, registries, or future modes
without a current consumer and reachable runtime. Never preserve a parallel owner
or turn incomplete search into a fallback design.

## Evidence

Report the packet, reuse decision, YAGNI result, SSOT authority, relevant SOLID
decision, exact owner and consumers changed, code removed by simplify/DRY, and
decisive runtime/gate results. Child failure remains causal and stops the route;
no later step or alternate tool makes it green.
