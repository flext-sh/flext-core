# Structural DRY remediation

## Evidence gate

Resolve public behavior, owner and caller graph, generators/projections,
dependencies, runtime, gates, and a green baseline. Confirm duplication by
semantics, responsibility, or measured repeated work; size and textual similarity
are insufficient. Missing baseline or ownership stops before edits.

An index, call graph, LSP result, or static dead-code report is candidate
evidence, never proof that a symbol has no consumer. Before removal or rename,
cross-check source, tests, examples, documentation, templates and generators;
public exports and lazy export maps; reflection and string-based lookup;
registries, plugins and entry points; facade aliases; inheritance/MRO and
protocol implementations; serialization names; and downstream package
consumers. A valid unresolved reference blocks the change. Edit a generated
owner or template and regenerate; never patch its projection.

Measure the same scoped files before and after with the project's metrics owner,
or `tokei` only when already available and required. Performance claims require
the same representative profile, query count, trace, or benchmark before and
after; fewer lines do not prove speed.

## Remediation

1. Protect public inputs, outputs, errors, ordering, effects, and invariants.
2. Extend the authority already selected by `ssot`; DRY never elects another
   owner. Rewire every current consumer atomically and remove inputs equal to
   the owner's canonical calculated defaults.
3. Split god units only along real policy, persistence, transport, presentation,
   or orchestration change boundaries. Keep a thin explicit orchestrator.
4. Remove duplicated rules and defaults, repeated I/O/computation, needless
   layers, broad interfaces, dead adapters, compatibility paths, fixtures, flags,
   dependencies, examples, and copied documentation. For symbol removal or
   rename, perform the dynamic-reference cross-check above before the first
   deletion and search the old identity again after rewiring.
5. Apply `simplify` to each changed unit, then recheck `yagni`, `ssot`, and
   `solid`. An abstraction requires a current shared semantic consumer.

Do not replace duplication with a utility bag, registry, framework, reflection,
configuration copy, generated copy, or dense code golf. Old and new owners never
coexist.

## Proof

Run the public runtime, focused contract/failure checks, and affected native
gates. Preserve the first child failure and publish no partial result. Repeat
semantic residue searches and the identical measurement/profile scope. Report
owners, consumers, deleted code, measurements, runtime/gates, causal failure,
zero effects, and zero residue.
