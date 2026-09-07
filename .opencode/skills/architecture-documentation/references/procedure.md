# Evidence-based architecture documentation

## Frame the artifact

State its audience, question, scope, observed revision, and authoritative
sources. Separate current behavior, approved decisions, and proposed changes.
If the repository has a documentation owner or template, use it.

Load and validate the complete source, owner, revision, template, destination,
and verification set before writing. The first missing, conflicting, or
inaccessible prerequisite stops with no partial note, diagram, ADR, or inferred
replacement source.

## Describe the system

1. Identify each component, its owner, responsibility, trust boundary, and
   maintained interface.
2. Trace critical flows from input through validation, transformation, storage,
   output, failure handling, and observable signals.
3. Link to schemas and interfaces at their owner. Summarize their role without
   copying definitions that can drift.
4. Add the smallest useful diagram when boundaries, dependencies, lifecycle,
   state transitions, or data flow are materially clearer visually. Ensure every
   node and edge is supported by a cited source.
5. Record operational constraints only when configuration or runtime evidence
   proves they are maintained behavior.

For a consequential decision, use the project's ADR owner and record context,
considered options, the chosen decision, consequences, and supporting evidence.
Do not turn an undocumented inference into a decision record.

## Verify

Resolve links and identifiers, compare diagrams with the owning configuration,
and exercise the selected representative runtime path. The first causal command
failure stops without retry or substitute evidence. Publish the complete verified
artifact atomically and leave no draft, copied schema, or superseded diagram
residue. Keep plans, rollout status, and task tracking in their owning surfaces.
