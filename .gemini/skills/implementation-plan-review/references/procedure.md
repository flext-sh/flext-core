# Decision-complete implementation-plan review

## Preflight

Freeze the exact plan under review and resolve its approved product intent,
audience, success behavior, repository instructions, architecture owners,
pinned dependencies, analogous implementations, consumers, runtime, native
gates, deployment path, and concurrent work. Use `product-capability` when the
product boundary is missing and `search-first` for the reusable technical
evidence packet.

Current external documentation is required only for load-bearing behavior that
may have changed since the pinned dependency was released. Use primary sources
for that exact version. Missing plan identity or product intent stops review
without inventing a direction.

## Review

Test each load-bearing workstream for:

- **outcome:** named actor, visible result, and observable success;
- **scope:** current requirement, non-goals, and smallest coherent delivery;
- **ownership:** canonical producer, current consumers, generated projections,
  and removal of superseded paths;
- **interfaces:** concrete inputs, outputs, schemas, states, ordering, and
  authorization boundaries;
- **failure:** first causal error, zero-effect boundary, cleanup, migration or
  recovery, and what remains unchanged;
- **operations:** rollout sequence, compatibility decision when explicitly
  required, observability, data volume, performance budget, and security model;
- **proof:** real runtime path, focused tests, native gates, integration, and
  landing boundary.

Classify a plan as too vague where the implementer would still choose product,
architecture, schema, dependency, or failure behavior. Classify it as too local
where detailed steps omit audience, outcome, sequencing, integration, or the
highest-risk boundary. Detail must follow risk rather than comfortable code.

Challenge terms such as scalable, robust, graceful, secure, flexible, fast, or
later unless the plan names the measurable contract or assigns the unresolved
decision to its real owner.

## Findings and promoted plan

Steelman each choice before rejecting it. A finding contains consequence,
decisive repository or primary-source evidence, and the owner-correct change:

- **Blocker:** the plan cannot implement the approved outcome or leaves a
  required decision to the implementer.
- **Major:** the plan can work but violates a current owner, omits a material
  failure/rollout boundary, or misallocates risk.
- **Minor:** non-blocking precision or sequencing improvement.

Preserve every validated decision. Put non-derivable product choices in one
targeted question rather than selecting a preferred answer.

Return verdict, ordered findings, supported choices, and one replacement plan
covering outcome, owner changes, interfaces, data flow, failure modes,
migration/rollout, tests, runtime proof, landing, assumptions, and explicit
non-goals. A plan is ready only when another competent implementer can execute it
without making a material design or product decision.
