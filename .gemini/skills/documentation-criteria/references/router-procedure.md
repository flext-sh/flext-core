# Documentation Criteria procedure

# Documentation Criteria

Select the smallest project-owned documentation set that preserves a current
requirement, decision, interface, implementation contract, or execution plan.
Do not create documents from file count, generic ceremony, or possible future use.

## Preflight

Before selecting or writing an artifact, resolve the current request, users,
acceptance criteria, semantic impact, existing owners and documents, contract and
data-flow changes, UI scope, dependency decisions, implementation boundaries,
approval state, templates, and publication destinations.

The first missing or conflicting classification input stops with zero document
effects. Do not guess a document type, create a generic placeholder, select a
default path, or publish a partial document set.

## Select by owned concern

Retain an artifact only when its current consumer needs its distinct authority:

| Artifact | Sole owned concern | Read when selected |
|---|---|---|
| PRD | user/business problem, current scope, measurable acceptance | `PRD template` (skill file) |
| UI specification | screens, states, transitions, interaction, accessibility | `UI template` (skill file) |
| ADR | durable technical decision, viable options, consequences | `ADR template` (skill file) |
| Design document | current-system evidence, interfaces, data flow, implementation and proof | `design template` (skill file) |
| Work plan | dependency-ordered delivery and integration evidence | `plan template` (skill file) |
| Task | one bounded owned result required by the project's execution owner | `task template` (skill file) |

Use the project's declared document owner and location. Do not copy mutable
schemas, policy, configuration values, rollout status, or tracker state into a
second authority; link to their owners.

## Ordering and publication

Create only currently required artifacts. Product intent precedes UI behavior;
approved durable decisions precede technical design; approved design precedes an
execution plan. A task exists only when the project execution owner requires it.

Every artifact must be independently complete, evidence-backed, linked to its
upstream authority, and approved at its declared boundary before a dependent
artifact begins. Publish each complete artifact atomically. A failed write or
approval leaves no dependent draft, alternate template, or stale superseded file.

Verify links, traceability, measurable acceptance, owner identity, and absence of
unnecessary documents. Never mark implementation or phase completion from a
document status alone.
