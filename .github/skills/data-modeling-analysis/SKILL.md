---
name: data-modeling-analysis
description: 'data models, serialization boundaries, validation design'
metadata:
  aihub.tags: '["policy:fail-loud","policy:no-fallback","policy:preflight-before-effects","policy:strict-execution","provenance:agents-owned","role:architecture","updates:manual","usage:on-demand"]'
---

# Data Modeling Analysis

Analyze how a codebase represents, validates, mutates, serializes, and evolves
data. Apply the project's language and framework contracts rather than assuming
one type library.

Load the complete schemas, types, serializers, consumers, versions, and runtime
evidence before issuing a correction. A missing or conflicting boundary stops at
the first defect with no rewrite, inferred library, generic model, or partial map.

1. Locate schemas, domain types, transport types, persistence mappings, and state
   containers.
2. Map each external boundary to its validation and normalization owner.
3. Identify ambiguous optionality, unbounded maps, duplicated representations,
   unsafe coercion, and mutation whose ownership is unclear.
4. Trace serialization and deserialization, including unknown fields, versioning,
   defaults, and failure behavior.
5. Compare persisted, in-memory, and public representations and identify any
   lossy or implicit conversion.
6. Report the data flow, invariants, concrete risks, evidence paths, and the
   smallest owner-first correction. Do not mutate code unless requested.

Prefer explicit project-native types and boundary validation. Immutability is a
tool for visible ownership, not a blanket rewrite requirement. Never recommend
unsafe deserialization or a parallel schema owner.
