---
applyTo: '**/*.py'
---

# Read configuration through the project's typed owner

Use the configuration/settings facade declared by the current repository. Parse
environment variables, files, and CLI input only at that boundary; consumers
receive validated typed values, never raw dictionaries or repeated environment
lookups.

- Fixed business rules and user-overridable settings have one owner each.
- No product-specific import, tool-home helper, duplicated constant,
  consumer-level home discovery, or inherited-secret precedence in generic
  Python guidance. A typed owner may derive the platform home when that value is
  the canonical calculated default.
- Tests vary valid inputs and verify schema, types, invariants, derivations,
  precedence, round-trip, consumer behavior, and generated structure.
- Never assert today's configurable paths, endpoints, model names, rankings,
  defaults, or scalars. Derive expectations from fixture input or the typed
  owner.
- When the schema changes, migrate every consumer atomically, reject the old
  format, delete superseded fixtures/docs, and prove second-run fixed point.
