---
applyTo: '**'
---

# Validate the complete workflow before effects

Load and validate every input, environment value, configuration owner, source,
destination, provider capability, external executable, authorization,
child-process contract, storage bound, ownership proof, and publication
condition required by the entire workflow before its first mutation or external
call.

Preflight is read-only and deterministic. Discovery performed after an effect,
lazy validation inside a mutation loop, and validate-as-you-go publication are
prohibited. If a prerequisite can change between preflight and effect, acquire
the declared ownership/serialization primitive before validating and retain it
through publication.

## Adjudicate divergent objects before replacement

For a bulk update, migration, regeneration, adoption, or replacement, inventory
the complete scoped population and establish each current object's physical
identity, ownership proof, provenance, consumers, and proposed owner before the
first effect. Normalize provider or storage representation only through a typed
owner transformation. Sampling, matching names, matching descriptions, similar
shape, age, location, or a legacy label never proves semantic equivalence.

Automatic adoption is permitted only when current and proposed bytes and
physical contract match exactly, or when an approved deterministic class rule
proves the entire normalized delta is representation-only, information-preserving,
and idempotent. Approval applies only to objects with the same proven owner,
schema, transformation, and normalized divergence fingerprint.

The first object or class with semantic divergence, ambiguous ownership, or no
approved rule stops the workflow before every publication. Preserve it in place
and present one operator decision packet containing its identity and paths,
ownership and provenance evidence, normalized semantic delta, affected consumers
and blast radius, proposed disposition, destructive effects, recovery contract,
and fixed-point validation plan. Do not overwrite, delete, quarantine, back up,
stash, roll back, or silently promote it to a source while that decision is open.

See also: `strict-execution.md` (rule file) — aggregate parent policy.

## Two producers reaching one conclusion collapse; two disagreeing stop

Independent producers legitimately reach the same object. One derives it from
what a previous run left behind, another from what this run planned; each is
correct on its own, and neither is redundant, because either can be the only
one that sees a given case.

Declare the identity key that makes them the same conclusion, and collapse on
it: the effect carries the object once. A consumer that accepts an object only
once refuses the whole batch over a repeat, so a duplicate is not cosmetic.

Identical key with divergent payload is not a duplicate. It means the producers
disagree about what the conclusion is, and that is an operator adjudication:
fail with both payloads and the key. Never take the first, the last, the
newest, or the merge — a rule that silently picks makes the disagreement
invisible exactly when it matters.

See also: `strict-execution.md` (rule file) — aggregate parent policy.
