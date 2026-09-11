# Atomic configuration-schema migration

## Approval and impact

Before changing a public contract, present the exact old and proposed schemas,
affected consumers, persisted-data transformation, breaking behavior, and
deletion plan for operator approval. Internal changes may proceed only when the
repository proves they are not externally observable.

Inventory the typed schema owner, generators, writers, stored instances,
loaders, validators, runtime consumers, commands, tests, examples, and documents.
If every scoped consumer cannot move in the same change, stop before introducing
a second live contract.

For every stored or generated object in a mass cutover, apply the canonical
divergent-object preflight. Group objects only by proven owner, schema,
deterministic transformation, and normalized divergence fingerprint. Exact
final-format matches and approved information-preserving representation changes
may proceed automatically. A semantic delta, ambiguous owner, or class without
an approved rule preserves the current object and stops before every write with
an operator decision packet covering provenance, consumers, impact, disposition,
recovery, and fixed-point proof. A representative sample never authorizes the
remaining population.

## Cutover

1. Define one final schema and a deterministic transformation.
2. Make the transformation fail loudly on ambiguous or invalid input and make a
   second run on final-format input produce no change.
3. Change the typed owner and official migration surface.
4. Declare each deterministic calculated default once at its typed owner and
   remove equal environment variables, settings, parameters, arguments, and
   persisted fields. Require only current external values the owner cannot derive.
5. Transform persisted data with the repository's atomic storage primitive.
6. Rewire every producer and consumer and regenerate managed outputs.
7. Delete old fields, loaders, writers, fixtures, examples, documentation, and
   terminology. Final runtime readers explicitly reject old-format input.

Do not add aliases for removed fields, fallback readers, dual writes, feature
flags, deprecation windows, hardcoded translations, or ad hoc backup files.
Validate the complete source and destination before the first persisted effect;
the first invalid or ambiguous value raises unchanged and publishes nothing.

## Proof

Run the final-format runtime through every supported public entry point before
adjusting tests. Then prove valid final input, invalid and old input rejection,
information preservation, a second no-op migration, generated fixed point, and
the repository's static, behavioral, security, and build gates. Search the
scoped source tree for every removed key and term; any live match blocks closure.
