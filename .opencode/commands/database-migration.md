---
description: "Plan and execute one project-owned database migration through its canonical schema owner."
---

# Database migration

Treat `$ARGUMENTS` as the approved database change. Stop before mutation when
the database, current schema, desired invariant, migration owner, or deployment
constraints cannot be identified from the project.

1. Read project law, schema/migration configuration, migration history, live
   consumers, deployment order, backup/restore owner, and native database gates.
2. Inspect the current runtime schema through the project-owned interface. Do
   not infer state from tests, ORM types, or generated files alone.
3. Design one forward migration and, where the project supports it, its explicit
   reversal or recovery procedure. Model locks, long-running rewrites,
   backfills, nullability transitions, indexes, concurrent writers, and version
   compatibility without adding dual-read or dual-write residue.
4. Generate or author the migration through the declared owner, update every
   consumer and generated type, and remove superseded schema paths in the same
   change. Never edit a production database through an ad hoc client command.
5. Run the migration against an isolated representative database, validate data
   and constraints, exercise the real application path, and run all native
   migration, test, static, security, and packaging gates.

Report the observed starting schema, migration owner, forward/recovery behavior,
consumer rewiring, exact runtime evidence, and blockers. A partial migration,
skipped validation, unavailable backup owner, or failed command remains red.
