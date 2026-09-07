# Python development procedure

## Preflight

Resolve project law, Python version, package/dependency and lock owners, source
layout, public APIs and consumers, generated owners, runtime, and canonical
format/lint/type/test/build gates before effects. Do not assume a manager, runner,
plugin, selector, cache mode, or raw command.

## Owner implementation

- Parse external input once into project-owned types; avoid `Any`, unsafe casts,
  ignores, and test-only values.
- Keep mutable state ownership explicit, functions cohesive, imports free of
  network/process/persistent effects, and resources under context managers.
- Let the first exception and traceback escape unchanged. Catch only for cleanup
  or rollback, attach secondary failure, and re-raise the original.
- Pass subprocess arguments as a sequence and propagate nonzero, timeout, and
  signal unchanged.
- Edit a schema/generator owner rather than its projection. Rewire consumers and
  remove dead, duplicate, fallback, compatibility, and old paths.

For a defect, capture expected/observed behavior, smallest reproducer, versions,
and exact exception; test falsifiable causes and correct the earliest wrong owner.
For tests, use the declared facade and complete owned selection. Zero collected or
executed tests is failure; never retry through another runner, clear/replace a
cache, reinstall tools, or turn warnings/skips into green.

When the project pins pytest-testmon, keep impact selection enabled for ordinary,
full, and CI runs; never add `--testmon-noselect` or disable Testmon to widen a
run. To force an explicit `FILE`, nodeid, or `MATCH`, resolve its exact nodeids,
reject an empty or owner-limit-exceeding set, invalidate only those rows in the
existing database transactionally, and run their intersection through
`--testmon-forceselect`. Never invalidate by broad pattern, guessed substring,
unbounded set, environment change, database replacement, or global cache clear.
Prove the same database identity survived, the invalidated count stayed within
the configured limit, Testmon selection remained active, and at least one
requested test executed.

Run public behavior before affected native gates. Missing evidence leaves zero
effects. Report material behavior, exact commands/exits/output, first cause,
executed scope, consumers rewired, and zero residue.
