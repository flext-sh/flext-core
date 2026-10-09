<!-- AUTO-GENERATED FILE — regenerate through `make gen` from the workspace root. -->
<!-- Source of truth: `<workspace-root>/docs/guides/testing.md`; adjust that workspace source, never this member projection. -->

# flext-core - Testing

> Project profile: `flext-core`

<!-- TOC START -->

- [Test design](#test-design)
- [Canonical execution](#canonical-execution)
- [Generated documentation](#generated-documentation)
- [Related guides](#related-guides)

<!-- TOC END -->

FLEXT tests prove observable runtime behavior through public package facades. The
workspace root `AGENTS.md` and the nearest package scope remain authoritative.

## Test design

- Exercise only public `api.py` surfaces and canonical `c`, `t`, `p`, `m`, and `u`
  facades.
- Put shared setup in the unified `conftest.py` and typed fixtures under
  `tests/fixtures/`.
- Use `tm` matchers and shared `flext-tests` builders for assertions and test data.
- Read project-owned values from typed config or settings. Never freeze current defaults
  in tests, examples, or golden files.
- Use real, bounded dependencies. Mocks, fakes, stubs, patching, monkeypatch mutation,
  and assertions about private construction are prohibited.
- Treat warnings, skips, empty collection, and suppressed failures as red.

## Canonical execution

Run tests only through the dispatcher at the workspace root:

```bash
make test
```

`make test` owns incremental impact selection and the project's persistent Testmon
database, whose location the flext-infra generated configuration owns. Its collection
inventory uses the same marker scope as execution. The runner accounts for every
selected and deselected test; it never infers selection from console output. Never clear
or bypass the database, or invoke the underlying runner directly.

Run the complete suite through its declared verb:

```bash
make test-full
```

The runner first completes the incremental operation, then executes the full suite using
the same database and one monotonic deadline. The first failure stops the sequence. The
full phase includes both configured `external-gate-markers` and `ci-excluded-markers` in
every context. External tests retain their declared services, network access, and
authentication requirements.

Incremental execution excludes `tooling.tools.pytest.external-gate-markers`. CI and
generated pre-commit hooks use the configured `make.ci.value` token and also exclude
`ci-excluded-markers`, consistently in collection, execution, and coverage. The runner
records these as `not_executed_external_gates` and `not_executed_ci_markers`; exclusions
are not passed tests. Both fields are empty for the full phase. Marker policy belongs to
the typed tooling configuration, not a separate command-line expression.

Each phase retains its mode, database, raw process outcome, collection manifest, and
diagnostics. The latest receipt names the current attempt even when collection fails.
Warning totals include selection, inventory, and suite occurrences, with blocking and
explicitly suspended warnings reported separately. The existing non-strict MRO
enforcement suspension remains visible in those receipts; skips still block acceptance.

Zero execution is accepted only as a typed incremental `cache_hit`: the database must
pass integrity checks, a complete nonempty inventory must be entirely deselected, and
there must be no failures, blocking warnings, or skips. A cache hit is never reported as
tests passed. Empty collection or zero execution in the full phase fails.

Run the complete verification gate through the same dispatcher:

```bash
make check
```

Selectors such as project names, file names, patterns, or changed-only flags are not
part of this command surface. If a required workflow is missing, repair the root Make
owner and rerun its declared verb.

## Generated documentation

Member copies of this guide are generated projections. Change this root source and
regenerate from the workspace root:

```bash
make gen
```

Do not edit a member projection by hand.

## Related guides

- Development
- Troubleshooting
- Testing standards
