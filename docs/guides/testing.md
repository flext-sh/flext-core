<!-- AUTO-GENERATED FILE — regenerate through `make gen` from the workspace root. -->
<!-- Source of truth: `docs/guides/testing.md`; adjust that source, never this projection. -->

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

- Exercise only public `api.py` surfaces and canonical `c`, `t`, `p`, `m`, and
  `u` facades.
- Put shared setup in the unified `conftest.py` and typed fixtures under
  `tests/fixtures/`.
- Use `tm` matchers and shared `flext-tests` builders for assertions and test
  data.
- Read project-owned values from typed config or settings. Never freeze current
  defaults in tests, examples, or golden files.
- Use real, bounded dependencies. Mocks, fakes, stubs, patching, monkeypatch
  mutation, and assertions about private construction are prohibited.
- Treat warnings, skips, empty collection, and suppressed failures as red.

## Canonical execution

Run tests only through the dispatcher at the workspace root:

```bash
make test
```

The test verb owns test selection and the retained Testmon cache. Never clear or
bypass that cache, and never invoke the underlying test runner directly.

Run the complete verification gate through the same dispatcher:

```bash
make check
```

Selectors such as project names, file names, patterns, or changed-only flags are
not part of this command surface. If a required workflow is missing, repair the
root Make owner and rerun its declared verb.

## Generated documentation

Member copies of this guide are generated projections. Change this root source
and regenerate from the workspace root:

```bash
make gen
```

Do not edit a member projection by hand.

## Related guides

- Development
- Troubleshooting
- Testing standards
