<!-- AUTO-GENERATED FILE — regenerate through `make gen APPLY=Y` from the workspace root. -->
<!-- Source of truth: `docs/guides/development.md`; adjust that source, never this projection. -->

# flext-core - Development

<!-- TOC START -->
- [Start at the workspace root](#start-at-the-workspace-root)
- [Forward workflow](#forward-workflow)
- [Architecture and tests](#architecture-and-tests)
- [Documentation ownership](#documentation-ownership)
- [Related guides](#related-guides)
<!-- TOC END -->

> Project profile: `flext-core`

The root `AGENTS.md`, branch-matched `flext-law`, nearest package scope, and
active Bead define the development contract. This page is the executable
summary, not a second policy owner.

## Start at the workspace root

Discover and prepare the declared command surface before changing code:

```bash
make setup APPLY=Y
make help
```

Use only verbs printed by `make help`. Do not add project, file, pattern,
changed-only, fix, or phase selectors to narrow a standard verb.

## Forward workflow

1. Read the canonical config, settings, generator, public facade, consumers,
   tests, and docs for the bounded change.
2. Use the semantic refactoring owner instead of manually rewiring consumers.
3. Remove the superseded owner in the same change and prove zero residue.
4. Regenerate every managed projection from its source.
5. Run the native gates without bypassing their orchestration.

```bash
make gen APPLY=Y
make mod APPLY=Y
make gen APPLY=Y
make gen APPLY=Y
make fix APPLY=Y
make fmt APPLY=Y
make check APPLY=Y
make test APPLY=Y
make conform APPLY=Y
```

The final generation run proves the fixed point. The test verb always retains
and uses Testmon; a direct test-runner invocation is invalid evidence.

## Architecture and tests

- Generic reusable behavior survives in canonical `c`, `t`, `p`, `m`, or `u`
  ownership, with runtime behavior in `u`, services, `api.py`, or `cli.py`.
- Tests use public facades, `tm`, the unified `conftest.py`, and typed shared
  fixtures.
- Mocks, fakes, stubs, patching, private construction, duplicated setup, and
  hardcoded project-owned values are prohibited.
- Failures and warnings are corrected at their owning source, never suppressed,
  retried, normalized, or bypassed.

## Documentation ownership

Root guides are the writable source for generated member guides. Update the
root source, then use `make gen APPLY=Y`; never hand-edit generated copies.

## Related guides

- Getting started
- Configuration
- Testing
- Troubleshooting
