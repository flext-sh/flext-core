<!-- AUTO-GENERATED FILE — regenerate through `make gen` from the workspace root. -->
<!-- Source of truth: `<workspace-root>/docs/guides/development.md`; adjust that workspace source, never this member projection. -->

# flext-core - Development

> Project profile: `flext-core`

<!-- TOC START -->

- [Start at the workspace root](#start-at-the-workspace-root)
- [Forward workflow](#forward-workflow)
- [Architecture and tests](#architecture-and-tests)
- [Documentation ownership](#documentation-ownership)
- [Related guides](#related-guides)

<!-- TOC END -->

The root `AGENTS.md`, branch-matched `flext-law`, nearest package scope, and active Bead
define the development contract. This page is the executable summary, not a second
policy owner. Resolve them through the governance router, and
record its engineering conformance review
for every bounded implementation slice.

## Start at the workspace root

Discover and prepare the declared command surface before changing code:

```bash
make setup
make help
```

Use only verbs printed by `make help`. Do not add project, file, pattern, changed-only,
fix, or phase selectors to narrow a standard verb.

## Forward workflow

1. Read the canonical config, settings, generator, public facade, consumers, tests, and
   docs for the bounded change.
2. Use the semantic refactoring owner instead of manually rewiring consumers.
3. Remove the superseded owner in the same change and prove zero residue. Delete
   without fear: Git history recovers tracked code, and excessive, wrong or duplicated
   code and executions are defects wherever they are found (global `zero-residue` rule).
4. Regenerate every managed projection from its source.
5. Run the native gates without bypassing their orchestration.

```bash
make gen
make mod
make gen
make gen
make fix
make fmt
make check
make tests
make gen
```

The final generation run proves the fixed point. The test verb always retains and uses
Testmon; a direct test-runner invocation is invalid evidence.

## Architecture and tests

- Compose facade MRO in `c -> t -> p -> m -> u` order. Declaration layers remain
  data-only; behavior belongs in `u`, reusable bases, services, `api.py`, or `cli.py`.
  Use canonical Pydantic presets and typed ports, not model-local algorithms.
- Search and reuse domain-owned public helpers first. SSOT, YAGNI, DRY, and SOLID
  prohibit speculative abstractions and parallel/divergent implementations.
- Inject `p` ports at the `api` composition root; keep Clean Architecture dependencies
  inward and config/settings at layer zero, with no runtime import of build tooling.
- Tests use public facades, `tm`, the unified `conftest.py`, and typed shared fixtures.
- Mocks, fakes, stubs, patching, private construction, duplicated setup, and hardcoded
  project-owned values are prohibited.
- Failures and warnings are corrected at their owning source, never suppressed, retried,
  normalized, or bypassed.

## Documentation ownership

Root guides are the writable source for generated member guides. Update the root source,
then use `make gen`; never hand-edit generated copies.

## Related guides

- Getting started
- Configuration
- Testing
- Troubleshooting
