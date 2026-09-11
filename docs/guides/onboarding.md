<!-- AUTO-GENERATED FILE — regenerate through `make gen` from the workspace root. -->
<!-- Source of truth: `docs/guides/onboarding.md`; adjust that source, never this projection. -->

# flext-core - Onboarding

> Project profile: `flext-core`

<!-- TOC START -->
- [Resolve authority](#resolve-authority)
- [Prepare and discover](#prepare-and-discover)
- [Establish the baseline](#establish-the-baseline)
- [Change safely](#change-safely)
- [Generated surfaces](#generated-surfaces)
- [Command grammar](#command-grammar)
- [Related guides](#related-guides)
<!-- TOC END -->

Use this sequence before changing any FLEXT package. All actions start at the
workspace root.

## Resolve authority

Read, in order:

1. the workspace root `AGENTS.md`;
2. the branch-matched `flext-law` skill;
3. the nearest package `AGENTS.md`;
4. the active Bead and its current evidence.

Confirm the package's canonical config, settings, public API, and owned `c`, `t`,
`p`, `m`, and `u` surfaces before adding or moving a symbol.

## Prepare and discover

```bash
make setup
make help
```

The workspace virtual environment and root dispatcher are the only command
surface. Never enter a member directory to run a parallel tool command.

## Establish the baseline

```bash
make gen
make fix
make fmt
make check
make test
make conform
```

Tests run only through the retained Testmon cache. Warnings, skips, empty
collection, missing tools, and suppressed failures are red.

## Change safely

- Use semantic refactoring automation for hierarchy discovery and rewiring.
- Put generic reusable behavior in canonical `c`, `t`, `p`, `m`, or `u`
  ownership.
- Exercise only public facades with `tm`, the unified `conftest.py`, and typed
  shared fixtures.
- Do not use mocks, fakes, stubs, patching, monkeypatch mutation, private
  construction, or hardcoded project values.
- Remove the old owner after every consumer is rewired; leave no compatibility
  path or duplicate registry.

## Generated surfaces

Change the source owner, then regenerate and prove the fixed point:

```bash
make gen
make mod
make gen
make gen
```

Generated member guides identify their root source and exact regeneration rule.
Never edit those projections by hand.

## Command grammar

Do not add project, file, pattern, phase, fix, or changed-only selectors to the
standard verbs. If `make help` does not expose a required workflow, repair its
canonical Make owner before continuing.

## Related guides

- Getting started
- Development
- Configuration
- Testing
- Using flext-core
- Using flext-cli
- Using flext-tests
