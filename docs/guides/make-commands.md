<!-- AUTO-GENERATED FILE — regenerate through `make gen APPLY=Y` from the workspace root. -->
<!-- Source of truth: `docs/guides/make-commands.md`; adjust that source, never this projection. -->

# flext-core - FLEXT Make Commands

<!-- TOC START -->
- [Discover commands](#discover-commands)
- [Canonical workflow](#canonical-workflow)
- [Test contract](#test-contract)
- [Failure contract](#failure-contract)
- [Scope and generation](#scope-and-generation)
- [Related guides](#related-guides)
<!-- TOC END -->

> Project profile: `flext-core`

`make help` at the workspace root is the executable authority for command
grammar. This guide records the invariants that every declared verb must keep.

## Discover commands

```bash
make setup APPLY=Y
make help
```

Never infer a target, flag, or selector from historical documentation. When a
required verb is missing or broken, repair the root dispatcher owner and rerun
that verb.

## Canonical workflow

Use the standard verbs directly from the workspace root:

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

The final generation pass proves the fixed point. `APPLY=Y` is the sole mutation
flag. No project, file, pattern, action, phase, fix, or changed-only selector may
be attached to a standard verb.

## Test contract

Every test execution uses `make test APPLY=Y`. The verb owns impact selection and
the retained Testmon cache, including complete-suite requests. Direct test-runner
commands and cache-clearing bypasses are prohibited.

## Failure contract

- The first exception, traceback, and non-zero exit propagate unchanged.
- Warnings, skips, empty output, and missing tools are failures.
- No retry, fallback, suppression, normalization, partial run, or alternate raw
  tool path can replace the canonical verb.

## Scope and generation

The root dispatcher resolves workspace scope from its typed topology. Generated
Make surfaces and documentation are changed at their template or configuration
owner, then regenerated with `make gen APPLY=Y`.

## Related guides

- Development
- Testing
- Getting started
