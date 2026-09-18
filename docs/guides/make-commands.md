<!-- AUTO-GENERATED FILE — regenerate through `make gen` from the workspace root. -->
<!-- Source of truth: `<workspace-root>/docs/guides/make-commands.md`; adjust that workspace source, never this member projection. -->

# flext-core - FLEXT Make Commands

> Project profile: `flext-core`

<!-- TOC START -->

- [Discover commands](#discover-commands)
- [Canonical workflow](#canonical-workflow)
- [Test contract](#test-contract)
- [Failure contract](#failure-contract)
- [Scope and generation](#scope-and-generation)
- [Related guides](#related-guides)

<!-- TOC END -->

`make help` at the workspace root is the executable authority for command
grammar. This guide records the invariants that every declared verb must keep.

## Discover commands

```bash
make setup
make help
```

Never infer a target, flag, or selector from historical documentation. When a
required verb is missing or broken, repair the root dispatcher owner and rerun
that verb.

## Canonical workflow

Use the standard verbs directly from the workspace root:

```bash
make setup
make gen
make mod
make gen
make gen
make fix
make fmt
make check
make test
make build
```

The consecutive generation passes prove the fixed point after structural
rewrites. `make build` packages the validated candidate; it does not replace
runtime verification. Each verb executes its declared operation directly. No
project, file, pattern, action, phase, fix, or changed-only selector may be
attached to a standard verb.

`make help` is the complete live inventory. Additional declared verbs such as
`deps`, `docs`, `audit`, `status`, `waza`, `duplication`, and the release verbs
retain their own single operation and are invoked only when their scope applies.

## Test contract

Every test execution uses `make test`. The verb owns impact selection and
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
owner, then regenerated with `make gen`.

## Related guides

- Development
- Testing
- Getting started
