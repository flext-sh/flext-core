<!-- AUTO-GENERATED FILE — regenerate through `make gen APPLY=Y` from the workspace root. -->
<!-- Source of truth: `docs/guides/troubleshooting.md`; adjust that source, never this projection. -->

# flext-core - Troubleshooting

> Project profile: `flext-core`

<!-- TOC START -->
- [Establish the command surface](#establish-the-command-surface)
- [Repair at the owner](#repair-at-the-owner)
- [Documentation failures](#documentation-failures)
- [Test failures](#test-failures)
- [Related guides](#related-guides)
<!-- TOC END -->

Troubleshooting preserves the same command, ownership, and failure contracts as
normal development. Diagnose from the workspace root and keep the first raw
traceback or non-zero exit as the causal evidence.

## Establish the command surface

```bash
make help
```

Use the exact declared root verb. Do not invoke an underlying linter, type
checker, test runner, generator, or ad-hoc script, and do not attach project,
file, pattern, changed-only, fix, or phase selectors.

## Repair at the owner

- Generated output is evidence, not an edit target. Correct config, settings,
  templates, or generator ownership first.
- A broken or missing Make verb is a defect in the root dispatcher. Repair that
  owner before resuming the workflow.
- Do not catch, retry, fall back, normalize, suppress, skip, or partially apply a
  failing operation.
- Warnings, skipped checks, empty output, and missing tools remain failures until
  their owner is corrected.

After correcting the source, use only the applicable canonical verbs:

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

Test diagnosis still runs through `make test APPLY=Y`, with the retained Testmon
cache. Direct or cache-clearing test runs do not count as evidence.

## Documentation failures

For stale member guides, change the matching root file under `docs/guides/` and
run `make gen APPLY=Y`. Generated member guides carry their source and exact
regeneration command in the header; never patch those projections.

For a docs audit failure, remove the invalid command or test-double example at
the canonical root source. Do not add an allowlist or weaken the audit.

## Test failures

Tests must observe public facades and use `tm`, the unified `conftest.py`, and
typed shared fixtures. Replace mocks, fakes, stubs, patching, monkeypatch
mutation, private construction, and hardcoded project values at their owning
test infrastructure.

## Related guides

- Configuration
- Testing
- Development
