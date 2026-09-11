<!-- AUTO-GENERATED FILE — regenerate through `make gen` from the workspace root. -->
<!-- Source of truth: `docs/guides/configuration.md`; adjust that source, never this projection. -->

# flext-core - Configuration

> Project profile: `flext-core`

<!-- TOC START -->
- [Ownership order](#ownership-order)
- [Documentation configuration](#documentation-configuration)
- [Apply and validate](#apply-and-validate)
- [Related guides](#related-guides)
<!-- TOC END -->

Configuration has one writable authority. Prefer typed `config/*.yaml` and
settings; use `pyproject.toml` only for package and tool metadata that it owns.
Derived files are generated projections.

## Ownership order

1. Typed config and settings own business rules, operational values, and
   environment-tunable behavior.
2. `pyproject.toml` owns package metadata and declared tool configuration.
3. Generators derive managed code, docs, CI, and workspace projections.

Never duplicate an owned value in tests, examples, JSON side files, templates,
or local registries. Tests read the same typed owner as production.

## Documentation configuration

Public API documentation is derived from declared public exports and docstrings.
Project descriptions, versions, package names, and URLs come from canonical
package metadata. Docs-only policy exists only when it cannot be derived from a
typed owner.

Root files under `docs/guides/` own generated member guides. Change the root
source, never the member projection.

## Apply and validate

Run configuration propagation and validation from the workspace root:

```bash
make gen
make gen
make check
make test
make conform
```

The second generation run must be a fixed point. Test validation retains the
canonical Testmon cache. Do not invoke underlying tools or add project, file,
pattern, phase, fix, or changed-only selectors.

Warnings, missing tools, stale projections, and empty output are failures. Fix
their canonical owner and rerun the same root verb.

## Related guides

- Development
- Testing
- Troubleshooting
