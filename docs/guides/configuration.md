<!-- AUTO-GENERATED FILE — regenerate through `make gen` from the workspace root. -->
<!-- Source of truth: `<workspace-root>/docs/guides/configuration.md`; adjust that workspace source, never this member projection. -->

# flext-core - Configuration

> Project profile: `flext-core`

<!-- TOC START -->

- [Ownership order](#ownership-order)
- [Public exports manifest](#public-exports-manifest)
- [Documentation configuration](#documentation-configuration)
- [Apply and validate](#apply-and-validate)
- [Related guides](#related-guides)

<!-- TOC END -->

Configuration has one writable authority. Prefer typed `config/*.yaml` and settings; use
`pyproject.toml` only for package and tool metadata that it owns. Derived files are
generated projections.

## Ownership order

1. Typed config and settings own business rules, operational values, and
   environment-tunable behavior.
2. `pyproject.toml` owns package metadata and declared tool configuration.
3. Generators derive managed code, docs, CI, and workspace projections.

Never duplicate an owned value in tests, examples, JSON side files, templates, or local
registries. Tests read the same typed owner as production.

## Public exports manifest

`config/exports.yaml` is the declarative SSOT for the workspace root's public export
contract (RC-B). When the manifest exists, the lazy-init filesystem scan becomes a
validator: any divergence between the manifest and the scan fails generation instead of
mutating silently.

- The manifest lists the names of the root package's static public export contract.
- Runtime lazy-alias machinery (single-letter facade aliases such as `d`, `e`, `h`, `r`,
  `x`) is not part of the static contract; the scan does not attribute those aliases to
  the root package and the manifest must not list them.
- Intentional public-API changes update the manifest in the same change; deleting a
  module without updating the manifest fails gen pointing at the orphaned entry.

## Documentation configuration

Public API documentation is derived from declared public exports and docstrings. Project
descriptions, versions, package names, and URLs come from canonical package metadata.
Docs-only policy exists only when it cannot be derived from a typed owner.

Root files under `docs/guides/` own generated member guides. Change the root source,
never the member projection.

## Apply and validate

Run configuration propagation and validation from the workspace root:

```bash
make gen
make gen
make check
make test
make gen
```

The second generation run must be a fixed point. Test validation retains the canonical
Testmon cache. Do not invoke underlying tools or add project, file, pattern, phase, fix,
or changed-only selectors.

Warnings, missing tools, stale projections, and empty output are failures. Fix their
canonical owner and rerun the same root verb.

## Related guides

- Development
- Testing
- Troubleshooting
