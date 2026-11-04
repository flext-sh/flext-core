<!-- AUTO-GENERATED FILE — regenerate through `make gen` from the workspace root. -->
<!-- Source of truth: `<workspace-root>/docs/guides/configuration.md`; adjust that workspace source, never this member projection. -->

# flext-core - Configuration

> Project profile: `flext-core`

<!-- TOC START -->

- [Ownership order](#ownership-order)
- [Connect configuration to domain models](#connect-configuration-to-domain-models)
- [Public exports manifest](#public-exports-manifest)
- [Documentation configuration](#documentation-configuration)
  - [Build API documentation with MkDocs and mkdocstrings](#build-api-documentation-with-mkdocs-and-mkdocstrings)
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

Config namespace schemas belong to layer zero, as defined by ADR-005 section 2.
Their Pydantic foundations must not import their own aggregate model facade,
directly or through an own-package `_models` leaf. An authenticated upstream
Core model facade remains the Pydantic SDK owner for other FLEXT packages; it is
not that package's unfinished facade. Preserve the schema's fields, validation,
frozen/extra policy and defaults when moving its owner. A public model identity
with consumers retains its route through an authored family holder composing the
single layer-zero declaration by MRO. Helpers are admitted only by native identity
and a complete acyclic candidate graph without facade dependencies, never by their
filename. Generated facades and lazy exports follow the source owners through their
existing publisher; schemas are never copied or replaced with re-export shims.

Layer-zero schema moves preserve model configuration and construction behavior,
not just fields. A preset with a schema-completion callback cannot be replaced by
plain Pydantic solely by copying its configuration flags: the callback's owner and
layer-zero applicability require an explicit architectural decision first. Native
JSON validation remains distinct from strict Python validation when YAML sequences
must populate tuple fields. Config singleton allocation, identity equality and
hashing remain with the config owner when schemas move to a pure helper.

## Connect configuration to domain models

`config`, `settings` and `m` all use typed validation, but their roles differ:

| Owner | What it represents | How it changes |
| --- | --- | --- |
| `config` | Reviewed rules, catalogs and generation policy | Governed data edit, owner validation and regeneration |
| `settings` | Deployment/process inputs such as the environment or CLI | The settings owner's declared sources and validated update/clone lifecycle |
| `c` | Immutable vocabulary, enum members and invariants | A deliberate domain-contract change, not an operator knob |
| `m` | Data crossing a domain or adapter boundary | Construction or an authorized validated state transition |

The settings owner is the single source reader. Preserve its prefixes, aliases,
source precedence and declared defaults; an invalid supplied value is an error,
not a request to try a default or another source. A validated settings snapshot
may be immutable while its owner supports runtime reconfiguration: runtime
adjustability does not authorize direct field assignment or singleton replacement
by a consumer. Do not copy the source precedence into services or tests.

The composition root reads validated policy and settings and passes the required
values or typed collaborators into the domain. Domain model declarations do not
import a live config/settings singleton merely to calculate a field default.
An endpoint or timeout cannot become a `c` constant just because a model needs it.
One value has one owner; the `t` alias or `p` capability describes its contract
without introducing another data source.

Layer-zero schemas are declared without depending on their own facade being
assembled. The rest of the package consumes the resulting validated singleton
through its public owner. This bootstrap rule and the family import order are
different from permission to create another config loader, resolver or registry.
Catalog entries remain typed collections; a new valid entry must not require a
new property, protocol member and orchestration branch merely to transport it.

Keep `model_config` policy at its preset owner. The latest enum target is
`use_enum_values=False`: preserve typed members internally and serialize native
values at the actual outward boundary. This is independent of `strict`, which
governs accepted inputs. Do not make all settings strict as a side effect of this
enum decision: environment decoding and the validated settings contract retain
their source-specific semantics.

For an authenticated textual document, the accepted ingress extension is
`u.Cli.config_decode(data, model)`. It validates through the model's native
Pydantic context rather than reparsing a file. Fields requiring representation
decoding declare `t.DocumentInput[T]`: enum tokens use native strict string
validation, and YAML arrays use a native list/tuple grammar before the original
strict tuple schema validates arity and items. Typed enum and tuple defaults
remain valid. Without document context, the same model retains its ordinary
strict Python constructor contract. This extension never serializes parsed YAML
back to JSON, loosens a field, or adds a conversion in a consumer. Its native
runtime and gate acceptance remain to be proven.

Across packages, compose upstream same-letter facades and add only the package's
own domain namespace. Do not rename an upstream namespace into a local one or
mix its records into another bounded context. Python MRO selects class members;
Pydantic's effective model configuration and the configured source precedence
must each be verified independently. Neither a deeper MRO nor a frozen wrapper
proves that the right config/settings source was loaded.

See Using Pydantic,
ADR-005
and ADR-019 for
the complete owner, model and DI contracts. The canonical settings behavior is
documented in the [versioned pydantic-settings source](https://github.com/pydantic/pydantic-settings/blob/v2.16.1/docs/index.md).

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

### Build API documentation with MkDocs and mkdocstrings

The Infra dependency catalog already declares MkDocs and its Python docstring
handler. Provision them through the physical environment elected by `make setup`;
do not install a separate documentation environment with `pip`. Dependency changes
belong to the existing catalog and `make upg`, subject to ADR-030's lock guard.

1. Write the public API's docstrings in its source module. Use Google-style sections
   such as `Args`, `Returns` and `Raises`, documenting observable behavior.
2. Put authored explanations in `docs/guides/`. The existing API planner derives
   public module pages and their `::: package.module` directives from source owners.
   A directive selects the real module; it does not copy its signature or docstring.
3. Change documentation policy at its typed configuration owner and the shared
   `mkdocs_macros.j2` template. Root/member templates consume that macro; never edit
   generated `mkdocs.yml` or duplicate the plugin block in a member.
4. Generate through the root `make gen`, complete the native validation cycle, and
   build through `make build`. The build uses strict MkDocs configuration and
   publishes its local site under the generated `site_dir`.

The approved plugin configuration places the docstring style under `options`:

```yaml
plugins:
  - mkdocstrings:
      handlers:
        python:
          options:
            docstring_style: google
            show_signature_annotations: true
            merge_init_into_class: true
```

This is a fragment of the generated configuration, not a replacement for its
other plugins or source paths. The discovery owner supplies the handler's `paths`,
including the actual workspace/member topology. Site title, URLs and package names
remain derived from metadata. See the
[native handler configuration](https://mkdocstrings.github.io/python/usage/configuration/docstrings/).

The existing and approved artifact layout is:

```text
src/<package>/                 # source and docstrings
docs/
  index.md
  guides/                      # authored guides
  api-reference/generated/     # derived public API directives
  reference/dependencies/      # approved offline references and provenance
  llms.txt
  llms-full.txt
mkdocs.yml                     # generated configuration
.reports/docs/site/            # local build output
```

The offline references, full llms context and published CRG wiki are accepted
targets in ADR-018, not an
assertion that acquisition or publication is already active. CRG creates its own
wiki; AI Hub authenticates its publication and Infra consumes that artifact in
the same documentation inventory. Do not copy the graph storage into `docs/`.

Run each public command separately with the operator's timeout and the project's
environment. For example:

```bash
rtk proxy timeout --signal=TERM --kill-after=5s 60s direnv exec . make setup
rtk proxy timeout --signal=TERM --kill-after=5s 60s direnv exec . make gen
rtk proxy timeout --signal=TERM --kill-after=5s 60s direnv exec . make build
```

Generation/build do not replace the required format, fix, mod, check and test
cycle. Verify rendered API signatures, docstrings, cross-references and local
llms artifacts with the real site consumer before claiming documentation delivery.

## Apply and validate

Run configuration propagation and validation from the workspace root:

```bash
make gen
make gen
make check
make tests
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
