<!-- AUTO-GENERATED FILE — regenerate through `make gen` from the workspace root. -->
<!-- Source of truth: `<workspace-root>/docs/guides/using-pydantic.md`; adjust that workspace source, never this member projection. -->

# flext-core - Using Pydantic in FLEXT

> Project profile: `flext-core`

<!-- TOC START -->

- [Authority](#authority)
- [Where each Pydantic concern lives](#where-each-pydantic-concern-lives)
- [Library boundaries](#library-boundaries)
- [What consumers never do](#what-consumers-never-do)
- [Enforcement](#enforcement)

<!-- TOC END -->

## Authority

The canonical, source-cited law is the `pydantic-development` skill
(`~/agents/skills/technology/pydantic-development/`, distributed to every agent home)
and the `rules/python/pydantic.md` rule. This guide maps that law onto the FLEXT
facades; it does not restate it. The Pydantic version in force is the one locked in
the committed `uv.lock` of the project that installs it; member locks and the
workspace lock that composes them agree by construction
(ADR-030).

Two 2026-10-25 operator rulings bind every FLEXT model (root `AGENTS.md`
"Pydantic law"; ADR-028 Decisions 14 and 19):

- models import `u` only from the upstream FLEXT project or, in very restricted
  cases, from a private `_utilities` class of their own namespace; they export no
  methods, and the Pydantic surface they need lives in `_models/pydantic.py`
  (tracker memory `operator-ruling-2026-10-25-models-u-imports-pydantic-home`);
- entity equality is Pydantic value equality with explicit identity by id
  (tracker memory `operator-ruling-2026-10-25-entity-equality-pydantic-way`; see
  Domain models).

## Where each Pydantic concern lives

| Concern | FLEXT owner | Consumer form |
| --- | --- | --- |
| Model configuration | `m.*` presets in `flext-core/src/flext_core/_models/_base/` | `class X(m.FrozenModel)` — never a raw base or a hand-written `ConfigDict` |
| Reusable constraints, validators, serializers | `t` aliases (`type X = Annotated[...]`) | `field: t.NonEmptyStr` |
| Closed domains | `StrEnum` classes in `c` | `kind: c.Shipping.EventKind`; union tags `Literal[c.Shipping.EventKind.CREATED]` |
| Dependency-typed fields | `@runtime_checkable` protocols in `p` | `sink: t.Port[p.ShipmentSink]` (`InstanceOf`) |
| Model protocols | generated into `p` by flext-infra during `make gen` | never hand-mirrored |
| Non-model adapters | module-level adapters owned by `u` / `t` | never `TypeAdapter(...)` inside a function |
| Outcomes | `r` / `p.Result[T]`; the `e`/`u` result boundary converts `ValidationError` | `u.model.validate_value(Model, raw, from_json=True)` |
| Runtime knobs | `settings.<Namespace>.*` (pydantic-settings) | read through the singleton, never `os.environ` |
| Static rules | `config.<Namespace>.*` from `config/*.yaml` | read through the singleton |

Only flext-core imports `pydantic`, `pydantic_settings` or `pydantic_core`.

## Library boundaries

Every payload a library returns enters FLEXT as an `m` model at the single adapter
that calls the library, and every payload FLEXT hands to a library leaves through
`model_dump` / `model_dump_json` at that adapter:

- JSON: `Model.model_validate_json(raw)`.
- YAML/TOML: the loaded value goes straight into `Model.model_validate(...)` or a
  module-level adapter — no `cast`, no `isinstance` walk.
- pydantic-settings custom sources return the mapping its `_read_file` contract
  requires; the settings model validates it.
- structlog processors receive a mapping built from the validated record in one place.

## What consumers never do

Re-validate a model they already received, `isinstance`/`cast` owned data, convert
between dict and model by hand, compare closed domains with string literals, hold a
second model for a shape an existing model already expresses, or call
`model_rebuild`, `model_construct` or `model_copy(update=...)` on validated data.

## Enforcement

Each banned form has an ast-grep rule with fixtures in
`flext-infra/src/flext_infra/config/rules/ast-grep/rules/`. `make mod` applies the
rewrites, `make check` fails on any remaining finding, and tests exercise models only
through the public facades (no mocks, private access or tautologies).
