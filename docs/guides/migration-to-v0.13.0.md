<!-- AUTO-GENERATED FILE — regenerate through `make gen APPLY=Y` from the workspace root. -->
<!-- Source of truth: `docs/guides/migration-to-v0.13.0.md`; adjust that source, never this projection. -->

# flext-core - Migration to v0.13.0

> Project profile: `flext-core`

<!-- TOC START -->
- [Purpose](#purpose)
- [Migration Order](#migration-order)
- [Track 1: flext-core](#track-1-flext-core)
- [Track 2: Platform Packages](#track-2-platform-packages)
- [Track 3: Domain Packages](#track-3-domain-packages)
- [Track 4: Integration Packages](#track-4-integration-packages)
- [Core Rename Table](#core-rename-table)
- [Method Replacement Table](#method-replacement-table)
- [Removed Concepts](#removed-concepts)
- [Taxonomy Migration Checklists](#taxonomy-migration-checklists)
  - [Tests](#tests)
  - [Examples](#examples)
  - [Scripts](#scripts)
- [Done Criteria](#done-criteria)
<!-- TOC END -->

## Purpose

This guide describes how to migrate the FLEXT workspace to the `0.13.0` platform baseline.

It is organized by project category so teams can move in a controlled order without reintroducing the patterns being
removed.

## Migration Order

Migrate in this order:

1. `flext-core`
2. platform packages
3. domain packages
4. integrations

## Track 1: flext-core

Required actions:

- add `di.py` with `FlextDi`
- replace `registry.py` with `catalog.py`
- replace `handlers.py` with `handler.py`
- move `FlextLogger` from `loggings.py` to `logger.py`
- remove public `x` responsibilities
- narrow `FlextRuntime` to normalization and validation only
- narrow `FlextContext` to execution context only
- narrow `d` to the forward decorator set
- reshape `FlextContainer` around `add_service/add_factory/add_resource`
- reshape `FlextDispatcher` to absorb handler registration
- keep public data boundaries on Pydantic v2 `m.BaseModel` subclasses
  validated with `model_validate(...)`; do not add `TypedDict`,
  `dataclass`, `NamedTuple`, or ad-hoc dictionary payload contracts in
  governed `src/` code

## Track 2: Platform Packages

Projects:

- `flext-cli`
- `flext-api`
- `flext-auth`
- `flext-web`
- `flext-grpc`
- `flext-observability`
- `flext-plugin`
- `flext-meltano`
- `flext-quality`

Required actions:

- consume the new `flext-core` runtime classes only
- stop creating local runtime primitives
- move reusable helpers into `u`
- split extension storage from execution logic
- rename public extension surfaces to direct nouns

Expected target names:

- `FlextCliCommands`
- `FlextCliOptions`
- `FlextApiComponents`
- `FlextAuthProviders`
- `FlextPlugins`

## Track 3: Domain Packages

Projects:

- `flext-ldap`
- `flext-ldif`
- `flext-db-oracle`
- `flext-oracle-wms`
- `flext-oracle-oic`

Required actions:

- remove generic registry usage
- replace hybrid extension abstractions with direct domain names
- consume `FlextCatalog` only when the project truly has extension storage
- remove local plugin storage where no real extension contract exists

Expected target names:

- `FlextLdifServers`
- `FlextDbOracleExtensions` only if a real extension contract remains

## Track 4: Integration Packages

Projects:

- all `flext-tap-*`
- all `flext-target-*`
- all `flext-dbt-*`

Required actions:

- stop touching platform primitives directly
- consume project facades and local aliases
- remove architecture-specific naming drift
- adopt the workspace taxonomy for tests, examples, and scripts

## Core Rename Table

| Current                   | Target                              | Action                                                                   |
| ------------------------- | ----------------------------------- | ------------------------------------------------------------------------ |
| `FlextRegistry`           | `FlextCatalog` or `FlextDispatcher` | Replace based on actual role                                             |
| `h`                       | `FlextHandler`                      | Rename and narrow to a single handler contract                           |
| `loggings.py`             | `logger.py`                         | Move logger to direct file name                                          |
| `x`                       | removed                             | Move retained behavior into service, handler, logger, decorators, or `u` |
| hybrid runtime DI helpers | `FlextDi`                           | centralize all `dependency_injector` bridge logic                        |

## Method Replacement Table

| Current method or pattern                    | Replacement                                              |
| -------------------------------------------- | -------------------------------------------------------- |
| `register(kind=...)`                         | `add_service`, `add_factory`, `add_resource`             |
| `register_handler(...)` on generic registry  | `add(...)` or `add_many(...)` on `FlextDispatcher`       |
| `register_plugin(...)` on generic registry   | `add(...)` on `FlextCatalog`                             |
| `get_plugin(...)`                            | `get(...)` or `require(...)` on `FlextCatalog`           |
| `list_plugins(...)`                          | `list()` on `FlextCatalog`                               |
| `log_operation(...)`                         | `log(...)`                                               |
| `track_operation(...)`                       | `measure(...)`                                           |
| `with_context(...)`                          | `scope(...)`                                             |
| `combined(...)`                              | `compose(...)`                                           |
| direct `dependency_injector` use in app code | `u.get_*`, `u.require_*`, or `self.*` runtime properties |

## Removed Concepts

These concepts do not survive into the forward public architecture:

- hybrid public registries
- public mixin buckets as runtime primitives
- nested public namespaces for runtime services
- hidden DI spread through unrelated classes
- public compatibility layers
- numbered examples as part of the forward taxonomy
- structured domain payloads modeled as `TypedDict`, `dataclass`,
  `NamedTuple`, or loose dictionaries instead of Pydantic v2 models

## Taxonomy Migration Checklists

### Tests

- move example smoke tests out of `tests/unit/`
- create or use:
  - `tests/unit/`
  - `tests/integration/`
  - `tests/architecture/`
  - `tests/performance/`
  - `tests/fixtures/`
- remove suffixes:
  - `_cov`
  - `_real`
  - `_smoke`
- remove `tests/examples` except `tests/integration/examples`

### Examples

- keep executable examples only
- rename numbered examples to semantic names
- move helper code to `examples/support/`
- remove `examples/tests`
- remove helper models such as `models/exNN.py`

### Scripts

- reorganize scripts into:
  - `analysis/`
  - `migration/`
  - `validation/`
  - `maintenance/`
- move reusable code out of scripts and into governed packages

## Done Criteria

A migration wave is done when:

- the project no longer depends on `FlextRegistry`
- the project no longer depends on public `x`
- runtime bootstrapping is owned by `s`
- extension storage, if it exists, is stored in `FlextCatalog`
- governed `src/` code has no remaining `TypedDict`, `dataclass`,
  `NamedTuple`, or `namedtuple(...)` domain-model declarations
- public structured inputs and outputs are represented by facade Pydantic v2
  models and are validated at boundaries with `model_validate(...)` or
  `model_validate_json(...)`
- `tests/`, `examples/`, and `scripts/` follow the baseline taxonomy
- local docs point to the workspace baseline instead of describing conflicting architecture
