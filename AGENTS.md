# AGENTS.md — flext-core

> **Parent workspace law** lives in [`../AGENTS.md`](../AGENTS.md) — read it first.
> Universal engineering core: `~/.agents/UNIVERSAL_CORE.md`. Composition: global skills + parent/root `AGENTS.md` + this
> scope delta. Do not re-embed universal law.
>
> **Standalone / independent mode:** when `../AGENTS.md` does not resolve, pin the parent raw `AGENTS.md` URL to the
> same branch/release as this package (never `main`).

<!-- AIHUB-AGENTS-SCOPE-LOCAL-BEGIN -->
**Package:** `flext_core` · ~30.9k src LOC · deps: **none** (foundation of the dependency graph)

## Overview

Enterprise Foundation Framework (Python 3.13 + Clean Architecture). Defines the facade alphabet every other `flext-*`
package composes via MRO. Depends on nothing; must never import another `flext-*` package. No public `api.py` — this
package *is* the foundation.

## Structure

```text
src/flext_core/
├── constants.py models.py protocols.py typings.py utilities.py  # facet roots (c/t/p/m/u)
├── result.py exceptions.py mixins.py handlers.py decorators.py  # r / e / x / h / d
├── service.py container.py context.py dispatcher.py registry.py # DI + CQRS (s)
├── runtime.py loggings.py lazy.py _config.py _settings.py
├── _constants/ _models/ _protocols/ _typings/ _utilities/ …_parts/
└── __init__.py  # AUTO-GENERATED lazy export map
```

## Code Map

| Symbol | Kind | Location | Role |
| --- | --- | --- | --- |
| `FlextTypes` (`t`) | class | `typings.py` | composite type aliases |
| `FlextConstants` (`c`) | class | `constants.py` | constants facade |
| `FlextProtocols` (`p`) | class | `protocols.py` | structural protocols |
| `FlextModels` (`m`) | class | `models.py` | Pydantic-2 models |
| `FlextResult` (`r`) | class | `result.py` | railway result (ADR-001) |
| `FlextContainer` | class | `container.py` | DI container |
| `FlextService` (`s`) | class | `service.py` | singleton service base |

## Conventions (specific to this package)

- `_config.py` / `_settings.py` are layer-0 pure (stdlib + pydantic only); they own `config` / `settings` singletons.
- `lazy.py` (`FlextLazy`) builds the PEP-562 export map used by every generated `__init__`.
- Never hand-edit `__init__.py` or `# AUTO-GENERATED` facet roots; do not collapse `_parts` without updating facade
  imports (breaks lazy map / cycles).
- Config/settings canonical pattern: ADR-012.
- Codemod governance (ast-grep + make mod): ADR-014.

## Commands

```bash
make check PROJECT=flext-core
make test  PROJECT=flext-core
```

<!-- AIHUB-AGENTS-SCOPE-LOCAL-END -->
