# AGENTS.md — flext-core

<!-- BEGIN AI-HUB MANAGED UNIVERSAL CORE -->
<!-- UNIVERSAL-GOVERNANCE v4 -->

## Universal Agent Engineering Core

`~/.agents` is the sole universal authority. AI Hub distributes and configures
it but never competes with it. Project law may be stricter; the newest explicit
operator instruction prevails and lower authority must be reconciled.

1. **Truth with evidence.** Claims require the exact command, working directory,
   exit status, decisive output, and bounded scope.
2. **Research before mutation.** Read current authority, intent, owner Bead,
   implementation owner, consumers, generated projections, concurrent WIP, and
   validation route. Never invent behavior or results.
3. **One active intent.** Preserve the goal, target, Bead, exclusions, phase,
   required gates, and stop condition through delegation and continuation.
4. **Root cause and one owner.** Change the canonical owner and complete the
   cutover. No bypass, fallback, shim, suppression, hardcode, fake, duplicate
   route, silent default, or old-and-new coexistence.
5. **Fix forward.** Preserve shared work; never destructively discard unknown
   changes. Re-read mutable files and classify relevant paths and hunks.
6. **Typed and generated boundaries.** Parse untrusted input once into canonical
   types. Change sources, not projections; regenerate and prove idempotence.
7. **Continuous green.** No completion while the project or environment is
   broken, partially migrated, dirty from task WIP, ahead of remote, missing
   real-use QA, or carrying stale generated output or docs. Run native global
   and changed-scope gates; Python requires Ruff, Pyrefly, Pyright, Mypy, and
   Pytest coverage plus applicable build and integrated validation.
8. **Beads is execution truth.** Beads owns work, plans, memory, dependencies,
   status, evidence, and closure. GitHub is its continuous external coordination,
   PR, review, and CI mirror after the orchestrator organizes Beads completely.
9. **Separated roles.** The orchestrator coordinates, owns semantic Beads state,
   validates, approves or rejects merges, rolls out, and closes; it does not
   implement. Workers directly implement one Bead in one branch and worktree but
   never merge or close. The standing documenter continuously audits, updates,
   validates, and removes stale canonical skills, ADRs, docs, Python docstrings,
   examples, and executable snippets under the same validated PR flow; the
   governance/CI helper also remains active.
10. **No stall by reporting.** Five-minute status reports include the agent table
    and epic evolution and never pause execution. Compaction, continuation, and
    status transfer context only.
11. **Historical material is evidence only.** Archives, generated or tool homes,
    backups, sessions, caches, and legacy trees are never live authority.
12. **Stop only for a real blocker.** Ask one precise question only when authority
   conflicts or an action would be destructive; otherwise continue to the
   observable stop condition.
13. **Short validated slices.** Deliver in small, independently validated
   units that merge to the integration branch quickly — one Bead, one
   reviewable PR, hours not days. Mega-lanes and long-lived WIP are defects;
   the orchestrator splits any unit that cannot merge green within a session.
14. **Living documentation.** Project knowledge is durable, never rebuilt
   per session. On entering a project, read its docs first and validate key
   claims quickly against live reality. Every change that produces new
   understanding or behavior updates the affected docs in the SAME change;
   stale docs are defects filed as beads, never worked around.
15. **Tests reflect canonical reality.** Tests are executable checks of current
    behavior, never a source of truth; a test that violates canonical policy is
    corrected to match the policy, not accommodated. Performance optimization is
    evidence-first: profile with cProfile to find the hot path before changing
    anything, then optimize with the project's typed OO/MRO/lazy-import patterns;
    accelerate test selection with impact analysis (e.g. pytest-testmon) and
    parallelism (pytest-xdist) rather than deleting or weakening coverage.
16. **Parametrized config, generators, and managed binaries.** config, settings,
    and templates are the sole source of configuration and business rules; the
    correct generator produces every derived surface (never hand-edit a
    projection). ai-hub owns the installation of binaries and the provisioning of
    environments; no manual, machine-specific path or binary hardcode. There is
    no product-, agent-, or daemon-specific hardcoded code anywhere — every such
    value is parametrized through config/settings/templates.

<!-- /UNIVERSAL-GOVERNANCE -->
<!-- END AI-HUB MANAGED UNIVERSAL CORE -->

> **General FLEXT law & workspace conventions live in the root [`../AGENTS.md`](../AGENTS.md) — read it first.** It is the SSOT for facade layering, config/settings access, the `make`-only workflow, the testing law, and multi-agent git discipline. This file adds ONLY `flext-core`-specific knowledge and never repeats the root.
>
> **Standalone / independent mode:** if this package is checked out on its own (imported as a dependency, vendored, or cloned solo) there is no parent workspace, so `../AGENTS.md` does not resolve. Then read the root law from the raw file on the SAME branch/release the project is on: <https://raw.githubusercontent.com/flext-sh/flext/0.12.0-dev/AGENTS.md> (pin the branch/tag to your working line, never `main`).

**Package:** `flext_core` · ~30.9k src LOC · deps: **none** (workspace root of the dependency graph)

## Overview

Enterprise Foundation Framework (Python 3.13 + Clean Architecture). Defines the facade alphabet every other `flext-*` package composes via MRO. Because it depends on nothing, it can never import another `flext-*` package.

## Structure

```
src/flext_core/
├── constants.py models.py protocols.py typings.py utilities.py   # AUTO-GENERATED facet roots (thin MRO facades)
├── result.py exceptions.py mixins.py handlers.py decorators.py    # operational: r / e / x / h / d
├── service.py container.py context.py dispatcher.py registry.py   # DI + CQRS + runtime: s
├── runtime.py loggings.py lazy.py                                 # runtime, logging factory, PEP-562 lazy exports
├── _config.py _settings.py                                        # LAYER-0 SSOT singletons (config / settings)
├── _constants/ _models/ _protocols/ _typings/ _utilities/         # domain impl behind each facet
│   └── …_parts/                                                   # fragmentation pattern (composition units, not APIs)
├── _result_parts/    # construction / composition / behavior / transforms / unwrap of FlextResult
├── _handlers_parts/ _exceptions/ _decorators/ _lazy_parts/ _runtime/ _beartype/
└── __init__.py                                                    # AUTO-GENERATED lazy export map
```

## Code Map

Central symbols (LSP reference counts within the package, approximate):

| Symbol | Kind | Location | Refs | Role |
|--------|------|----------|------|------|
| `FlextTypes` (`t`) | class | `typings.py:23` | ~54 | composite type aliases |
| `FlextConstants` (`c`) | class | `constants.py:32` | ~34 | constants facade |
| `FlextProtocols` (`p`) | class | `protocols.py:23` | ~31 | structural protocols |
| `FlextModels` (`m`) | class | `models.py:38` | ~22 | Pydantic-2 models |
| `FlextResult` (`r`) | class | `result.py:24` | ~17 | railway result channel (ADR-001) |
| `FlextContainer` | class | `container.py:41` | ~14 | DI container (bind/scope/wire/config sync) |
| `FlextExceptions` (`e`) | class | `exceptions.py:23` | ~14 | exception factories |
| `FlextService` (`s`) | class | `service.py:31` | ~6 | singleton service base (`fetch_global()`) |

Facades are thin MRO aggregators — each root imports its domain components from the matching `_<facet>/` dir. There is **no `api.py`**: this package *is* the foundation.

## Conventions (specific to this package)

- `_config.py` / `_settings.py` are **layer-0 pure**: import only stdlib + pydantic/pydantic-settings, never a project facade. `_config.py` lazily loads `config/config.yaml` into the `config` singleton; `_settings.py` owns the `settings` singleton (incl. universal XDG dirs accessed at the root singleton, e.g. `settings.work_dir`).
- `lazy.py` (`FlextLazy`) provides `build_lazy_import_map` / `install_lazy_exports` used by every package's generated `__init__`.

## Anti-Patterns / Gotchas

- **Never hand-edit** `__init__.py` or the generated facet roots (`# AUTO-GENERATED`).
- **Do not collapse `_parts` composition** without updating the facade imports + export machinery — it breaks the lazy map.
- Preserve lazy imports + `TYPE_CHECKING`-only reverse references: operational modules import public aliases from `flext_core`, so changing export timing creates import cycles.

## Commands

```bash
make check PROJECT=flext-core       # ruff/pyrefly/mypy/pyright
make test  PROJECT=flext-core       # tests/{unit,integration,benchmark,templates}
```

<!-- AIHUB-WORKSPACE-PROVIDERS-BEGIN -->
## Workspace providers

These routes are generated from provider-owned manifests.

- flext: read `.agents/skills/flext-context-routing/SKILL.md` first.
<!-- AIHUB-WORKSPACE-PROVIDERS-END -->
