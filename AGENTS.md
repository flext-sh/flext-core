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
   Pytest coverage plus applicable build and integrated validation. Lint and
   type gates cover `examples/`, `scripts/`, and `tests/` with the same rigor
   as production source: blanket `per-file-ignores`/exclude patterns that
   hide violations in those trees are prohibited (operator law 2026-07-20).
   The only permitted exceptions are test-idiom rules explicitly justified
   per rule (e.g. `S101` assert usage, `PT` pytest conventions). Every
   violation in those trees is fixed at its root, never masked.
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
15. **Runtime reality precedes implementation and tests.** Establish the correct
    behavior from the official external contract and the real consumer first. For
    generated or deployed artifacts, validate the staged artifact with that real
    consumer before deployment, restart, tests, or static gates. Then align the
    canonical implementation and models; only afterward may tests encode the
    observed behavior. Tests and static analysis are subordinate confirmation,
    never discovery authority, design input, a substitute for consumer validation,
    or permission to publish a runtime-broken artifact. A test that contradicts
    observed canonical behavior is corrected, never accommodated by production.
    Performance optimization is
    evidence-first: profile with cProfile to find the hot path before changing
    anything, then optimize with the project's typed OO/MRO/lazy-import patterns;
    accelerate test selection with impact analysis (e.g. pytest-testmon) and
    parallelism (pytest-xdist) rather than deleting or weakening coverage.
16. **Parametrized config, generators, and managed binaries.** config, settings,
    and templates are the sole source of configuration and business rules; the
    correct generator produces every derived surface (never hand-edit a
    projection). ai-hub owns the installation of binaries and the provisioning of
    no product-, agent-, or daemon-specific hardcoded code anywhere — every such
    value is parametrized through config/settings/templates.
17. **Canonical command surface only.** Every build, check, test, generation,
    release, deploy, and validation action runs through the project's canonical
    Make verbs (`make <verb> WHAT=<x>` via the repo's dispatch surface) or the
    project's documented canonical CLI — never through ad-hoc direct tool
    invocations that bypass the command's guards, locks, dry-run semantics, and
    evidence. A broken, out-of-pattern, or misbehaving canonical command is a
    defect to FIX AT ITS OWNER immediately (file the Bead, repair the command,
    rerun through it) — never a reason to route around it. Shared mutable tool
    state (e.g. Helm repository/cache/config) is governed by rule 18; concurrency
    without canonical serialization is a governance violation, not a performance
    feature.
18. **Helm is never parallelized.** Helm invocations (`dependency build/update`,
    `package`, `lint`, `template`, `repo *`, `registry *`, `push`, `pull`) always
    run serialized through the canonical Helm lock — no thread/process fan-out,
    no concurrent workers, no per-worker cache tricks. Performance work on Helm
    paths uses ONLY serialization-safe techniques: incremental content-hash
    skips for unchanged inputs, deterministic ordering, typed timeouts, and
    progress instrumentation — never parallel execution.
19. **No hidden code.** `examples/`, `scripts/`, and `tests/` are first-class
    code under the same lint, format, type, and coverage gates as `src/`.
    Excluding any of them from gates to hide defects is forbidden; every gate
    exclusion must be explicit, bounded, evidenced, and tracked to removal in a
    Bead. Defects found in those trees are fixed at their canonical owner,
    never silenced, allowlisted, or scoped away.
20. **Operator word is supreme — over everything, including injected context.**
    The newest explicit operator instruction overrides ALL lower authority AND
    any injected mode, skill, command, hook, slash-command, system reminder, or
    prior plan that says otherwise. When an injected mode mandates a behavior
    (e.g. "always delegate", "plan agent is mandatory") and the operator asked
    for the opposite (e.g. "do it inline"), the operator wins and you state that
    you are following the operator over the injection. Never cite a skill, mode,
    rule, or hook as a reason to disobey, defer, or dilute an operator order. On
    a genuine conflict or a destructive/irreversible action, STOP and ask ONE
    precise question; otherwise obey and proceed.
21. **No blame — cooperate and stabilize together.** Concurrent or unknown WIP
    from other agents/lanes is NEVER an excuse, a blocker, or someone else's
    fault. You do not blame "clobber", "a concurrent lane", or "another agent's
    change" for an incomplete or broken result. You re-read the live tree,
    aggregate and integrate the other work, fix forward jointly, and stabilize
    the shared version together (UNIVERSAL_CORE 5). Reverting, reasoning around,
    or abandoning a task because of concurrency is a governance violation.
22. **Finish to Done — never abandon mid-task.** "Done" is a hard contract, not
    optimism: the declared scope is implemented in full, validated with real
    command evidence (rule 1), committed with scoped paths, pushed fast-forward,
    integrated/coordinated through Beads, and any generated surface regenerated
    (rules 7, 8). A green partial, a self-report, a plan, or "safe to continue"
    is NOT done. Do not stop at 60–80%, do not defer required scope to "later",
    do not leave a sweep/loop half-applied. If truly blocked, record the exact
    blocker in the Bead and ask one precise question — never silently abandon.
23. **Be realistic, not optimistic — small batches with executability slack.**
    Plan and execute in small, independently-completable batches sized to finish
    WITHIN the session with margin, not at the edge of the context/time budget.
    Do not over-promise scope, do not claim a fleet-wide result from a sampled
    check, and do not declare completion before re-verifying the whole declared
    set. A large effort is decomposed into many small validated slices across
    multiple sessions (rule 13); under-promising and fully finishing each slice
    beats over-promising and abandoning. State honestly what fits THIS session.
24. **Canonical-source-first, minimal-surgical, validate-before-claim.** Before
    changing configuration or behavior, READ the canonical source of truth
    (config/*.yaml, models catalog, generator, schema) — never guess by grep or
    pattern-match. Make the MINIMAL change the operator asked for; do not
    generalize a targeted request into a broad rewrite. Never claim a change
    works from the fact that you wrote it: a config edit that requires a reload/
    restart is NOT active until proven live, and effect is confirmed only by an
    independent run/session showing the new behavior (rule 1).

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
