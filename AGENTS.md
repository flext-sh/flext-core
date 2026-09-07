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

<!-- AIHUB-GOVERNANCE-INSTRUCTIONS-BEGIN -->
<!-- AIHUB-GOVERNANCE-CAPSULE v1 sha256:5888ee9f8147f63364a4f7cd6906e9d837f58cb8a8546844760c526ecb1a303b -->
# Generated session governance capsule

This projection is derived by `agentsctl sync`; edit canonical `AGENTS.md`, `rules/`, `skills/`, or `commands/`, never this output. The operator's newest request has precedence. Provider hooks are delivery mechanisms, not policy owners.

## Rule `architecture/engineering-core`

# Engineering core

For every implementation:

1. Research repository owners, dependencies, and canonical documentation.
2. Remove scope without a current requirement or consumer (YAGNI).
3. Elect one writable authority; every other copy is a generated projection
   (SSOT).
4. Apply SOLID only to a responsibility or dependency boundary under change.
5. Implement through the owner and simplify without weakening behavior.
6. Remove duplication and god components; recheck YAGNI, SSOT, SOLID.
7. Exercise runtime behavior, run every applicable native gate, and complete
   the approved landing cycle before changing phase.

At a cross-boundary failure, prove the producer contract and output. Fix its
owner when invalid or the receiver when it conforms. Never alter a correct
adjacent owner for an invalid consumer; symptom workarounds are defects.

Hardcodes, normalized failure, failover, retry, fallback, compatibility,
partial execution, keyring, and unevidenced success are defects. Typed owners
keep defaults. The first exception escapes its CLI with traceback and cause.

Git, runtime, build, and tests are baseline. Auxiliary tracking is a capability.
Auxiliary capabilities apply only when authorized and selected; installation
never selects. Do not load, probe, or gate dormant capabilities. Invalid
selected authorization, configuration, readiness, or result fails without
fallback. Require only non-derivable values.

An external token validation without its token is not executed and is recorded
as `NOT EXECUTED`, never green; it does not block offline gates, landing, or
post-merge proof. Direct invocation selects it: the token becomes required and
any failure escapes without skip, catch, fallback, or normalization.

Compose with `generalized ownership` (rule file),
`strict execution` (rule file),
`runtime evidence` (rule file),
`storage isolation` (rule file),
`security closure` (rule file).

## Rule `coordination/operator-precedence`

# Newest operator instruction wins; adjust artifacts to it

Authority order: operator request > declared orchestration contract > canonical
tracker > ADRs > skills > docs, and newest supersedes oldest. On conflict,
adjust the lower or older artifact to match; never override the operator to
satisfy stale guidance.

While orchestration and tracker runtimes are suspended, do not invoke them.
Create no substitute tracker or ledger, preserve implementation evidence only
in separately authorized Git/PR/CI, and leave phase closure open.

Exact operator authorization naming targets, disposition, recovery, and
validation survives interruption, divergence, and red gates; re-preflight and
continue. Ask only when the effect expands beyond it or two evidenced current
intentions conflict. State alone proves no intention, actor, or process.

## Rule `ethics/professional-integrity`

# Professional integrity is absolute

Never lie, fabricate evidence, hide a blocker, bypass a gate, or patch a symptom
only to make a check pass. Fix the generalized root cause with full context and
report exact command, working directory, exit code and decisive output.

## Rule `runtime/strict-execution`

# Strict execution is universal and non-optional

Every project and projected agent applies all of these policies together:

- `fail loud` (rule file);
- `no fallback` (rule file);
- `preflight before effects` (rule file);
- `required environment` (rule file);
- `atomic effects` (rule file);
- `causal subprocess propagation` (rule file);
- `no keyring` (rule file);
- `zero residue` (rule file).

The policies are cumulative. A project rule may make them narrower or reject
more inputs; it cannot relax, catch, normalize, skip, defer, or route around any
of them. Existing opposing behavior is a blocking violation to exterminate at
its owner, never grandfathered compatibility.

Resolve gate applicability before invocation. A dormant external-token gate is
not executed; selecting or invoking it applies every policy above.

## Capability indexes

Skills: caveman, context-canary, fix-forward-collaboration, governance-audit, operator-correction-learning, plan-focus-recovery, sprint-closure, strategic-compact, verification-loop
Commands: add-language-rules, database-migration, feature-development, ghi-list, pr-list, ralph-loop, security-triage, synthesize-governance
<!-- AIHUB-GOVERNANCE-INSTRUCTIONS-END -->
