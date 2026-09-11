# AGENTS.md — flext-core

> **Parent workspace law** lives in [`../AGENTS.md`](../AGENTS.md) — read it first.
> Universal engineering core: `~/.agents/UNIVERSAL_CORE.md`. Composition: global skills + parent/root `AGENTS.md` + this scope delta. Do not re-embed universal law.
>
> **Standalone / independent mode:** when `../AGENTS.md` does not resolve, pin the parent raw `AGENTS.md` URL to the same branch/release as this package (never `main`).

<!-- AIHUB-AGENTS-SCOPE-LOCAL-BEGIN -->
**Package:** `flext_core` · ~30.9k src LOC · deps: **none** (foundation of the dependency graph)

## Overview

Enterprise Foundation Framework (Python 3.13 + Clean Architecture). Defines the facade alphabet every other `flext-*` package composes via MRO. Depends on nothing; must never import another `flext-*` package. No public `api.py` — this package *is* the foundation.

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
- Never hand-edit `__init__.py` or `# AUTO-GENERATED` facet roots; do not collapse `_parts` without updating facade imports (breaks lazy map / cycles).
- Config/settings canonical pattern: ADR-005 (§1–§2) + `_settings.py`/`_config.py` module docstrings.
- Codemod governance (ast-grep + make mod): ADR-014.

## Commands

```bash
make check PROJECT=flext-core
make test  PROJECT=flext-core
```
<!-- AIHUB-AGENTS-SCOPE-LOCAL-END -->

<!-- AIHUB-GOVERNANCE-INSTRUCTIONS-BEGIN -->
<!-- AIHUB-GOVERNANCE-CAPSULE v1 sha256:e1b7a9ae088643bf660b78ff1aea55d49076ec46518707976d77c78ed23db5db -->
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

Compose with generalized ownership,
strict execution,
runtime evidence,
storage isolation,
security closure.

## Rule `coordination/fix-forward-collaboration`

# Adopt the current state and collaborate by fix-forward

Treat every current authorized-repository change as owned input regardless of
provenance or age. Re-read shared files, attribute overlapping intent, preserve
compatible contributions, and adopt them through the integration lane. The
combined result is the target; provenance never exempts a defect from fix-forward.

Preserve unmanifested provider output, generated destinations, external files,
and ambiguous objects without promoting them to canonical input. Bulk adoption
or replacement requires divergent-object adjudication.

Never stash, reset, restore, revert, rebase, force-push, roll back code/history,
or replace shared files to remove work. Fix the canonical owner forward.
Transaction rollback may undo only its failing invocation's effects.

A severe conflict exists only when two current intentions require incompatible
behavior or when preserving both would violate a higher authority. Stop before
the conflicting effect, present both intentions and their evidence, and ask the
operator one precise question. Ordinary overlap, divergence, a red gate, or
integration work is not severe: reconcile, validate, and continue forward.
Unexpected state requires fresh preflight; it proves no actor or intention and
never authorizes an unchanged retry.

Compose this invariant with shared-file coordination,
operator precedence,
plan adoption, and
the destructive Git guard.

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

## Rule `coordination/session-governance`

# Rehydrate governance at every agent context boundary

Static provider instructions own the complete standing contract. Provider hooks
refresh a compact governance capsule at every native equivalent of session
start, prompt submission, context compaction, and subagent start. A hook is a
delivery mechanism generated by optionless `agentsctl sync`; it is never a
policy owner, public command, fallback, daemon, or second runtime path.

At session start, load the current operator and repository instructions before
work. At each prompt, apply the newest operator intent and route only the skills
material to that request. After compaction, restore the active goal, evidence,
scope, exclusions, accepted concurrent work, first red gate, and next action.
Every subagent inherits the current authority, fix-forward contract, and a
bounded assignment; it may not discard, stash, roll back, or overwrite another
actor's work.

Provider capability is explicit in `config/projections.json`. An exact native
event is used when available. A documented per-turn or pre-model equivalent is
used when it is the provider's only delivery point, and observational events
remain observational. Never claim an exact lifecycle semantic that the provider
does not expose. Declarative instructions and projected skills/rules remain the
standing guarantee when a hook can only observe or advise.

Compose with operator precedence,
fix-forward collaboration,
strict execution, and
runtime evidence.

## Rule `ethics/professional-integrity`

# Professional integrity is absolute

Never lie, fabricate evidence, hide a blocker, bypass a gate, or patch a symptom
only to make a check pass. Fix the generalized root cause with full context and
report exact command, working directory, exit code and decisive output.

## Rule `runtime/strict-execution`

# Strict execution is universal and non-optional

Every project and projected agent applies all of these policies together:

- fail loud;
- no fallback;
- preflight before effects;
- required environment;
- atomic effects;
- causal subprocess propagation;
- no keyring;
- zero residue.

The policies are cumulative. A project rule may make them narrower or reject
more inputs; it cannot relax, catch, normalize, skip, defer, or route around any
of them. Existing opposing behavior is a blocking violation to exterminate at
its owner, never grandfathered compatibility.

Resolve gate applicability before invocation. A dormant external-token gate is
not executed; selecting or invoking it applies every policy above.

## Rule `workflow/canonical-commands`

# Run agent functions only through the optionless CLI

`agentsctl` is the sole agent-runtime facade: `help`, `doctor`, `check`, `sync`,
`evaluate`, `secure`, `clean`, and `live`. Each invocation has exactly one verb
and no option, argument, mode, selector, alias, or compatibility syntax.

Make is development support and gate composition. A Make target that needs
runtime behavior invokes one public `agentsctl` verb; it never imports a private
runtime function, reconstructs orchestration, or creates a second API.

Ad hoc shell, inline Python, diagnostics, and test helpers used as operational
substitutes never import or execute private agent runtime. Repository and GitHub
work uses its declared `make`, `git`, and `gh` owner, never an executable
lower-level substitute.

A broken or out-of-pattern command is a defect to fix at its owner and rerun
through the same surface. Bypasses are blocking violations, not warnings.

## Rule `workflow/runtime-is-reality`

# Reality is the running system; tests are checks, not the SSOT

Validate against the real declared runtime (CLI, daemon, config, or public API)
and exercise the actual feature—type and lint green are necessary, not
sufficient.

- A test that only passes by keeping removed or legacy artifacts is wrong: fix or
  delete the test; never restore legacy just to make it pass.
- A config/settings test that breaks when a valid SSOT value changes is defective.
  Test contracts and derivations across arbitrary valid inputs; goldens may lock
  generated structure, never mutable config-owned values.
- A missing facade constant fails only at runtime — import and run the real path.
- Before concluding root cause, prove the running or installed artifact matches
  the declared authoritative revision or release. An editable checkout, local
  cache, generated copy, or stale environment is not evidence of remote/runtime
  behavior until identity is verified.
- Use the newest released version of every required tool. Every diagnostic it
  emits is blocking. A cap, downgrade, substitution, suppression, compatibility
  classification, or false-positive classification requires prior operator
  discussion, reproducible evidence, and explicit authorization; without all
  three, correct the owner and rerun that released version.

## Capability indexes

Skills: caveman, context-canary, fix-forward-collaboration, governance-audit, operator-correction-learning, plan-focus-recovery, sprint-closure, strategic-compact, verification-loop
Commands: add-language-rules, database-migration, feature-development, ghi-list, pr-list, ralph-loop, security-triage, synthesize-governance
<!-- AIHUB-GOVERNANCE-INSTRUCTIONS-END -->
