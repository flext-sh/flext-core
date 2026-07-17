<!-- AUTO-GENERATED — DO NOT EDIT MANUALLY -->

# flext-core

**Version**: `0.20.0` | **Python**: 3.13+ | **Project class**: `platform`

> **Alpha (0.20.0).** This package is alpha quality. Every package in the workspace must be re-checked and re-validated at 0.20.0 before any promotion beyond alpha; treat interfaces as unstable.

## Purpose

Enterprise Foundation Framework - Modern Python 3.13 + Clean Architecture

## Module Map

::: flext_core
    options:
      members: false
      show_root_heading: false
      show_root_toc_entry: false
      show_source: false

## Collection Rules

Read [`/flext/AGENTS.md`](../AGENTS.md) §9 — Agent Execution Pre-requisites — for the canonical pre-change checklist (parent MRO chain, Scope bootstrap, skill loading, zero-debt baseline, slot registry verification).

## Operation Flow

- Public surface: see [`docs/index.md`](docs/index.md) and [`docs/api-reference/README.md`](docs/api-reference/README.md).
- Generated module overview: [`docs/api-reference/generated/overview.md`](docs/api-reference/generated/overview.md).
- Settings env prefix: see project `pyproject.toml` `[tool.flext]` and `FlextSettings` ConfigDict.

## Integration Points

- Parent MRO chain: read this project's `pyproject.toml` `dependencies` array filtered by `flext-*`. The MRO cascade is encoded in the inheritance lists of the facade classes listed under Module Map above.
- Public extensions exposed by this project: `FlextConfig`, `FlextConstants`, `FlextContainer`, `FlextContext`, `FlextDecorators`, `FlextDispatcher` (+14 more).
- Library abstraction boundaries: see AGENTS.md §2.7.

## Quality Gates

Canonical `make` verbs (`check`, `test`, `val`, `docs`) — see `AGENTS.md` §5 (Make Contract) and the [`flext-quality-gates`](../.agents/skills/flext-quality-gates/SKILL.md) skill for selectors and thresholds.

## Governance Pointer

- Engineering law: [`/flext/AGENTS.md`](../AGENTS.md)
- Governance + ADRs: [`/flext/docs/GOVERNANCE.md`](../docs/GOVERNANCE.md)
- Skills index: [`/flext/.agents/skills/`](../.agents/skills/)
- Onboarding: [`/flext/docs/guides/onboarding.md`](../docs/guides/onboarding.md)
- Full project portal: [`docs/index.md`](docs/index.md).
