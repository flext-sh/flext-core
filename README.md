<!-- AUTO-GENERATED — DO NOT EDIT MANUALLY -->

# flext-core

**Version**: `unknown` | **Python**: 3.13+ | **Project class**: `platform`

## Purpose

Enterprise Foundation Framework - Modern Python 3.13 + Clean Architecture

## Module Map

- Primary facades: `FlextModelsEnforcement`, `FlextUtilitiesGuards`, `FlextProtocolsLogging`, `FlextUtilitiesLoggingContext`, `FlextModelsCollections`, `FlextConstantsCqrs` (+118 more)
- Alias namespaces: `c`, `d`, `e`, `h`, `m`, `p`, `r`, `s`, `t`, `u`, `x`
- Public symbol exports: `127`
- Exported module shortcuts: `lazy`

## Collection Rules (regras de coletas)

Required pre-work before changing this project (per AGENTS.md §9):

1. Read [`/flext/AGENTS.md`](../AGENTS.md) (governance) and this project's `pyproject.toml`.
2. Confirm parent MRO chain via `pyproject.toml` `dependencies` filtered by `flext-*` (excluding `flext-core` self).
3. Verify Scope: `cd <project> && scope status` (re-bootstrap per `flext-scope-bootstrap` if absent).
4. Load skills relevant to the change scope from [`/flext/.agents/skills/`](../.agents/skills/) (start with `flext-mro-namespace-rules`, `flext-import-rules`, `flext-patterns`).
5. Confirm the canonical zero-debt baseline:
    - `make check` exits 0
    - `make val VALIDATE_SCOPE=project` exits 0
    - `make docs DOCS_PHASE=audit` reports zero issues
6. Cross-check the c/p/t/m/u slot registry in [`flext-mro-namespace-rules`](../.agents/skills/flext-mro-namespace-rules/SKILL.md) to confirm this project's owned slots before adding/renaming any symbol.

## Operation Flow

- Public surface: see [`docs/index.md`](docs/index.md) and [`docs/api-reference/README.md`](docs/api-reference/README.md).
- Generated module overview: [`docs/api-reference/generated/overview.md`](docs/api-reference/generated/overview.md).
- Settings env prefix: see project `pyproject.toml` `[tool.flext]` and `FlextSettings` ConfigDict.

## Integration Points

- Parent MRO chain: read this project's `pyproject.toml` `dependencies` array filtered by `flext-*`. The MRO cascade is encoded in the inheritance lists of the facade classes listed under Module Map above.
- Public extensions exposed by this project: `FlextModelsEnforcement`, `FlextUtilitiesGuards`, `FlextProtocolsLogging`, `FlextUtilitiesLoggingContext`, `FlextModelsCollections`, `FlextConstantsCqrs` (+118 more).
- Library abstraction boundaries: see AGENTS.md §2.7.

## Quality Gates

- `make check` — Lint suite (ruff, pyrefly, mypy, pyright per project).
- `make test` — Pytest with project coverage threshold from `pyproject.toml`.
- `make val VALIDATE_SCOPE=project` — Validation gates (complexity, docstring, namespace).
- `make docs DOCS_PHASE=audit` — Docs audit (broken links, stale symbols, missing docstrings).
- `make docs DOCS_PHASE=build` — Build mkdocs HTML output to `.reports/docs/site/`.

## Governance Pointer

- Canonical engineering law: [`/flext/AGENTS.md`](../AGENTS.md).
- Project skills index: [`/flext/.agents/skills/`](../.agents/skills/).
- Workspace onboarding: [`/flext/docs/guides/onboarding.md`](../docs/guides/onboarding.md).
- Full project portal: [`docs/index.md`](docs/index.md).
