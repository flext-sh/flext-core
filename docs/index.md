<!-- AUTO-GENERATED — DO NOT EDIT MANUALLY -->

# flext-core Documentation

- Version: `unknown`
- Project class: `platform`
- Package: `flext_core`
- Description: Enterprise Foundation Framework - Modern Python 3.13 + Clean Architecture

This project portal is generated from `pyproject.toml`, package exports, and real docstrings.

## Start Here

- [Guides](guides/README.md)
- [API Reference](api-reference/README.md)
- [Generated API Overview](api-reference/generated/overview.md)
- [Generated Module Index](api-reference/generated/modules/index.md)

## Public Surface Summary

- Primary facades: `FlextModelsBase`, `FlextConstantsRegex`, `FlextConstantsValidation`, `FlextUtilitiesConversion`, `FlextExceptionsBase`, `FlextModelsCollections` (+118 more)
- Alias namespaces: `c`, `d`, `e`, `h`, `m`, `p`, `r`, `s`, `t`, `u`, `x`
- Public symbol exports: `127`
- Exported module shortcuts: `lazy`

## Collection Rules (regras de coletas)

Required pre-work before changing this project (per AGENTS.md §9):

1. Read `/flext/AGENTS.md` (governance) and the project's `pyproject.toml`.
2. Confirm parent MRO chain via `pyproject.toml` `dependencies` filtered by `flext-*` (excluding `flext-core` self).
3. Verify Scope: `cd <project> && scope status` (re-bootstrap per `flext-scope-bootstrap` if absent).
4. Load skills relevant to the change scope in `.agents/skills/` (start with `flext-mro-namespace-rules`, `flext-import-rules`, `flext-patterns`).
5. Confirm the canonical zero-debt baseline:
    - `cd <project> && make check` exits 0
    - `cd <project> && make val VALIDATE_SCOPE=project` exits 0
    - `cd <project> && make docs DOCS_PHASE=audit` reports zero issues
6. Cross-check the c/p/t/m/u slot registry in `.agents/skills/flext-mro-namespace-rules/SKILL.md` to confirm this project's owned slots before adding/renaming any symbol.

## Quality Gates

- `make check` — Lint suite (ruff, pyrefly, mypy, pyright per project).
- `make test` — Pytest with project coverage threshold from `pyproject.toml`.
- `make val VALIDATE_SCOPE=project` — Validation gates (complexity, docstring).
- `make docs DOCS_PHASE=audit` — Docs audit (broken links, stale symbols, missing docstrings).
- `make docs DOCS_PHASE=build` — Build mkdocs HTML output to `.reports/docs/site/`.

## Governance Pointer

- Canonical engineering law: [`/flext/AGENTS.md`](../../../AGENTS.md).
- Project skills index: [`/flext/.agents/skills/`](../../../.agents/skills/).
- Workspace onboarding: [`/flext/docs/guides/onboarding.md`](../../../docs/guides/onboarding.md).
