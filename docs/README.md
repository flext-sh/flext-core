# FLEXT-Core Documentation

<!-- TOC START -->

- [Scope and Compatibility](#scope-and-compatibility)
- [Navigation](#navigation)
- [🔧 Quality Assurance](#quality-assurance)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Style Expectations](#style-expectations)
- [Support](#support)

<!-- TOC END -->

**Reviewed**: 2026-02-17 | **Scope**: Canonical rules alignment and link consistency

Comprehensive reference and guidance for FLEXT-Core, the dispatcher-first foundation library built around railway-oriented programming, dependency injection, and domain-driven design. Content follows PEP 8, PEP 257, and the project documentation standards in `docs/standards/`.

## Scope and Compatibility

- **Version:** 0.9.9
- **Python:** 3.13+ (per `pyproject.toml`)
- **Architecture:** Dispatcher-centric CQRS with `FlextResult`, `FlextContainer`, `FlextDispatcher`, and DDD primitives.

## Navigation

## 🔧 Quality Assurance

This project integrates with FLEXT's comprehensive quality assurance system:

- **Pattern Enforcement**: Automatic validation of architectural patterns

- **Consolidation Guidance**: SOLID-based refactoring recommendations

- **Quality Validation**: Continuous checks for enterprise standards

- **Quick start:** `quick-start.md`

- **Architecture:** `architecture/overview.md` and `architecture/clean-architecture.md`

- **API reference:** `api-reference/` grouped by layer (foundation, domain, application, infrastructure)

- **Guides:** `guides/` for patterns such as railway execution, DI, DDD, error handling, configuration, testing, and troubleshooting

- **Standards:** `standards/` for code, documentation, and templates

- **Contributing:** `development/contributing.md`

- **DI pattern prompt:** `dependency_injector_prompt.md` for the canonical dependency-injector bridge guidance

## Quick Start

Install the package and verify imports:

```bash
pip install flext-core
python - <<'PY'
from flext_core import FlextDispatcher, FlextResult
print('flext-core ready', FlextDispatcher.__name__, FlextResult.__name__)
PY
```

## Core Concepts

1. **Railway-oriented programming (`FlextResult`)** — express success/failure without exceptions and chain operations with `map`/`flat_map`.
1. **Dependency injection (`FlextContainer`)** — register and resolve shared collaborators explicitly; avoid implicit globals.
1. **CQRS dispatcher (`FlextDispatcher`)** — route commands, queries, and domain events through handler registries with optional middleware.
1. **Domain-driven design (`FlextModels`, `FlextService`)** — model entities/values and encapsulate domain services that return `FlextResult`.
1. **Layered dependency-injector bridge** — isolate dependency-injector usage to the runtime/container while handlers use `provide`/`inject` only.

## Style Expectations

- Prefer docstrings that follow PEP 257 sentence-style summaries and keep examples PEP 8 compliant.
- Cross-reference the `docs/standards/documentation.md` templates when adding new material.
- Avoid duplicated sections across guides; link to existing topics instead of restating them.
- When documenting DI, align with the bridge/handler layering in `dependency_injector_prompt.md` and show examples using the re-exported `provide`/`inject` helpers.

## Support

- **Issues and questions:** GitHub Issues/Discussions
- **Code of conduct and contribution flow:** see `development/contributing.md`
