# Architecture

> Synchronised with `docs/architecture/flext_modernization_plan.md` (baseline dated 2025-09-17).

The architecture summary focuses on the three modernization pillars mandated for the 1.0.0 release: unified dispatcher flow, context-first observability, and configuration/domain-service alignment.

---

## Layering

```
src/flext_core
├── Foundations         # result.py, typings.py, constants.py, version.py
├── Runtime Surfaces    # container.py, domain_services.py, models.py, utilities.py
├── Execution Flow      # bus.py, dispatcher.py, dispatcher_registry.py, handlers.py, processing.py, cqrs.py
├── Context & Logging   # context.py, loggings.py, mixins.py
└── Configuration       # config.py
```

- **Foundations** stay ABI-compatible across the 1.x line. Dual `.value`/`.data` access on `FlextResult` remains permanent.
- **Runtime Surfaces** expose the shared ergonomics expected by downstream domain services.
- **Execution Flow** introduces the unified dispatcher façade and registry used for Phase 1 migrations.
- **Context & Logging** provide the context-first observability primitives that downstream packages must adopt.
- **Configuration** ensures a single entry point for application settings (`FlextConfig`).

---

## Pillar 1: Unified Dispatcher Flow

- `FlextDispatcher` orchestrates `FlextBus` execution, handler registration, and metadata propagation.
- `FlextDispatcherRegistry` offers batch registration summaries for CLI/connector bootstraps.
- `handlers.py` keeps handler contracts stable so downstream packages can migrate without rewrites.
- Integration tests guard the exports: `tests/integration/test_wildcard_exports_clean.py`.

## Pillar 2: Context-First Observability

- `FlextContext` manages correlation, request, service, and performance scopes. Context propagation helpers (e.g., `context_scope`) replace bespoke solutions.
- `FlextLogger` attaches context automatically and emits structured events compatible with observability pipelines.
- Mixins (`FlextMixins.Loggable`) surface consistent logging behaviour throughout domain services.

## Pillar 3: Configuration & Domain Service Alignment

- `FlextConfig` centralises environment parsing, `.env` handling, and layered settings.
- `FlextContainer` remains the singleton DI surface, now referenced directly by dispatcher/setup guides.
- `FlextDomainService` and `FlextModels` ensure domain logic remains immutable, typed, and context-aware.

---

## Metrics & Goals

- **Coverage**: 79% baseline → ≥85% for 1.0.0 (tracked via `make test` + coverage reports).
- **Compatibility**: `__all__` exports locked; `tests/integration/test_wildcard_exports_clean.py` protects public surface.
- **Performance**: `FlextResult` operations remain sub-microsecond; dispatcher adds minimal overhead via context scopes.

---

## Change Management

1. Architecture changes must reference the modernization plan workstream (Phase 1–4).
2. New modules update `src/flext_core/README.md` and the relevant pillar section above.
3. Downstream migrations should leverage `FlextDispatcherRegistry` summaries to confirm handler state.

This architecture guide is the canonical reference point for the 1.0.0 release.
