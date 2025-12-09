# Architecture Overview

**Status**: Production Ready | **Version**: 0.10.0 | **Date**: 2025-12-07

FLEXT-Core implements CQRS on top of a clean-architecture skeleton. The outline
below mirrors the `src/flext_core` layout (Python 3.13+, Pydantic v2) and
references [`clean-architecture.md`](./clean-architecture.md) for the
dependency rules and rationale.

## Layered Topology

```
┌─────────────────────────────────────────────────────────────┐
│              Application / Orchestration (L3)               │
│  dispatcher.py, handlers.py, decorators.py, registry.py     │
│  _dispatcher/reliability.py, _dispatcher/timeout.py         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│             Domain & Infrastructure Services (L2)           │
│  models.py, _models/*, mixins.py, service.py                │
│  utilities.py, _utilities/*, config.py, context.py          │
│  loggings.py, container.py                                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                Foundation & Bridge Layers (L1)              │
│  result.py, exceptions.py, registry.py                      │
│  runtime.py (structlog/dependency-injector bridge)          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Pure Contracts (L0)                      │
│  constants.py, typings.py, protocols.py                     │
└─────────────────────────────────────────────────────────────┘
```

## Layer Catalog (source-aligned)

- **L0 – contracts and primitives**
  - `constants.py` centralizes error codes, retry defaults, cache TTLs, and
    logging keys as immutable data.
  - `typings.py` provides structured aliases for handler callables, cache
    payloads, configuration shapes, and message generics.
  - `protocols.py` exposes runtime-checkable interfaces for configuration,
    contexts, containers, and handlers.

- **L1 – runtime bridge and results**
  - `runtime.py` wraps structlog and dependency-injector factories so higher
    layers can configure logging and DI without depending on third-party types.
  - `result.py` delivers the railway-oriented `FlextResult`; `exceptions.py`
    contains the CQRS exception hierarchy consumed by handlers.
  - `registry.py` offers the shared registration helpers reused by dispatcher,
    container, and decorators.

- **L2 – domain and infrastructure services**
  - Domain façade modules (`models.py`, `_models/*`, `mixins.py`, `service.py`)
    host Pydantic-backed DDD entities, aggregates, validators, and mixins for
    timestamps, versioning, and domain events.
  - Infrastructure sits beside the domain types: `config.py` (settings via
    `BaseSettings`), `context.py` (contextvars metadata propagation),
    `loggings.py` (structlog defaults), `utilities.py`/`_utilities/*`
    (validation, pagination, caching, data mappers, reliability helpers), and
    `container.py` (dependency-injector singleton plus scoped container
    factory).

- **L3 – application orchestration**
  - `dispatcher.py` drives CQRS routing with reliability policies from
    `_dispatcher/reliability.py` (circuit breakers, retries, rate limiting) and
    `_dispatcher/timeout.py` (deadline enforcement).
  - `handlers.py`, `decorators.py`, and the application `registry.py` define the
    handler pipeline, middleware hooks, and registration helpers consumed by
    services.

## Key Execution Flows

- **Command/query dispatch** — `FlextDispatcher.dispatch` enriches the
  `FlextContext`, applies rate limiting, circuit breaking, retries, and timeout
  enforcement, then executes the registered handler with structured logging and
  optional query caching.
- **Dependency injection** — `FlextContainer` hosts a dependency-injector
  container. Registrations and resolutions return `FlextResult` so handler
  wiring can surface errors without raising exceptions.
- **Domain validation** — `FlextModels` exposes Pydantic entities, values, and
  aggregates. Domain events collected on aggregates can be published through
  dispatcher subscribers.

These flows keep orchestration decoupled from infrastructure concerns while
preserving the unidirectional boundaries described in the clean-architecture
guide.

## Next Steps

1. **Clean Architecture**: Deep dive into [Clean Architecture](./clean-architecture.md) for dependency rules
2. **CQRS Patterns**: Explore [CQRS Architecture](./cqrs.md) for handler and dispatcher patterns
3. **Architecture Patterns**: See [Architecture Patterns](./patterns.md) for common patterns
4. **Decision Records**: Review [Architecture Decisions](./decisions.md) for design rationale
5. **Guides**: Check [Getting Started](../guides/getting-started.md) for practical usage

## Related Documentation

**Within Project**:

- [Clean Architecture](./clean-architecture.md) - Dependency rules and layer responsibilities
- [CQRS Architecture](./cqrs.md) - Handler pipeline and dispatcher orchestration
- [Architecture Patterns](./patterns.md) - Common patterns and best practices
- [Architecture Decisions](./decisions.md) - ADRs documenting design choices
- [Getting Started Guide](../guides/getting-started.md) - Practical implementation guide
- [Service Patterns](../guides/service-patterns.md) - Domain service patterns
- [API Reference](../api-reference/foundation.md) - Foundation layer APIs

**External Resources**:

- [PEP 257 - Docstring Conventions](https://peps.python.org/pep-0257/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
