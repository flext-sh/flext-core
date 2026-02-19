# Clean Architecture


<!-- TOC START -->
- [Layer Hierarchy](#layer-hierarchy)
- [Dependency Rules](#dependency-rules)
- [Layer Responsibilities](#layer-responsibilities)
- [Next Steps](#next-steps)
- [See Also](#see-also)
- [Verification Commands](#verification-commands)
<!-- TOC END -->

**Status**: Production Ready | **Version**: 0.10.0 | **Date**: 2025-12-07

Canonical references:

- `./overview.md`
- `./cqrs.md`
- `../../README.md`

FLEXT-Core keeps CQRS orchestration, domain modeling, and infrastructure
aligned through unidirectional dependencies. Each layer maps directly to
modules in `src/flext_core` and remains intentionally small so contracts stay
explicit. See the Architecture Overview for the visual layout; this guide
describes the dependency rules and per-layer responsibilities.

## Layer Hierarchy

```
┌─────────────────────────────────────┐
│  L3: Application                    │  dispatcher.py, handlers.py, decorators.py
│  (orchestration & middleware)       │  _dispatcher/reliability.py, _dispatcher/timeout.py
├─────────────────────────────────────┤
│  L2: Domain & Infrastructure        │  models.py, _models/*, mixins.py, service.py
│  (DDD, DI, config, context)         │  utilities.py, _utilities/*, config.py, context.py,
│                                     │  loggings.py, container.py
├─────────────────────────────────────┤
│  L1: Foundation & Bridge            │  result.py, exceptions.py, registry.py
│  (railway result, error surface)    │  runtime.py (structlog/dependency-injector bridge)
├─────────────────────────────────────┤
│  L0: Pure Contracts                 │  constants.py, typings.py, protocols.py
│  (immutable constants & protocols)  │
└─────────────────────────────────────┘
```

## Dependency Rules

- **Inward only:** higher layers import lower ones, never the inverse.
- **Bridge isolation:** `runtime.py` pulls external libraries but does not import
  dispatcher or domain modules.
- **Foundation purity:** `constants.py`, `typings.py`, and `protocols.py` avoid
  internal imports so they remain safe for all layers.

```python
# ✅ Correct: application layer depends on domain + foundation
from flext_core import FlextDispatcher, FlextResult

# ❌ Forbidden: foundation pulling from application
from flext_core.dispatcher import FlextDispatcher  # not allowed inside result.py
```

## Layer Responsibilities

- **L0 – Contracts**
  - `constants.py` keeps error codes, retry defaults, cache TTLs, and logging
    keys immutable.
  - `typings.py` defines structured aliases for dispatcher callbacks, cached
    payloads, and configuration schemas.
  - `protocols.py` exposes runtime-checkable interfaces used by container,
    dispatcher, and context implementations.

- **L1 – Foundation & Bridge**
  - `result.py` delivers the railway-oriented `FlextResult` that propagates
    errors without raising.
  - `exceptions.py` centralizes typed exceptions surfaced by dispatcher
    orchestration.
  - `registry.py` shares low-level registration helpers reused by dispatcher and
    container flows.
  - `runtime.py` bridges structlog and dependency-injector while deliberately
    avoiding imports from L2/L3 to prevent cycles.

- **L2 – Domain & Infrastructure**
  - Domain modules (`models.py`, `_models/`, `mixins.py`, `service.py`) wrap
    Pydantic v2 for aggregates, events, validators, and cross-cutting mixins
    (timestamps, versioning, soft deletes).
  - Infrastructure modules carry operational concerns: `config.py` (settings),
    `context.py` (contextvars propagation), `loggings.py` (structured logging
    defaults), `_utilities/` and `utilities.py` (pagination, validators, cache
    helpers, reliability utilities), and `container.py` (DI singleton plus
    scoped containers).

- **L3 – Application / Orchestration**
  - `dispatcher.py` coordinates middleware, rate limiting, circuit breaking,
    retries, timeouts, and handler invocation.
  - `_dispatcher/reliability.py` and `_dispatcher/timeout.py` encapsulate
    resilience policies.
  - `handlers.py`, `decorators.py`, and the application `registry.py` expose the
    handler surface, middleware hooks, and registration helpers consumed by
    services.

Keeping documentation, examples, and code aligned with these boundaries prevents
circular dependencies and keeps FLEXT-Core safe for reuse across services.

## Next Steps

1. **Architecture Overview**: See [Architecture Overview](./overview.md) for visual layer layout
2. **CQRS Patterns**: Explore [CQRS Architecture](./cqrs.md) for application layer patterns
3. **Domain-Driven Design**: Review [DDD Guide](../guides/domain-driven-design.md) for domain patterns
4. **Dependency Injection**: Check [Advanced DI Guide](../guides/dependency-injection-advanced.md) for DI patterns
5. **Service Patterns**: See [Service Patterns](../guides/service-patterns.md) for domain services

## See Also

- [Architecture Overview](./overview.md) - Visual layer topology and execution flows
- [CQRS Architecture](./cqrs.md) - Application layer orchestration patterns
- [Architecture Patterns](./patterns.md) - Implementation patterns
- [Domain-Driven Design Guide](../guides/domain-driven-design.md) - DDD patterns with FlextModels
- [Dependency Injection Advanced](../guides/dependency-injection-advanced.md) - DI container usage
- [Service Patterns Guide](../guides/service-patterns.md) - Domain service implementation
- `../../README.md`: architecture principles and development workflow entrypoint

## Verification Commands

Run from `flext-core/`:

```bash
make lint
make type-check
make test-fast
```
