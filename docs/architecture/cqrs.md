# CQRS Architecture

**Version:** 1.0 (2025-12-03)
**Python:** 3.13+
**Pydantic:** 2.x
**Status:** V1 production baseline; V2 modernization in progress

This document describes the Command Query Responsibility Segregation (CQRS)
implementation in flext-core, including the handler pipeline, dispatcher
orchestration, and reliability patterns.

---

## Overview

FLEXT-Core implements CQRS through two primary components:

- **`FlextHandlers`** (`handlers.py`) – Base class for message handlers
- **`FlextDispatcher`** (`dispatcher.py`) – Orchestration and routing

Both components follow railway-oriented programming with `FlextResult` and
integrate with the infrastructure provided by `FlextMixins`.

```
┌─────────────────────────────────────────────────────────────────┐
│                    FlextDispatcher (L3)                         │
│  ├── CQRS routing (command, query, event)                       │
│  ├── Reliability patterns (circuit breaker, retry, timeout)     │
│  └── Context propagation and observability                      │
├─────────────────────────────────────────────────────────────────┤
│                    FlextHandlers (L3)                           │
│  ├── Message validation pipeline                                │
│  ├── Execute → Validate → Handle flow                           │
│  └── Metrics and context tracking                               │
├─────────────────────────────────────────────────────────────────┤
│                    FlextService (L2.5)                          │
│  └── Domain services called by handlers                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## FlextHandlers

### Current Implementation (V1)

Handlers derive from `FlextHandlers[MessageT, ResultT]` and implement the
abstract `handle()` method:

```python
from flext_core.handlers import FlextHandlers
from flext_core.result import r

class CreateUserHandler(FlextHandlers[CreateUserCommand, User]):
    def handle(self, command: CreateUserCommand) -> r[User]:
        # Business logic
        user = User(name=command.name, email=command.email)
        return r[User].ok(user)
```

### Execution Pipeline

The `execute()` method orchestrates validation and handling:

```
execute(message)
    │
    ├─► validate(message)
    │       └─► Returns r[bool] (fail fast on error)
    │
    └─► handle(message)
            └─► Returns r[ResultT]
```

### Handler Configuration

Handlers accept optional configuration via `FlextModelsCqrs.Handler`:

```python
from flext_core._models.cqrs import FlextModelsCqrs

config = FlextModelsCqrs.Handler(
    handler_id="user_handler_001",
    handler_name="CreateUserHandler",
    handler_mode=c.Cqrs.HandlerType.COMMAND,
)

handler = CreateUserHandler(config=config)
```

### Metrics and Context (V1 – Manual)

The current implementation uses manual state management:

```python
# Internal state (handlers.py lines 177-178)
self._context_stack: list[dict[str, t.GeneralValueType]] = []
self._metrics: dict[str, t.GeneralValueType] = {}

# Methods for state management
handler.push_context({"operation": "create_user"})
handler.record_metric("users_created", 1)
metrics = handler.get_metrics()
handler.pop_context()
```

> **TODO(handlers.py::FlextHandlers):** Migrate to `FlextMixins.CQRS` utilities
> for metrics and context once Phase 1 of CQRS modernization lands. See
> [Modernization Roadmap](#modernization-roadmap).

---

## FlextDispatcher

### Current Implementation (V1)

The dispatcher initializes reliability managers internally:

```python
from flext_core.dispatcher import FlextDispatcher

dispatcher = FlextDispatcher()
dispatcher.register_handler(CreateUserCommand, CreateUserHandler())
result = dispatcher.dispatch(CreateUserCommand(name="Alice", email="alice@example.com"))
```

### Reliability Patterns

The dispatcher applies layered reliability controls:

| Pattern         | Manager Class           | Configuration Source       |
| --------------- | ----------------------- | -------------------------- |
| Circuit Breaker | `CircuitBreakerManager` | `config.circuit_breaker_*` |
| Rate Limiting   | `RateLimiterManager`    | `config.rate_limit_*`      |
| Retry           | `RetryPolicy`           | `config.max_retry_*`       |
| Timeout         | `TimeoutEnforcer`       | `config.enable_timeout_*`  |

### Dispatch Flow

```
dispatch(message)
    │
    ├─► Rate limiter check
    │       └─► Fail if limit exceeded
    │
    ├─► Circuit breaker check
    │       └─► Fail if circuit open
    │
    ├─► Timeout + Retry wrapper
    │       └─► Execute handler.execute(message)
    │
    └─► Update circuit breaker state
            └─► Record success/failure
```

### Handler Registration

```python
# Register by message type
dispatcher.register_handler(CreateUserCommand, handler)

# Register with explicit mode
dispatcher.register_command(CreateUserCommand, handler)
dispatcher.register_query(GetUserQuery, handler)
dispatcher.register_event(UserCreatedEvent, handler)
```

> **TODO(dispatcher.py::FlextDispatcher.**init**):** Accept `container` parameter
> for dependency injection of reliability managers. See Phase 2 of
> [Modernization Roadmap](#modernization-roadmap).

---

## Integration with FlextService

Handlers orchestrate while services execute domain logic:

```python
class CreateUserHandler(FlextHandlers[CreateUserCommand, User]):
    def handle(self, command: CreateUserCommand) -> r[User]:
        # Handler orchestrates
        validation_result = ValidateEmailService(email=command.email).execute()
        if validation_result.is_failure:
            return r[User].fail(validation_result.error or "Validation failed")

        # Service executes domain logic
        return CreateUserService(
            name=command.name,
            email=command.email,
        ).execute()
```

See [Service Patterns Guide](../guides/service-patterns.md) for service usage.

---

## Modernization Roadmap

### Current State (V1)

| Component       | Issue                            | Impact                |
| --------------- | -------------------------------- | --------------------- |
| FlextHandlers   | Manual `_metrics` dict           | Code duplication      |
| FlextHandlers   | Manual `_context_stack`          | Not using FlextMixins |
| FlextDispatcher | Managers hardcoded in `__init__` | No DI, hard to test   |

### Planned Phases

| Phase | Focus                             | Status      | Target   |
| ----- | --------------------------------- | ----------- | -------- |
| 0     | Document current stack            | ✅ Complete | Nov 2025 |
| 1     | `FlextMixins.CQRS` for metrics    | 🔴 Pending  | Dec 2025 |
| 2     | Dispatcher DI via FlextContainer  | 🔴 Pending  | Jan 2026 |
| 3     | Promote mixins to default usage   | 🔴 Pending  | Feb 2026 |
| 4     | Align with `FlextResult.and_then` | 🔴 Pending  | Mar 2026 |
| 5     | Zero-ceremony handler scaffolding | 🔴 Pending  | Apr 2026 |

### Phase 1: FlextMixins.CQRS

Proposed nested class in `mixins.py`:

```python
class FlextMixins:
    class CQRS:
        class MetricsTracker:
            def record(self, key: str, value: float) -> None: ...
            def get(self, key: str) -> float: ...
            def all(self) -> dict[str, float]: ...

        class ContextStack:
            def push(self, ctx: dict) -> None: ...
            def pop(self) -> dict | None: ...
            def current(self) -> dict: ...
```

### Phase 2: Dispatcher DI

Target API:

```python
container = FlextContainer.get_global()
container.register("circuit_breaker", CustomCircuitBreaker())

dispatcher = FlextDispatcher(container=container)
```

---

## Handler Patterns

### V1 Handler (Current Production)

The current handler pattern uses manual metrics and context management:

```python
class UpdateUserHandler(FlextHandlers[UpdateUserCommand, UserDto]):
    def handle(self, command: UpdateUserCommand) -> r[UserDto]:
        # Manual metrics tracking
        self._metrics["commands_processed"] = self._metrics.get("commands_processed", 0) + 1

        # Manual context management
        self.push_context({"command_id": command.id})
        try:
            domain_result = self._process(command)
            return r[UserDto].ok(domain_result)
        except Exception as exc:
            return r[UserDto].fail(str(exc))
        finally:
            self.pop_context()
```

### V2 Handler (Target - Phase 3+)

The target pattern uses `FlextMixins` infrastructure automatically:

```python
class UpdateUserHandler(FlextHandlers[UpdateUserCommand, UserDto]):
    def handle(self, command: UpdateUserCommand) -> r[UserDto]:
        # Automatic metrics via FlextMixins.CQRS
        self.cqrs_metrics.record("commands_processed", 1)

        # Automatic tracking via FlextMixins
        with self.track("handle_update_user"):
            result = self._process(command)

        return r[UserDto].ok(result)
```

### Migration Path

1. **Phase 1:** Add `cqrs_metrics` and `cqrs_context` properties to `FlextMixins.CQRS`
2. **Phase 2:** Deprecate `record_metric()`, `push_context()`, `pop_context()` with warnings
3. **Phase 3:** Update all handlers to use new patterns
4. **Phase 4:** Remove deprecated methods in v3.0

---

## Modernization Roadmap

### Current State (V1) vs Target (V2)

| Aspecto                   | V1 (Atual)                                | V2 (Target)                              |
| ------------------------- | ----------------------------------------- | ---------------------------------------- |
| **Métricas**              | `self._metrics` manual (50+ linhas)       | `self.cqrs_metrics` via FlextMixins.CQRS |
| **Contexto**              | `self._context_stack` manual (30+ linhas) | `self.context` via FlextMixins.CQRS      |
| **Logging**               | Inconsistente, pouco usado                | `self.logger` automático                 |
| **Tracking**              | Manual ou inexistente                     | `self.track()` automático                |
| **Managers (Dispatcher)** | Hardcoded (700+ linhas)                   | Injetados via FlextContainer             |
| **Circuit Breaker**       | `self._circuit_breaker` interno           | `container.get("circuit_breaker")`       |
| **Rate Limiter**          | `self._rate_limiter` interno              | `container.get("rate_limiter")`          |

### Timeline

```
V1 (Atual)           V2 Integration         V2 Complete
    │                      │                      │
    │  Manual metrics      │  FlextMixins.CQRS    │  Full observability
    │  Manual context      │  Container DI        │  Auto-discovery
    │  Hardcoded managers  │  Protocol-based      │  Zero ceremony
────┼──────────────────────┼──────────────────────┼─────────────────→
    │                      │                      │
 Nov 2025           Jan 2026 (Phase 1-2)    Mar 2026 (Phase 3-5)
```

### Problems Addressed

**FlextHandlers (Tier 3.1):**

- ❌ **50+ linhas** de métricas manuais (`self._metrics` dict)
- ❌ **30+ linhas** de contexto manual (`self._context_stack` list)
- ❌ **Logging não utilizado** (`self.logger` nunca chamado em `_run_pipeline`)
- ❌ **Tracking não utilizado** (`self.track()` nunca chamado)
- ❌ **Validação duplicada** entre handlers

**FlextDispatcher (Tier 3.2):**

- ❌ **700+ linhas** de managers hardcoded no `__init__`
- ❌ **Sem DI** - impossível injetar managers customizados
- ❌ **100+ linhas** de cache manual
- ⚠️ **Logging moderado** (18 chamadas) mas inconsistente
- ⚠️ **Tracking mínimo** (2 chamadas) insuficiente

**Impacto:**

- 🔴 Duplicação de código em 32+ projetos dependentes
- 🔴 Impossibilidade de customizar comportamento de reliability
- 🔴 Métricas inconsistentes entre projetos
- 🔴 Difícil debugging sem logging estruturado

### Solution Strategy

**FlextMixins.CQRS (Phase 1):**

1. Extrair métricas para `self.cqrs_metrics`
2. Extrair contexto para `self.context`
3. Integrar logging/tracking no pipeline
4. Deprecar métodos manuais com grace period

**FlextContainer DI (Phase 2):**

1. Definir protocols para managers
2. Extrair managers para módulo `_managers/`
3. Refatorar `FlextDispatcher.__init__()` para aceitar container
4. Registrar managers default no container

**Expected Benefits:**

- ✅ **Zero ceremony** - infraestrutura automática
- ✅ **Customização** - managers injetáveis via DI
- ✅ **Consistência** - métricas/logging unificados
- ✅ **Testabilidade** - mock de managers via container
- ✅ **Observabilidade** - tracking automático

---

## TODO Backlog

> Esta seção rastreia as pendências de modernização do CQRS. Veja também os TODOs nos docstrings dos arquivos de código.

| Item                                                                    | Fase    | Descrição                                            | Referência                   |
| ----------------------------------------------------------------------- | ------- | ---------------------------------------------------- | ---------------------------- |
| Migrar handlers para `self.logger`, `self.track`, e `self.cqrs_metrics` | Phase 3 | Substituir métricas/contexto manuais por FlextMixins | `handlers.py`                |
| Forçar construção do dispatcher via container                           | Phase 2 | Uma vez que todos os call sites migrarem             | `dispatcher.py`              |
| Atualizar `_dispatcher.reliability` para usar `FlextResult.and_then`    | Phase 4 | Paridade de nomenclatura                             | `_dispatcher/reliability.py` |
| Scaffolding CLI para handlers zero-ceremony                             | Phase 5 | Geração automática de handlers                       | CLI tools                    |

---

## Testing Guidance

### Test Structure

```
tests/
├── unit/
│   ├── test_handlers.py           # FlextHandlers unit tests
│   ├── test_dispatcher.py         # FlextDispatcher unit tests
│   └── test_managers/
│       ├── test_circuit_breaker.py
│       ├── test_rate_limiter.py
│       ├── test_timeout_enforcer.py
│       └── test_retry_policy.py
├── integration/
│   ├── test_dispatcher_handlers.py # Dispatcher + Handlers
│   ├── test_container_di.py        # DI integration
│   └── test_full_pipeline.py       # End-to-end tests
└── performance/
    ├── test_handler_throughput.py
    └── test_dispatcher_latency.py
```

### Running Tests

- Unit tests for handlers: `tests/unit/test_handlers.py`
- Unit tests for dispatcher: `tests/unit/test_dispatcher.py`
- Integration tests: `tests/integration/test_cqrs_pipeline.py`

Running isolated test files may fail the coverage gate (`fail-under=79`).
Execute the full suite for accurate coverage metrics.

### Performance Benchmarks

Target metrics for CQRS components:

| Component              | Metric  | Target   |
| ---------------------- | ------- | -------- |
| Handler throughput     | ops/sec | > 50,000 |
| Dispatcher avg latency | ms      | < 1.0    |
| Dispatcher P99 latency | ms      | < 5.0    |

### Success Metrics by Version

> Tracking modernization progress from current state through V3.

| Metric                   | Current | Target V2 | Target V3 |
| ------------------------ | ------- | --------- | --------- |
| Lines in FlextHandlers   | ~604    | ~500      | ~400      |
| Lines in FlextDispatcher | ~1200   | ~900      | ~700      |
| Code duplication %       | ~30%    | ~15%      | ~5%       |
| Coverage handlers.py     | 65%     | 85%       | 95%       |
| Coverage dispatcher.py   | 60%     | 80%       | 90%       |

---

## References

### Internal

- `flext_core/handlers.py` – Handler base class
- `flext_core/dispatcher.py` – Dispatcher implementation
- `flext_core/_dispatcher/` – Reliability managers
- `flext_core/mixins.py` – Infrastructure properties
- [Architecture Overview](./overview.md)
- [Architecture Patterns](./patterns.md)
- [Service Patterns Guide](../guides/service-patterns.md)

### External Resources

**CQRS Pattern:**

- [Martin Fowler - CQRS](https://martinfowler.com/bliki/CQRS.html)
- [Microsoft - CQRS Pattern](https://docs.microsoft.com/en-us/azure/architecture/patterns/cqrs)
- [Greg Young - CQRS Documents](https://cqrs.files.wordpress.com/2010/11/cqrs_documents.pdf)

**Reliability Patterns:**

- [Microsoft - Circuit Breaker Pattern](https://docs.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker)
- [Microsoft - Retry Pattern](https://docs.microsoft.com/en-us/azure/architecture/patterns/retry)
- [Netflix Hystrix (Circuit Breaker)](https://github.com/Netflix/Hystrix/wiki)
