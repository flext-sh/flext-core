# CQRS Architecture

<!-- TOC START -->

- [Overview](#overview)
- [FlextHandlers](#flexthandlers)
  - [Current Implementation (V1)](#current-implementation-v1)
  - [Execution Pipeline](#execution-pipeline)
  - [Handler Configuration](#handler-configuration)
  - [Metrics and Context (V1 – Manual)](#metrics-and-context-v1-manual)
- [FlextDispatcher](#flextdispatcher)
  - [Current Implementation (V1)](#current-implementation-v1)
  - [Reliability Patterns](#reliability-patterns)
  - [Dispatch Flow](#dispatch-flow)
  - [Handler Registration](#handler-registration)
- [Integration with FlextService](#integration-with-flextservice)
- [Modernization Roadmap](#modernization-roadmap)
  - [Current State (V1)](#current-state-v1)
  - [Planned Phases](#planned-phases)
  - [Phase 1: FlextMixins.CQRS](#phase-1-flextmixinscqrs)
  - [Phase 2: Dispatcher DI](#phase-2-dispatcher-di)
- [Handler Patterns](#handler-patterns)
  - [V1 Handler (Current Production)](#v1-handler-current-production)
  - [V2 Handler (Target - Phase 3+)](#v2-handler-target-phase-3)
  - [Migration Path](#migration-path)
- [Modernization Roadmap](#modernization-roadmap)
  - [Current State (V1) vs Target (V2)](#current-state-v1-vs-target-v2)
  - [Timeline](#timeline)
  - [Problems Addressed](#problems-addressed)
  - [Solution Strategy](#solution-strategy)
- [TODO Backlog](#todo-backlog)
- [Testing Guidance](#testing-guidance)
  - [Test Structure](#test-structure)
  - [Running Tests](#running-tests)
  - [Performance Benchmarks](#performance-benchmarks)
  - [Success Metrics by Version](#success-metrics-by-version)
- [References](#references)
  - [Internal](#internal)
  - [External Resources](#external-resources)
- [Next Steps](#next-steps)
- [See Also](#see-also)
- [Verification Commands](#verification-commands)

<!-- TOC END -->

**Status**: Production Ready | **Version**: 0.10.0 | **Date**: 2025-12-07
**Python:** 3.13+ | **Pydantic:** 2.x

This document describes the Command Query Responsibility Segregation (CQRS)
implementation in flext-core, including the handler pipeline, dispatcher
orchestration, and reliability patterns.

Canonical references:

- `./overview.md`
- `./clean-architecture.md`
- `../../README.md`

______________________________________________________________________

## Overview

FLEXT-Core implements CQRS through two primary components:

- **`FlextHandlers`** (`handlers.py`) – Base class for message handlers
- **`FlextDispatcher`** (`dispatcher.py`) – Orchestration and routing

Both components follow railway-oriented programming with `FlextResult` and
integrate with the infrastructure provided by `FlextMixins`.

````text
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
```text

______________________________________________________________________

## FlextHandlers

### Current Implementation (V1)

Handlers derive from `FlextHandlers[MessageT, ResultT]` and implement the
abstract `handle()` method:

```python
from flext_core import FlextHandlers
from flext_core import r

class CreateUserHandler(FlextHandlers[CreateUserCommand, User]):
    def handle(self, command: CreateUserCommand) -> r[User]:
        # Business logic
        user = User(name=command.name, email=command.email)
        return r[User].ok(user)
```text

### Execution Pipeline

The `execute()` method orchestrates validation and handling:

```text
execute(message)
    │
    ├─► validate(message)
    │       └─► Returns r[bool] (fail fast on error)
    │
    └─► handle(message)
            └─► Returns r[ResultT]
```text

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
```text

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
```text

> **TODO(handlers.py::FlextHandlers):** Migrate to `FlextMixins.CQRS` utilities
> for metrics and context once Phase 1 of CQRS modernization lands. See
> Modernization Roadmap.

______________________________________________________________________

## FlextDispatcher

### Current Implementation (V1)

The dispatcher initializes reliability managers internally:

```python
from flext_core import FlextDispatcher

dispatcher = FlextDispatcher()
dispatcher.register_handler(CreateUserCommand, CreateUserHandler())
result = dispatcher.dispatch(CreateUserCommand(name="Alice", email="alice@example.com"))
```text

### Reliability Patterns

The dispatcher applies layered reliability controls:

| Pattern         | Manager Class           | Configuration Source       |
| --------------- | ----------------------- | -------------------------- |
| Circuit Breaker | `CircuitBreakerManager` | `config.circuit_breaker_*` |
| Rate Limiting   | `RateLimiterManager`    | `config.rate_limit_*`      |
| Retry           | `RetryPolicy`           | `config.max_retry_*`       |
| Timeout         | `TimeoutEnforcer`       | `config.enable_timeout_*`  |

### Dispatch Flow

```text
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
```text

### Handler Registration

```python
# Register by message type
dispatcher.register_handler(CreateUserCommand, handler)

# Register with explicit mode
dispatcher.register_command(CreateUserCommand, handler)
dispatcher.register_query(GetUserQuery, handler)
dispatcher.register_event(UserCreatedEvent, handler)
```text

> **TODO(dispatcher.py::FlextDispatcher.**init**):** Accept `container` parameter
> for dependency injection of reliability managers. See Phase 2 of
> Modernization Roadmap.

______________________________________________________________________

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
```text

See Service Patterns Guide for service usage.

______________________________________________________________________

## Modernization Roadmap - Phase Overview

### Current State (V1) - Phase Overview

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
```text

### Phase 2: Dispatcher DI

Target API:

```python
container = FlextContainer.get_global()
container.register("circuit_breaker", CustomCircuitBreaker())

dispatcher = FlextDispatcher(container=container)
```text

______________________________________________________________________

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
```text

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
```text

### Migration Path

1. **Phase 1:** Add `cqrs_metrics` and `cqrs_context` properties to `FlextMixins.CQRS`
1. **Phase 2:** Deprecate `record_metric()`, `push_context()`, `pop_context()` with warnings
1. **Phase 3:** Update all handlers to use new patterns
1. **Phase 4:** Remove deprecated methods in v3.0

______________________________________________________________________

## Modernization Roadmap - Detailed Strategy

### Current State (V1) vs Target (V2) - Detailed

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

```text
V1 (Atual)           V2 Integration         V2 Complete
    │                      │                      │
    │  Manual metrics      │  FlextMixins.CQRS    │  Full observability
    │  Manual context      │  DI        │  Auto-discovery
    │  Hardcoded managers  │  Protocol-based      │  Zero ceremony
────┼──────────────────────┼──────────────────────┼─────────────────→
    │                      │                      │
 Nov 2025           Jan 2026 (Phase 1-2)    Mar 2026 (Phase 3-5)
```text

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

1. Extract metrics to `self.cqrs_metrics`
1. Extract context to `self.context`
1. Integrate logging/tracking in the pipeline
1. Deprecate manual methods with grace period

**FlextDI (Phase 2):**

1. Define protocols for managers
1. Extract managers to `_managers/` module
1. Refactor `FlextDispatcher.__init__()` to accept container
1. Register default managers in container

**Expected Benefits:**

- ✅ **Zero ceremony** - automatic infrastructure
- ✅ **Customization** - injectable managers via DI
- ✅ **Consistency** - unified metrics/logging
- ✅ **Testability** - mock managers via container
- ✅ **Observability** - automatic tracking

______________________________________________________________________

## TODO Backlog

> This section tracks CQRS modernization backlog items. See also TODOs in code docstrings.

| Item                                                                     | Phase   | Description                                     | Reference                    |
| ------------------------------------------------------------------------ | ------- | ----------------------------------------------- | ---------------------------- |
| Migrate handlers to `self.logger`, `self.track`, and `self.cqrs_metrics` | Phase 3 | Replace manual metrics/context with FlextMixins | `handlers.py`                |
| Force dispatcher construction via container                              | Phase 2 | Once all call sites migrate                     | `dispatcher.py`              |
| Update `_dispatcher.reliability` to use `FlextResult.and_then`           | Phase 4 | Naming parity                                   | `_dispatcher/reliability.py` |
| Scaffolding CLI for zero-ceremony handlers                               | Phase 5 | Automatic handler generation                    | CLI tools                    |

______________________________________________________________________

## Testing Guidance

### Test Structure

```text
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
```text

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

______________________________________________________________________

## References

### Internal

- `flext_core/handlers.py` – Handler base class
- `flext_core/dispatcher.py` – Dispatcher implementation
- `flext_core/_dispatcher/` – Reliability managers
- `flext_core/mixins.py` – Infrastructure properties
- Architecture Overview
- Architecture Patterns
- Service Patterns Guide

### External Resources

**CQRS Pattern:**

- [Martin Fowler - CQRS](https://martinfowler.com/bliki/CQRS.html)
- [Microsoft - CQRS Pattern](https://docs.microsoft.com/en-us/azure/architecture/patterns/cqrs)
- [Greg Young - CQRS Documents](https://cqrs.files.wordpress.com/2010/11/cqrs_documents.pdf)

**Reliability Patterns:**

- [Microsoft - Circuit Breaker Pattern](https://docs.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker)
- [Microsoft - Retry Pattern](https://docs.microsoft.com/en-us/azure/architecture/patterns/retry)
- [Netflix Hystrix (Circuit Breaker)](https://github.com/Netflix/Hystrix/wiki)

## Next Steps

1. **Clean Architecture**: Review Clean Architecture for layer boundaries
1. **Architecture Overview**: See Architecture Overview for layer topology
1. **Service Patterns**: Check Service Patterns Guide for handler implementation
1. **Dependency Injection**: See Advanced DI Guide for dispatcher configuration
1. **Railway Patterns**: Review Railway-Oriented Programming for result composition

## See Also

- Clean Architecture - Layer responsibilities and dependency rules
- Architecture Overview - Visual layer layout and execution flows
- Architecture Patterns - Common CQRS and handler patterns
- Service Patterns Guide - Handler and service implementation
- Dependency Injection Advanced - Dispatcher reliability configuration
- Railway-Oriented Programming - Result composition patterns
- `../../README.md`: architecture principles and development workflow entrypoint

## Verification Commands

Run from `flext-core/`:

```bash
make lint
make type-check
make test-fast
```text
````
