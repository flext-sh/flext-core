# Architecture Overview

Comprehensive architecture guide for FLEXT-Core v0.9.9 - the foundation library implementing Clean Architecture principles with railway-oriented programming, dependency injection, and domain-driven design.

## Overview

FLEXT-Core follows **Clean Architecture** with clear separation of concerns across **five distinct layers** (enhanced v0.9.9 architecture with Layer 0 and Layer 0.5):

```

┌─────────────────────────────────────────────────────────────┐
│                  Layer 4: Application                        │
│   (Use cases, orchestration)                                │
│   FlextHandlers, FlextBus, FlextDispatcher                  │
│   FlextRegistry, FlextProcessors, FlextDecorators           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Layer 3: Domain                             │
│   (Business logic, entities)                                │
│   FlextModels, FlextService, FlextMixins, FlextUtilities    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Layer 2: Infrastructure                     │
│   (External concerns, I/O, frameworks)                      │
│   FlextConfig, FlextLogger, FlextContext                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Layer 1: Foundation                        │
│   (Core primitives, minimal dependencies)                   │
│   FlextResult, FlextContainer, FlextExceptions              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Layer 0.5: Integration Bridge                   │
│   (External library connectors, no Layer 1+ imports)        │
│   FlextRuntime - structlog, dependency_injector access      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Layer 0: Pure Constants                         │
│   (Zero dependencies, pure Python)                          │
│   FlextConstants, FlextTypes, FlextProtocols                │
└─────────────────────────────────────────────────────────────┘
```

**Dependency Rule**: Dependencies flow downward only (higher → lower). Lower layers know nothing about higher layers. This **strict hierarchy prevents circular dependencies** across the entire ecosystem.

## Layer Details

### Layer 0: Pure Constants (NEW in v0.9.9)

**Purpose**: Provide immutable constants and type definitions with **zero dependencies** (pure Python).

**Modules**:

| Module         | Coverage | Purpose                                               |
| -------------- | -------- | ----------------------------------------------------- |
| `constants.py` | 100%     | 50+ error codes, validation patterns, config defaults |
| `typings.py`   | 100%     | Type system (50+ TypeVars, protocols, aliases)        |
| `protocols.py` | 99%      | Runtime-checkable interfaces                          |

**Key Features**:

- ✅ **50+ Error Codes**: Categorized exception handling across ecosystem
- ✅ **Validation Patterns**: Email, URL, UUID, phone number regex patterns
- ✅ **Configuration Defaults**: Timeouts, network settings, logging levels
- ✅ **Platform Constants**: HTTP status codes, encodings, file paths
- ✅ **Complete Immutability**: All constants marked with `typing.Final`

**Design Principles**:

- Zero dependencies (not even internal flext_core imports)
- All constants immutable
- Foundation for entire ecosystem
- Eliminates circular dependencies

### Layer 0.5: Runtime Bridge (NEW in v0.9.9)

**Purpose**: Expose external libraries while maintaining proper dependency hierarchy.

**Modules**:

| Module       | Coverage | Purpose                                                       |
| ------------ | -------- | ------------------------------------------------------------- |
| `runtime.py` | N/A      | External library integration (structlog, dependency_injector) |

**Key Features**:

- ✅ **Type Guards**: Email, URL, UUID validation using Layer 0 patterns
- ✅ **Serialization**: JSON conversion with FLEXT defaults
- ✅ **External Libraries**: Direct access to structlog, dependency_injector
- ✅ **Structured Logging**: Pre-configured with FLEXT patterns
- ✅ **No Layer 1+ Imports**: Prevents circular dependencies

**Design Principles**:

- Bridge between pure constants and foundation
- No imports from Layer 1 or higher
- External library integration point
- Dependency hierarchy enforcement

### Layer 1: Foundation (Core Primitives)

**Purpose**: Provide fundamental building blocks with minimal dependencies (only Layers 0 and 0.5).

**Modules**:

| Module          | Coverage | Purpose                                        |
| --------------- | -------- | ---------------------------------------------- |
| `result.py`     | 95%      | Railway-oriented programming with Result monad |
| `container.py`  | 99%      | Dependency injection singleton                 |
| `exceptions.py` | 62%      | Exception hierarchy with error codes           |

**Key Patterns**:

1. **Railway Pattern** (`FlextResult[T]`):
   - Monadic error handling without exceptions
   - Dual `.value`/`.data` access (ABI stability guarantee)
   - Composable with `.map()`, `.flat_map()`, `.map_error()`
   - Type-safe success/failure states

2. **Dependency Injection** (`FlextContainer`):
   - Singleton pattern with global access
   - Typed service keys for type safety
   - Lifecycle management (register, get, reset)
   - Thread-safe operations

3. **Type System** (`FlextTypes`):
   - Comprehensive TypeVar collection (T, U, V, E, F, etc.)
   - Domain-specific types (TCommand, TQuery, TEvent)
   - Plugin system types (TPlugin, TPluginConfig, etc.)
   - Protocol definitions for runtime checks

**Design Principles**:

- No external dependencies
- Immutable by default
- Type-safe operations
- Zero runtime overhead

### Layer 2: Domain (Business Logic)

**Purpose**: Implement business rules and domain models independent of infrastructure.

**Modules**:

| Module         | Coverage | Purpose                                        |
| -------------- | -------- | ---------------------------------------------- |
| `models.py`    | 65%      | DDD patterns (Entity, Value, AggregateRoot)    |
| `service.py`   | 92%      | Domain service base class                      |
| `mixins.py`    | 57%      | Reusable behaviors (timestamps, serialization) |
| `utilities.py` | 66%      | Domain utilities (validation, conversion)      |

**Key Patterns**:

1. **Domain-Driven Design** (`FlextModels`):

   ```python
   # Entity - has identity
   class User(FlextModels.Entity):
       name: str
       email: str

   # Value Object - compared by value
   class Address(FlextModels.Value):
       street: str
       city: str

   # Aggregate Root - consistency boundary
   class Order(FlextModels.AggregateRoot):
       items: list[OrderItem]
       total: Decimal

       def add_item(self, item: OrderItem) -> FlextResult[None]:
           self.items.append(item)
           self.add_domain_event("ItemAdded", {"item_id": item.id})
           return FlextResult[None].ok(None)
   ```

2. **Domain Services** (`FlextService`):
   - Encapsulate business logic not belonging to entities
   - Pydantic Generic[T] base for validation
   - Return `FlextResult[T]` for all operations
   - Context-aware with FlextLogger integration

3. **Mixins** (`FlextMixins`):
   - Timestamp tracking (created_at, updated_at)
   - Serialization (to_dict, from_dict)
   - Validation helpers
   - Loggable behavior

**Design Principles**:

- Business logic isolation
- Framework independence
- Testable without infrastructure
- Pydantic v2 validation

### Layer 3: Application (Use Cases)

**Purpose**: Orchestrate business logic and coordinate between domain and infrastructure.

**Modules**:

| Module          | Coverage | Purpose                               |
| --------------- | -------- | ------------------------------------- |
| `bus.py`        | 94%      | Message bus with middleware pipeline  |
| `cqrs.py`       | 100%     | CQRS patterns (Command, Query, Event) |
| `handlers.py`   | 66%      | Handler registry and execution        |
| `dispatcher.py` | 45%      | Unified command/query dispatcher      |
| `registry.py`   | 91%      | Handler registry management           |
| `processors.py` | 56%      | Message processing orchestration      |

**Key Patterns**:

1. **CQRS** (`FlextBus`):

   ```python
   # Command - write operation
   class CreateUserCommand:
       name: str
       email: str

   # Query - read operation
   class GetUserQuery:
       user_id: str

   # Event - domain event
   class UserCreatedEvent:
       user_id: str
       timestamp: datetime

   # Bus usage
   bus = FlextBus()
   result = bus.execute(CreateUserCommand(name="Alice", email="alice@example.com"))
   ```

2. **Message Bus** (`FlextBus`):
   - Command/Query/Event routing
   - Middleware pipeline (validation, logging, caching)
   - Result caching with LRU strategy
   - Context propagation

3. **Handler Registry** (`FlextHandlers`, `FlextRegistry`):
   - Type-based handler registration
   - Handler discovery and resolution
   - Lifecycle management
   - Batch registration support

4. **Dispatcher** (`FlextDispatcher`):
   - Unified dispatch façade
   - Multi-handler support
   - Metadata propagation
   - Error aggregation

**Design Principles**:

- Use case orchestration
- No business logic (delegates to domain)
- Infrastructure-agnostic
- Composable pipelines

### Layer 4: Infrastructure (External Concerns)

**Purpose**: Provide interfaces to external systems, frameworks, and I/O.

**Modules**:

| Module         | Coverage | Purpose                                    |
| -------------- | -------- | ------------------------------------------ |
| `config.py`    | 90%      | Configuration management (env, TOML, YAML) |
| `loggings.py`  | 72%      | Structured logging with context            |
| `context.py`   | 66%      | Request/operation context tracking         |
| `protocols.py` | 99%      | Runtime-checkable interfaces               |

**Key Patterns**:

1. **Configuration** (`FlextConfig`):

   ```python
   class AppConfig(FlextConfig):
       """Application configuration with Pydantic validation."""
       app_name: str = "myapp"
       debug: bool = False
       database_url: str

   # Load from environment
   config = AppConfig()

   # Load for specific environment
   config = AppConfig()
   ```

2. **Structured Logging** (`FlextLogger`):

   ```python
   logger = FlextLogger(__name__)
   logger.info("User action", extra={
       "user_id": "user_123",
       "action": "login",
       "ip": "192.168.1.1"
   })
   ```

3. **Context Management** (`FlextContext`):

   ```python
   # Set context
   FlextContext.Request.set_correlation_id("req_123")
   FlextContext.Request.set_user_id("user_456")

   # Context propagates automatically to loggers
   logger.info("Processing request")  # Includes correlation_id
   ```

4. **Protocols** (`FlextProtocols`):
   - Runtime-checkable interfaces
   - Type narrowing with isinstance()
   - No hasattr() checks needed
   - Protocol-based polymorphism

**Design Principles**:

- External system abstraction
- Framework independence
- Dependency inversion
- Easy testing/mocking

## Cross-Cutting Concerns

### Error Handling Strategy

**Primary**: Railway-oriented programming with `FlextResult[T]`

```python
def operation() -> FlextResult[Data]:
    # Success case
    return FlextResult[Data].ok(data)

    # Failure case
    return FlextResult[Data].fail("Error message", error_code="ERROR_CODE")
```

**Secondary**: Exceptions for truly exceptional cases

- Use exceptions for programming errors
- Use FlextResult for expected failures
- Never catch generic Exception in business logic

### Dependency Flow

```
Infrastructure → Application → Domain → Foundation
     (outer)         ↓          ↓         (inner)

Only outer layers depend on inner layers, never reversed.
```

**Example**:

- `FlextLogger` (Infrastructure) depends on `FlextContext` (Infrastructure)
- `FlextBus` (Application) depends on `FlextResult` (Foundation)
- `FlextModels` (Domain) depends on `FlextResult` (Foundation)
- `FlextResult` (Foundation) depends on nothing

### Testing Strategy

**Foundation Layer**:

- Pure unit tests
- No mocks needed
- 100% coverage target

**Domain Layer**:

- Domain logic tests
- Minimal infrastructure
- Focus on business rules

**Application Layer**:

- Integration tests
- Test use case flows
- Mock infrastructure

**Infrastructure Layer**:

- Integration tests
- Real external systems (Docker)
- No mocks (use flext_tests)

## Module Dependency Graph

```
FlextResult ←─────────────────────┐
    ↑                              │
    │                              │
FlextContainer                     │
    ↑                              │
    │                              │
FlextModels ←────── FlextService   │
    ↑                   ↑          │
    │                   │          │
    |          FlextHandlers       │
    ↑              ↑               │
    │              │               │
FlextBus ──────────┘               │
    ↑                              │
    │                              │
FlextDispatcher                    │
    ↑                              │
    │                              │
FlextConfig ───────────────────────┘
FlextLogger
FlextContext
```

**Key Points**:

- No circular dependencies
- Clear unidirectional flow
- Foundation has zero dependencies
- Infrastructure depends on everything

## Design Patterns Used

### Creational Patterns

1. **Singleton**: `FlextContainer.get_global()`
2. **Factory Method**: `FlextResult.ok()`, `FlextResult.fail()`
3. **Builder**: Pydantic model construction

### Structural Patterns

1. **Façade**: `FlextDispatcher` (simplifies FlextBus/Registry)
2. **Composite**: `FlextModels.AggregateRoot` with child entities
3. **Decorator**: Middleware pipeline in `FlextBus`

### Behavioral Patterns

1. **Strategy**: Pluggable handlers in `FlextRegistry`
2. **Observer**: Domain events in `FlextModels.AggregateRoot`
3. **Chain of Responsibility**: Middleware pipeline
4. **Template Method**: `FlextService` base class

### Functional Patterns

1. **Monad**: `FlextResult[T]` railway pattern
2. **Functor**: `.map()` operations
3. **Applicative**: `.flat_map()` (monadic bind)

## Quality Metrics

### Current State (v0.9.9)

| Metric              | Value | Target (1.0.0) |
| ------------------- | ----- | -------------- |
| **Test Coverage**   | 80%   | 79%+           |
| **Total Tests**     | 1,235 | 1,500+         |
| **Passing Tests**   | 1,143 | 1,500+         |
| **Failed Tests**    | 92    | 0              |
| **Ruff Violations** | 0     | 0              |
| **Type Errors**     | 1,743 | 0              |
| **Modules**         | 20    | 20 (stable)    |
| **Public Exports**  | 60+   | 60+ (locked)   |

### Coverage by Layer

| Layer              | Coverage | Status       |
| ------------------ | -------- | ------------ |
| **Foundation**     | 92%      | ✅ Excellent |
| **Domain**         | 71%      | ✅ Good      |
| **Application**    | 75%      | ✅ Good      |
| **Infrastructure** | 79%      | ✅ Good      |

## Phase 1 Context Enrichment (v0.9.9)

**Status**: ✅ **COMPLETED** - Major architectural enhancement providing zero-boilerplate context management for distributed tracing and audit trails.

### New FlextService Capabilities

Automatic context enrichment for structured logging and distributed tracing:

```python
from flext_core import FlextService, FlextTypes, FlextResult

class PaymentService(FlextService[FlextTypes.Dict]):
    """Service with automatic context enrichment."""

    def process_payment(self, payment_id: str, amount: float, user_id: str) -> FlextResult[dict]:
        # Generate correlation ID for distributed tracing
        correlation_id = self._with_correlation_id()

        # Set user context for audit trail
        self._with_user_context(user_id, payment_id=payment_id)

        # Set operation context for tracking
        self._with_operation_context("process_payment", amount=amount)

        # All logs now include full context automatically
        self.logger.info("Processing payment", payment_id=payment_id, amount=amount)

        return FlextResult[dict].ok({"status": "completed", "correlation_id": correlation_id})
```

### Complete Automation Helper

`FlextService.execute_with_context_enrichment()` provides full automation:

```python
class OrderService(FlextService[Order]):
    def process_order(self, order_id: str, customer_id: str, correlation_id: str | None = None) -> FlextResult[Order]:
        return self.execute_with_context_enrichment(
            operation_name="process_order",
            correlation_id=correlation_id,
            user_id=customer_id,
            order_id=order_id,
        )
        # Automatically handles: correlation ID, user context, operation tracking, logging, cleanup
```

### Architectural Benefits

- ✅ **Zero Boilerplate** - No manual context setup required
- ✅ **Distributed Tracing** - Automatic correlation ID generation
- ✅ **Audit Trail** - User context automatically captured
- ✅ **Ecosystem Ready** - Available to all 32+ dependent projects
- ✅ **Performance Tracking** - Operation lifecycle monitoring
- ✅ **Clean Architecture** - Maintains layer separation while providing cross-cutting concerns

See `examples/automation_showcase.py` for complete working examples.

## Extension Points

### Adding New Features

1. **New Domain Entity**:
   - Extend `FlextModels.Entity` or `Value`
   - Implement validation in `model_post_init`
   - Add to domain service

2. **New Command/Query**:
   - Define message class
   - Create handler
   - Register with `FlextBus` or `FlextDispatcher`

3. **New Middleware**:
   - Implement middleware callable
   - Register with `FlextBus.apply_middleware()`

4. **New Context Type**:
   - Extend `FlextContext` with new scope
   - Update `FlextLogger` integration

## Performance Considerations

### Bottlenecks

1. **FlextResult operations**: Sub-microsecond (negligible)
2. **Container lookups**: O(1) dictionary access
3. **Handler resolution**: O(1) with type-based keys
4. **Middleware pipeline**: O(n) where n = middleware count

### Optimization Guidelines

1. Use middleware sparingly (each adds overhead)
2. Cache handler lookups when possible
3. Use `FlextResult` early returns to avoid unnecessary work
4. Batch operations when using `FlextRegistry`

## Migration Guidelines

### From Other Patterns

**From try/except to FlextResult**:

```python
# Before
def old_way():
    try:
        result = risky_operation()
        return result
    except Exception as e:
        return None

# After
def new_way() -> FlextResult[Data]:
    result = risky_operation()
    if result is None:
        return FlextResult[Data].fail("Operation failed")
    return FlextResult[Data].ok(result)
```

**From global variables to FlextContainer**:

```python
# Before
LOGGER = logging.getLogger(__name__)

# After
container = FlextContainer.get_global()
logger_result = container.get("logger")
logger = logger_result.unwrap() if logger_result.is_success else FlextLogger(__name__)
```

## References

- **Clean Architecture**: Robert C. Martin
- **Domain-Driven Design**: Eric Evans
- **Railway-Oriented Programming**: Scott Wlaschin
- **Functional Programming**: Haskell/F# patterns

## Appendix: Module Responsibilities

### Foundation

- **result.py**: Monad, error handling, composition
- **container.py**: DI, service lifecycle
- **typings.py**: Type system, TypeVars
- **constants.py**: Enums, error codes
- **exceptions.py**: Exception hierarchy

### Domain

- **models.py**: Entities, values, aggregates
- **service.py**: Domain service base
- **mixins.py**: Reusable behaviors
- **utilities.py**: Validation, conversion

### Application

- **bus.py**: Message routing, middleware
- **cqrs.py**: Command/Query patterns
- **handlers.py**: Handler execution
- **dispatcher.py**: Unified façade
- **registry.py**: Handler registry
- **processors.py**: Message processing

### Infrastructure

- **config.py**: Configuration loading
- **loggings.py**: Structured logging
- **context.py**: Context management
- **protocols.py**: Runtime interfaces

---

**FLEXT-Core Architecture** - Clean, testable, and maintainable foundation for enterprise applications.
