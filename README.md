# FLEXT-Core

<p align="center">
  <strong>Enterprise Foundation Framework for Python 3.13+</strong><br>
  Railway-Oriented Programming · Dependency Injection · CQRS · Domain-Driven Design
</p>

<p align="center">
  <a href="https://github.com/flext-sh/flext-core/actions/workflows/ci.yml">
    <img src="https://github.com/flext-sh/flext-core/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://pypi.org/project/flext-core/">
    <img src="https://img.shields.io/pypi/v/flext-core.svg" alt="PyPI version">
  </a>
  <a href="https://pypi.org/project/flext-core/">
    <img src="https://img.shields.io/pypi/pyversions/flext-core.svg" alt="Python versions">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT">
  </a>
</p>

---

**FLEXT-Core** is the foundational library for the [FLEXT](https://github.com/flext-sh/flext) ecosystem. It provides a cohesive set of architectural primitives for building enterprise-grade Python applications using Clean Architecture, enforcing type safety from the ground up with Python 3.13.

> **Python 3.13+ required.** FLEXT-Core is intentionally forward-only, leveraging the latest type system improvements and performance gains.

---

<!-- TOC START -->

- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Core Modules](#core-modules)
  - [FlextResult — Railway-Oriented Programming](#flextresult--railway-oriented-programming)
  - [FlextModels — Domain-Driven Design](#flextmodels--domain-driven-design)
  - [FlextContainer — Dependency Injection](#flextcontainer--dependency-injection)
  - [FlextDispatcher — CQRS](#flextdispatcher--cqrs)
  - [FlextHandlers — Handler Base Classes](#flexthandlers--handler-base-classes)
  - [FlextService — Domain Services](#flextservice--domain-services)
  - [FlextLogger — Structured Logging](#flextlogger--structured-logging)
  - [FlextSettings — Configuration](#flextsettings--configuration)
  - [FlextContext — Context Propagation](#flextcontext--context-propagation)
  - [FlextDecorators — Cross-Cutting Concerns](#flextdecorators--cross-cutting-concerns)
  - [FlextRegistry — Handler Discovery](#flextregistry--handler-discovery)
  - [FlextUtilities — Utility Toolbox](#flextutilities--utility-toolbox)
- [Runtime Aliases](#runtime-aliases)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

<!-- TOC END -->

---

## Key Features

| Feature | Description |
|---|---|
| **Railway-Oriented Programming** | `FlextResult[T]` replaces exceptions with composable success/failure values. Never lose an error path. |
| **Type-Safe DI Container** | `FlextContainer` manages singletons, factories, and resources with thread-safe initialization. |
| **CQRS Dispatcher** | `FlextDispatcher` + `FlextRegistry` route commands, queries, and domain events to typed handlers. |
| **Domain-Driven Design** | `FlextModels` provides `Entity`, `Value`, `AggregateRoot`, `Command`, `Query`, and `DomainEvent` base types. |
| **Protocol-First Architecture** | All contracts are `Protocol`-based — depend on abstractions, not implementations. |
| **Structured Logging** | `FlextLogger` wraps `structlog` with scoped contexts, correlation IDs, and performance tracking. |
| **Configuration Management** | `FlextSettings` uses Pydantic v2 `BaseSettings` with environment variable support and runtime validation. |
| **Context Propagation** | `FlextContext` carries correlation metadata, request data, and timing across CQRS pipelines. |
| **Decorator Toolkit** | `FlextDecorators` provides `@inject`, `@log_operation`, `@railway`, `@with_context`, and `@retry`. |
| **Reliability Patterns** | Built-in retry, timeout, backoff, and circuit-breaker utilities via `FlextUtilities`. |
| **Zero `Any` Tolerance** | All public APIs are fully typed. No `Any`, no `# type: ignore`. |

---

## Installation

```bash
# pip
pip install flext-core

# Poetry
poetry add flext-core

# uv
uv add flext-core
```

**Requirements**: Python 3.13+

---

## Quick Start

```python
from flext_core import r, m, FlextContainer, FlextService

# 1. Railway-Oriented Result
def validate_email(email: str) -> r[str]:
    if "@" not in email:
        return r[str].fail("Invalid email address")
    return r[str].ok(email)

result = validate_email("alice@example.com")
if result.is_success:
    print(f"Valid: {result.value}")   # Valid: alice@example.com
else:
    print(f"Error: {result.error}")

# 2. Chain operations (monadic composition)
final = (
    validate_email("alice@example.com")
    .map(str.lower)
    .flat_map(lambda e: r[str].ok(f"Welcome, {e}"))
)
print(final.value)  # Welcome, alice@example.com

# 3. Domain Model
class User(m.Entity):
    name: str
    email: str

user = User(name="Alice", email="alice@example.com")
print(user.entity_id)   # auto-generated UUID

# 4. Domain Service
class UserService(FlextService[User]):
    def execute(self) -> r[User]:
        return r[User].ok(User(name="Alice", email="alice@example.com"))

service = UserService()
result = service.execute()
print(result.value.name)  # Alice
```

---

## Architecture Overview

FLEXT-Core is structured around **Clean Architecture** and **SOLID principles**, organized into strict dependency layers:

```
+----------------------------------------------------------+
|  Layer 3 -- Application                                  |
|  FlextHandlers  FlextService  FlextDecorators            |
|  FlextDispatcher  FlextRegistry  FlextMixins             |
+----------------------------------------------------------+
|  Layer 2 -- Infrastructure                               |
|  FlextLogger  FlextContainer  FlextContext               |
+----------------------------------------------------------+
|  Layer 1 -- Foundation                                   |
|  FlextResult  FlextModels  FlextProtocols                |
|  FlextTypes  FlextExceptions  FlextUtilities             |
+----------------------------------------------------------+
|  Layer 0.5 -- Integration Bridge                         |
|  FlextRuntime  FlextSettings                             |
+----------------------------------------------------------+
|  Layer 0 -- Constants                                    |
|  FlextConstants  FlextTypes (primitives)                 |
+----------------------------------------------------------+
```

**Design principles enforced at every layer:**

- **Protocols First** — all contracts use `typing.Protocol`; no concrete coupling.
- **Strict Dependency Direction** — upper layers depend on lower layers, never the reverse.
- **Railway Pattern** — business logic returns `FlextResult`, never raises exceptions.
- **Zero `Any`** — all APIs carry complete type information.

---

## Core Modules

### FlextResult — Railway-Oriented Programming

`FlextResult[T]` (alias `r`) models operation outcomes as composable success/failure values, eliminating surprise exceptions in business logic.

```python
from flext_core import r

def divide(a: float, b: float) -> r[float]:
    if b == 0:
        return r[float].fail("Division by zero")
    return r[float].ok(a / b)

# Compose with map / flat_map
result = (
    divide(10, 2)
    .map(lambda x: x * 100)
    .map(lambda x: f"{x:.0f}%")
)
print(result.value)  # 500%

# Error recovery
safe = divide(1, 0).unwrap_or(0.0)

# Transform errors
guarded = divide(1, 0).map_error(lambda e: f"Calc failed: {e}")
```

**Key API:**

| Method | Description |
|---|---|
| `r[T].ok(value)` | Wrap a success value |
| `r[T].fail("msg")` | Wrap a failure message |
| `result.is_success` | Check outcome |
| `result.value` | Extract success value |
| `result.error` | Extract error message |
| `result.map(fn)` | Transform success value |
| `result.flat_map(fn)` | Chain result-returning operations |
| `result.map_error(fn)` | Transform error message |
| `result.unwrap_or(default)` | Provide fallback on failure |

---

### FlextModels — Domain-Driven Design

`FlextModels` (alias `m`) exposes Pydantic v2-backed base classes for rich domain modeling.

```python
from flext_core import m
from pydantic import Field

class Money(m.Value):
    """Immutable value object — compared by value, not identity."""
    amount: float = Field(gt=0)
    currency: str = Field(min_length=3, max_length=3)

class Order(m.Entity):
    """Entity — compared by identity, carries domain events."""
    customer_id: str
    total: Money

class OrderPlaced(m.DomainEvent):
    """Domain event emitted after state transition."""
    order_id: str

# AggregateRoot enforces consistency boundaries
class OrderAggregate(m.AggregateRoot):
    order: Order

    def place(self) -> None:
        self.add_domain_event(OrderPlaced(order_id=self.order.entity_id))
```

**Available base classes:**

| Class | Description |
|---|---|
| `m.Entity` | Identity-based equality, auto UUID, timestamps, domain events |
| `m.Value` | Value-based equality, immutable |
| `m.AggregateRoot` | Consistency boundary, event collection |
| `m.Command` | CQRS command with auto correlation ID |
| `m.Query` | CQRS query with pagination support |
| `m.DomainEvent` | Typed domain event with metadata |

---

### FlextContainer — Dependency Injection

`FlextContainer` provides a thread-safe singleton DI container built on `dependency-injector`.

```python
from flext_core import FlextContainer, FlextLogger

container = FlextContainer.get_global()

# Register a singleton
logger = FlextLogger(__name__)
container.register("logger", logger, singleton=True)

# Register a factory (new instance per resolution)
container.register(kind="factory", name="session", factory=lambda: DatabaseSession())

# Resolve
result = container.get("logger")
if result.is_success:
    result.value.info("Ready")

# Scoped container for tests
with FlextContainer.scoped() as scope:
    scope.register("mock_db", FakeDatabase())
    # ... test code ...
```

---

### FlextDispatcher — CQRS

`FlextDispatcher` routes messages (commands, queries, events) to their registered handlers.

```python
from flext_core import FlextDispatcher, FlextRegistry, FlextService, r, m

class CreateOrder(m.Command):
    customer_id: str
    items: list[str]

class OrderService(FlextService[str]):
    def handle_create(self, cmd: CreateOrder) -> r[str]:
        if not cmd.items:
            return r[str].fail("Order must contain at least one item")
        return r[str].ok(f"Order created for {cmd.customer_id}")

    def execute(self) -> r[str]:
        return r[str].fail("Use handle_create instead")

registry = FlextRegistry()
service = OrderService()
registry.register_command(CreateOrder, service.handle_create)

dispatcher = FlextDispatcher(registry=registry)
result = dispatcher.dispatch(CreateOrder(customer_id="C-001", items=["SKU-1"]))
print(result.value)  # Order created for C-001
```

---

### FlextHandlers — Handler Base Classes

`FlextHandlers[MessageT, ResultT]` (alias `h`) is the generic base for all CQRS handlers with built-in validation and railway execution.

```python
from flext_core import h, r, m

class GetUserQuery(m.Query):
    user_id: str

class GetUserHandler(h[GetUserQuery, str]):
    def validate(self, message: GetUserQuery) -> r[GetUserQuery]:
        if not message.user_id:
            return r[GetUserQuery].fail("user_id is required")
        return r[GetUserQuery].ok(message)

    def execute(self, message: GetUserQuery) -> r[str]:
        # fetch user from repository
        return r[str].ok(f"User:{message.user_id}")
```

---

### FlextService — Domain Services

`FlextService[T]` (alias `s`) is the base class for domain services with built-in DI access, logging, and context management.

```python
from flext_core import FlextService, r

class NotificationService(FlextService[bool]):
    def execute(self) -> r[bool]:
        self.logger.info("Sending notification")
        return r[bool].ok(True)

    def send(self, recipient: str, message: str) -> r[bool]:
        if not recipient:
            return r[bool].fail("Recipient required")
        self.logger.info("notification_sent", recipient=recipient)
        return r[bool].ok(True)
```

---

### FlextLogger — Structured Logging

`FlextLogger` wraps `structlog` with scoped contexts, correlation IDs, and performance-tracking integration.

```python
from flext_core import FlextLogger

logger = FlextLogger(__name__)

# Structured key-value logging
logger.info("user_created", user_id="123", plan="pro")
logger.warning("quota_exceeded", user_id="123", limit=100)
logger.error("payment_failed", user_id="123", error="card_declined")

# Scoped context (auto-cleared on exit)
with logger.bind(request_id="req-abc"):
    logger.info("processing")  # includes request_id automatically

# Performance tracking
with logger.log_operation("db_query", track_perf=True):
    pass  # duration logged on exit
```

---

### FlextSettings — Configuration

`FlextSettings` extends Pydantic `BaseSettings` with thread-safe singleton access, environment variable loading (`FLEXT_` prefix), and DI integration.

```python
from flext_core import FlextSettings

class AppConfig(FlextSettings):
    database_url: str = "postgresql://localhost/mydb"
    debug: bool = False
    max_workers: int = 4

# Load from environment (FLEXT_DATABASE_URL, FLEXT_DEBUG, ...)
config = AppConfig.get_global()
print(config.database_url)

# Runtime overrides (useful in tests)
config = AppConfig.get_global(overrides={"debug": True})
```

---

### FlextContext — Context Propagation

`FlextContext` carries correlation metadata, request identities, and timing information across the entire CQRS pipeline using `contextvars`.

```python
from flext_core import FlextContext

# Set correlation context for the current async/thread scope
FlextContext.set_correlation_id("corr-001")
FlextContext.set_service_name("order-service")

# Access anywhere in the call stack
ctx = FlextContext.current()
print(ctx.get("correlation_id"))  # corr-001

# Scoped context manager (auto-restored on exit)
with FlextContext.scope(user_id="user-123", operation="checkout"):
    # all log statements include user_id and operation
    pass
```

---

### FlextDecorators — Cross-Cutting Concerns

`FlextDecorators` (alias `d`) provides composable decorators that automate infrastructure boilerplate.

```python
from flext_core import d

# Auto-inject dependencies from FlextContainer
@d.inject
def process_order(order_id: str, db=d.Provide["db_session"]):
    return db.find(order_id)

# Log entry/exit with structured metadata
@d.log_operation(name="create_user", track_perf=True)
def create_user(email: str):
    ...

# Wrap return value in FlextResult automatically
@d.railway
def risky_operation(value: int) -> int:
    if value < 0:
        raise ValueError("Negative value")
    return value * 2

# Retry with exponential backoff
@d.retry(attempts=3, delay=0.5, backoff=2.0)
def call_external_api():
    ...
```

---

### FlextRegistry — Handler Discovery

`FlextRegistry` maintains the handler-to-message-type mapping consumed by `FlextDispatcher`.

```python
from flext_core import FlextRegistry, r, m

registry = FlextRegistry()

# Register individual handlers
registry.register_command(CreateOrder, handle_create_order)
registry.register_query(GetOrderQuery, handle_get_order)

# Batch registration with summary
summary = registry.register_batch([
    (CreateOrder, handle_create_order),
    (GetOrderQuery, handle_get_order),
])
print(f"Registered: {len(summary.registered)}, Failed: {len(summary.failed)}")
```

---

### FlextUtilities — Utility Toolbox

`FlextUtilities` (alias `u`) is a flat-namespace utility facade covering common operational needs.

```python
from flext_core import u

# Type-safe parsing
value = u.parse("42", int)          # 42
items = u.parse("[1,2,3]", list)    # [1, 2, 3]

# Safe dict access
name = u.get(data, "user.name", default="anonymous")

# Collection helpers
batches = u.batch(items, size=100)   # generator of 100-item chunks
mapped = u.map(items, transform_fn)

# Type guards
u.is_non_empty_string("hello")  # True
u.is_dict_like({})              # True
```

**Utility sub-modules:** `args`, `cache`, `checker`, `collection`, `configuration`, `context`, `conversion`, `deprecation`, `domain`, `enum`, `generators`, `guards`, `mapper`, `model`, `pagination`, `parser`, `reliability`, `text`.

---

## Runtime Aliases

FLEXT-Core ships with single-letter runtime aliases for concise, consistent call sites:

```python
from flext_core import c, d, e, h, m, p, r, s, t, u, x
```

| Alias | Full Class | Purpose |
|---|---|---|
| `r` | `FlextResult` | Railway-Oriented Programming |
| `m` | `FlextModels` | DDD base models |
| `p` | `FlextProtocols` | Structural typing protocols |
| `t` | `FlextTypes` | Type aliases and generics |
| `u` | `FlextUtilities` | Utility functions |
| `c` | `FlextConstants` | Immutable constants |
| `d` | `FlextDecorators` | Cross-cutting decorators |
| `e` | `FlextExceptions` | Exception hierarchy |
| `h` | `FlextHandlers` | CQRS handler base class |
| `s` | `FlextService` | Domain service base class |
| `x` | `FlextMixins` | Composable infrastructure mixins |

All aliases are lazy-loaded via PEP 562 `__getattr__` — importing `flext_core` has zero import cost until an alias is first accessed.

---

## Examples

The [`examples/`](examples/) directory contains self-contained runnable scripts covering every major pattern:

| Example | Topic |
|---|---|
| [`ex_01_flext_result.py`](examples/ex_01_flext_result.py) | Railway-Oriented Programming |
| [`ex_02_flext_settings.py`](examples/ex_02_flext_settings.py) | Configuration management |
| [`ex_03_flext_logger.py`](examples/ex_03_flext_logger.py) | Structured logging |
| [`ex_04_flext_dispatcher.py`](examples/ex_04_flext_dispatcher.py) | CQRS dispatcher |
| [`ex_05_flext_mixins.py`](examples/ex_05_flext_mixins.py) | Composable mixins |
| [`ex_06_flext_context.py`](examples/ex_06_flext_context.py) | Context propagation |
| [`ex_07_flext_exceptions.py`](examples/ex_07_flext_exceptions.py) | Exception hierarchy |
| [`ex_08_flext_container.py`](examples/ex_08_flext_container.py) | Dependency injection |
| [`ex_09_flext_decorators.py`](examples/ex_09_flext_decorators.py) | Decorator toolkit |
| [`ex_10_flext_handlers.py`](examples/ex_10_flext_handlers.py) | Handler base classes |
| [`ex_11_flext_service.py`](examples/ex_11_flext_service.py) | Domain services |
| [`ex_12_flext_registry.py`](examples/ex_12_flext_registry.py) | Handler registry |

```bash
# Install in development mode
pip install -e .

# Run any example
python examples/ex_01_flext_result.py
```

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for the quick workflow, and [`docs/development/contributing.md`](docs/development/contributing.md) for full guidelines.

**Local quality gates:**

```bash
make lint        # Ruff + isort + type-check
make test        # Full test suite
make validate    # All gates (required before opening a PR)
```

**Commit conventions:** use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`).

1. Fork the repository
1. Create a focused branch (`git checkout -b feat/my-feature`)
1. Keep changes scoped to one concern per commit
1. Run `make validate` before pushing
1. Open a Pull Request with a clear problem statement and validation evidence

---

## License

Released under the [MIT License](LICENSE). Copyright © 2025 FLEXT Contributors.
