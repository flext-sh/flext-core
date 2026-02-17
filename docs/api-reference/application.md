# Application Layer API Reference

The application layer coordinates domain logic through CQRS-style handlers, reliability policies, and structured observability.

Canonical architecture context:

- `README.md`
- `docs/architecture/overview.md`
- `docs/architecture/cqrs.md`

Primary components in this layer: `FlextDispatcher`, `h`, `FlextRegistry`, and `FlextDecorators`.

## FlextDispatcher - Unified CQRS Dispatcher

`FlextDispatcher` is the command/query dispatcher that orchestrates handler execution while applying reliability features such as retries, rate limiting, timeouts, and optional caching.

```python
from flext_core import FlextDispatcher

# Define messages
class CreateUserCommand:
    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email

class GetUserQuery:
    def __init__(self, user_id: str):
        self.user_id = user_id

# Create dispatcher and register handlers
dispatcher = FlextDispatcher()
dispatcher.register_handler(CreateUserCommand, handle_create_user)
dispatcher.register_handler(GetUserQuery, handle_get_user)

# Dispatch with built-in reliability (retries, timeouts, rate limiting)
create_result = dispatcher.dispatch(CreateUserCommand("Alice", "alice@example.com"))
user_result = dispatcher.dispatch(GetUserQuery("user-123"))
```

**Key capabilities**

- Command/query dispatch with duck-typed protocol compliance
- Reliability controls: retries, timeouts, rate limiting, and optional LRU caching
- Context propagation for correlation IDs and structured logging
- Batch dispatch helpers for processing collections of messages

## h - CQRS Handler Base

`h` provides the abstract base class for implementing command and query handlers. It supplies validation hooks, context propagation, and `FlextResult`-based error handling.

```python
from flext_core.handlers import h
from flext_core.result import r

class CreateUserHandler(h[CreateUserCommand, bool]):
    def handle(self, message: CreateUserCommand) -> r[bool]:
        if "@" not in message.email:
            return r[bool].fail("Invalid email")
        # Business logic here
        return r[bool].ok(True)
```

**Highlights**

- Validation pipeline via `validate()` before executing `handle()`
- Context-aware execution for tracing/logging
- Generic typing for precise return values

## FlextRegistry - Handler Registration Utilities

`FlextRegistry` centralizes handler registration and discovery, providing idempotent batch operations and summary reporting.

```python
from flext_core import FlextRegistry

registry = FlextRegistry()
registry.register_handler(CreateUserCommand, create_user_handler)
registry.register_handler(GetUserQuery, get_user_handler)

summary = registry.register_handlers([
    (CreateUserCommand, create_user_handler),
    (GetUserQuery, get_user_handler),
])
```

**Highlights**

- Batch or single registration with duplicate detection
- Returns `FlextResult` objects describing successes and skips
- Designed to pair with `FlextDispatcher` for execution

## FlextDecorators - Cross-Cutting Concerns

`FlextDecorators` packages common cross-cutting behaviors so handlers stay focused on business logic.

```python
from flext_core.decorators import FlextDecorators
from flext_core.result import r

@FlextDecorators.retry(attempts=3)
@FlextDecorators.timeout(seconds=2)
@FlextDecorators.inject
def handle_create_user(cmd: CreateUserCommand, logger) -> r[bool]:
    logger.info("creating user", user=cmd.name)
    return r[bool].ok(True)
```

**Highlights**

- Reliability decorators: `@retry`, `@timeout`, `@with_correlation`
- Dependency injection: `@inject` to resolve services from `FlextContainer`
- Railway pattern helper: `@railway` to wrap callables with `FlextResult`

## Quick Start Checklist

1. Define command/query messages and corresponding handlers inheriting `h`.
2. Register handlers with `FlextDispatcher` or via `FlextRegistry` batch helpers.
3. Apply `FlextDecorators` for retries, timeouts, context propagation, or DI.
4. Dispatch messages through `FlextDispatcher.dispatch(...)` and work with `FlextResult` outputs.

## Verification Commands

Run from `flext-core/`:

```bash
make lint
make type-check
make test-fast
```
