# FLEXT-Core

FLEXT-Core is the dispatcher-centric foundation library for the FLEXT ecosystem. It provides railway-oriented programming primitives, an explicit dependency injection container, CQRS dispatching, and domain-driven design helpers for Python 3.13+.

## Key Capabilities

- **Railway-oriented results:** `FlextResult` models success and failure without raising exceptions in business code.
- **Dependency injection:** `FlextContainer` manages shared services with clear scopes (global or custom containers).
- **CQRS dispatcher:** `FlextDispatcher` routes commands, queries, and domain events through registered handlers and middleware.
- **Domain primitives:** `FlextModels`, `FlextService`, and mixins simplify entity/value modeling and domain service composition.
- **Infrastructure helpers:** configuration loading, structured logging, and execution context propagation are bundled for consistent cross-cutting concerns.

## Architecture at a Glance

```
src/flext_core/
├── config.py        # Environment-aware configuration helpers
├── constants.py     # Shared defaults and immutables
├── container.py     # Dependency injection container
├── context.py       # Request/operation context propagation
├── decorators.py    # Cross-cutting decorators and middleware helpers
├── dispatcher.py    # CQRS dispatcher for commands, queries, and events
├── exceptions.py    # Exception hierarchy aligned with result codes
├── handlers.py      # Handler interfaces and base implementations
├── loggings.py      # Structured logging helpers
├── mixins.py        # Reusable mixins for models and services
├── models.py        # DDD entities, values, and aggregates
├── registry.py      # Handler registry utilities
├── result.py        # Railway-oriented `FlextResult`
├── runtime.py       # Integration bridge to external libraries
├── service.py       # Domain service base classes
├── typings.py       # Shared typing aliases and protocols
└── utilities.py     # General-purpose helpers
```

## Quick Start

Install and verify the core imports:

```bash
pip install flext-core
python - <<'PY'
from flext_core import FlextContainer, FlextDispatcher, FlextResult
container = FlextContainer.get_global()
print('flext-core ready', FlextDispatcher.__name__, bool(container))
PY
```

Register and dispatch a simple command:

```python
from dataclasses import dataclass
from flext_core import FlextDispatcher, FlextResult

@dataclass
class CreateUser:
    email: str

dispatcher = FlextDispatcher()

def handle_create_user(message: CreateUser) -> FlextResult[str]:
    if "@" not in message.email:
        return FlextResult[str].fail("invalid email")
    return FlextResult[str].ok(f"created {message.email}")

dispatcher.register_handler(CreateUser, handle_create_user)
result = dispatcher.dispatch(CreateUser(email="user@example.com"))
assert result.is_success
```

## Documentation

Full documentation lives in [`docs/`](./docs/) and follows the standards in [`docs/standards/`](./docs/standards/). Notable entry points:

- [`docs/QUICK_START.md`](./docs/QUICK_START.md) — five-minute overview
- [`docs/architecture/overview.md`](./docs/architecture/overview.md) — layer summary
- [`docs/api-reference/`](./docs/api-reference/) — module-level API reference by layer
- [`examples/`](./examples/) — runnable examples demonstrating handlers, dispatcher flows, and supporting utilities

## Contributing and Support

- Review [`docs/development/contributing.md`](./docs/development/contributing.md) for coding, documentation, and testing expectations (PEP 8/257 compliant).
- Open issues or discussions on GitHub for bug reports or design questions.
