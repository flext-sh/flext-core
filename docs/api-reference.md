# API Reference

Guaranteed public surface for the FLEXT Core 1.x line. Updated to highlight dispatcher, context, and configuration alignment required by the modernization plan.

---

## Top-Level Imports

```python
from flext_core import (
    FlextBus,
    FlextConfig,
    FlextContainer,
    FlextContext,
    FlextCqrs,
    FlextDispatcher,
    FlextDispatcherRegistry,
    FlextDomainService,
    FlextExceptions,
    FlextHandlers,
    FlextLogger,
    FlextMixins,
    FlextModels,
    FlextProcessing,
    FlextProtocols,
    FlextResult,
    FlextTypes,
    FlextUtilities,
    FlextVersionManager,
    __version__,
)
```

All names above are covered by tests that enforce export stability. Removing or renaming them requires a major-version bump.

---

## Key Types

### `FlextResult[T]`

```python
from flext_core import FlextResult

result = (
    FlextResult[int]
    .ok(1)
    .map(lambda value: value + 1)
    .flat_map(lambda value: FlextResult[int].ok(value * 2))
)

print(result.value)  # 4 (legacy `.data` also supported)
```

### `FlextContainer`

```python
from flext_core import FlextContainer

container = FlextContainer.get_global()
container.register("logger", object())
logger = container.get("logger").unwrap()
```

### `FlextDispatcher`

```python
from flext_core import FlextDispatcher, FlextResult

dispatcher = FlextDispatcher()

class EchoHandler:
    def handle(self, message: str) -> FlextResult[str]:
        return FlextResult[str].ok(message.upper())

dispatcher.register_command(str, EchoHandler())
assert dispatcher.dispatch("ping").unwrap() == "PING"
```

### `FlextContext`

```python
from flext_core import FlextContext

with FlextContext.Operation.scope("import-customers"):
    FlextContext.Request.set_user_id("cli")
    # Dispatch or log; handlers and loggers extract the metadata automatically
```

### `FlextDomainService`

```python
from flext_core import FlextDomainService, FlextResult

class ActivateAccount(FlextDomainService[FlextResult[None]]):
    def execute(self, account_id: str) -> FlextResult[None]:
        if not account_id:
            return FlextResult[None].fail("missing id")
        self.logger.info("activated", account_id=account_id)
        return FlextResult[None].ok(None)
```

---

## Dispatcher Registry Helpers

```python
from flext_core import FlextDispatcher, FlextDispatcherRegistry, FlextHandlers, FlextResult

dispatcher = FlextDispatcher()
registry = FlextDispatcherRegistry(dispatcher)

class ExampleHandler(FlextHandlers[str, str]):
    def handle(self, message: str) -> FlextResult[str]:
        return FlextResult[str].ok(message)

summary = registry.register_pairs([(str, ExampleHandler())])
assert summary.is_success
assert dispatcher.dispatch("modernization").unwrap() == "modernization"
```

> **Note:** `FlextHandlers` trusts incoming Pydantic models and skips redundant
> revalidation by default. Pass `revalidate_pydantic_messages=True` when
> instantiating a handler if you need the framework to perform an extra
> validation round.

`FlextDispatcherRegistry.Summary` exposes `registered`, `skipped`, and `errors` lists so CLI and connector packages can produce migration reports.

---

## Logging and Context Utilities

```python
from flext_core import FlextLogger

logger = FlextLogger(__name__)
with logger.context(operation="sync", correlation_id="abc123"):
    logger.info("dispatcher_run", handlers=1)
```

Log records automatically include context metadata from `FlextContext`.

---

## Version Helpers

```python
from flext_core import FlextVersionManager, __version__

print(__version__)
print(FlextVersionManager.VERSION_MAJOR)
```

The modernization plan keeps the version helpers intact for the entire 1.x lifecycle.

---

## Extending the API

- Prefer wrapping existing primitives (`FlextDispatcher`, `FlextContext`) instead of introducing new surfaces.
- When new exports are required, add them to `src/flext_core/__init__.py`, update this reference, and add coverage via integration tests.
- Document the change in the modernization plan dashboard.
