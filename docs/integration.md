# Integration Guide

Guidance for integrating FLEXT Core into downstream projects while aligning with the 1.0.0 modernization pillars.

---

## Core Principles

1. **Dispatcher Unification** – adopt `FlextDispatcher` (or `FlextDispatcherRegistry`) as the single routing surface. Downstream packages should not implement bespoke match/case dispatchers.
2. **Context-First Observability** – propagate correlation/request metadata via `FlextContext`. Loggers, metrics, and tracing hooks rely on this shared context.
3. **Configuration Alignment** – boot applications with `FlextConfig` + `FlextContainer`, ensuring domain services consume the same configuration contract.

---

## Bootstrap Template

```python
from flext_core import (
    FlextConfig,
    FlextContainer,
    FlextContext,
    FlextDispatcher,
    FlextDispatcherRegistry,
    FlextLogger,
    FlextResult,
)

class TargetConfig(FlextConfig):
    endpoint: str
    batch_size: int = 1000

config = TargetConfig()
container = FlextContainer.get_global()
container.register(TargetConfig.__name__, config)
container.register("logger", FlextLogger("connector"))

dispatcher = FlextDispatcher()
registry = FlextDispatcherRegistry(dispatcher)

class LoadBatchHandler:
    def handle(self, command: dict) -> FlextResult[str]:
        FlextContext.Operation.set_operation_name("load_batch")
        # ... perform work using container.get(TargetConfig.__name__)
        return FlextResult[str].ok("ok")

registry.register_pairs([(dict, LoadBatchHandler())])
```

This pattern mirrors the pilot migrations for CLI and Oracle connectors described in the modernization plan.

---

## Migrating Legacy Dispatchers

1. Wrap legacy handler functions in `FlextDispatcher.register_function` to maintain behaviour while moving registrations into the shared surface.
2. Replace manual context propagation with `FlextContext` scopes:

```python
from flext_core import FlextContext

with FlextContext.Operation.scope("legacy-import", metadata={"source": "legacy"}):
    dispatcher.dispatch(command)
```

3. Surface migration progress using `FlextDispatcherRegistry.summary` outputs (registered/skipped/errors).

---

## Configuration Best Practices

- Expose a single `FlextConfig` subclass per package and document mandatory environment variables.
- Use the container to distribute configuration instances instead of sharing module-level globals.
- When overriding configuration in tests, use helper utilities provided in `tests/fixtures` to maintain consistent behaviour.

---

## Context & Logging

- Use `FlextLogger` everywhere (including CLI scripts) so correlation IDs from `FlextContext` automatically appear in logs.
- Ensure asynchronous or threaded code wraps work in `FlextContext` scopes to retain metadata.
- Observability teams can extend log processors without patching downstream packages because the shared context is enforced.

---

## Validation & Compatibility

- Run `pytest tests/patterns/test_patterns_commands.py` in downstream projects to ensure dispatcher contracts are respected.
- Monitor `tests/integration/test_wildcard_exports_clean.py` for changes to the public API surface – integration code should import only through `flext_core` top-level exports.

Keep this guide aligned with the modernization plan as additional integration examples or tooling become available.
