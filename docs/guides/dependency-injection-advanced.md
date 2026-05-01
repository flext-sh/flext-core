# Dependency Injection Advanced

<!-- TOC START -->
- [Overview](#overview)
- [Reusing Official Example Code](#reusing-official-example-code)
- [Core Container Operations](#core-container-operations)
- [Scoped Containers](#scoped-containers)
- [Batch Registration Pattern](#batch-registration-pattern)
- [Best Practices](#best-practices)
<!-- TOC END -->

## Overview

This guide focuses on real `FlextContainer` usage with the current API.
Examples are backed by executable code from the `examples/` package.

## Reusing Official Example Code

Use the canonical container example as the reference path.

```python
from examples.ex_08_flext_container import Ex08FlextContainer

demo = Ex08FlextContainer("docs/guides/dependency-injection-advanced.md")
container = demo._exercise_singleton_and_creation()

assert container is not None
```

The `Ex08FlextContainer` flow exercises binding, factories, resolution, and scoped containers.

## Core Container Operations

```python
from flext_core import FlextContainer, u

container = FlextContainer()

_ = container.bind("settings_name", "flext-core")
_ = container.factory("logger", lambda: u.fetch_logger(__name__))

settings_name = container.resolve("settings_name")
logger = container.resolve("logger")

assert settings_name.success
assert logger.success
```

## Scoped Containers

```python
from flext_core import FlextContainer

root = FlextContainer()
_ = root.bind("tenant", "default")

scoped = root.scope(subproject="tenant_a")
tenant = scoped.resolve("tenant")

assert tenant.success
assert tenant.value == "default"
```

## Batch Registration Pattern

`FlextContainer` does not expose `batch_register`; use explicit loop registration for deterministic failure points.

```python
from collections.abc import Callable, Mapping, MutableMapping, MutableSequence, Sequence

from flext_core import FlextContainer, p, r, t


def bind_services(
    container: FlextContainer,
    services: t.SequenceOf[tuple[str, t.RegisterableService]],
) -> p.Result[bool]:
    for name, service in services:
        _ = container.bind(name, service)
        resolved = container.resolve(name)
        if resolved.failure:
            return r[bool].fail(f"failed_to_bind:{name}")
    return r[bool].ok(True)


container = FlextContainer()
result = bind_services(
    container,
    (
        ("service_a", "ok"),
        ("service_b", "ok"),
    ),
)

assert result.success
```

## Best Practices

- Keep service names stable and explicit.
- Prefer `bind` for concrete instances and `factory` for deferred construction.
- Validate each critical resolution step with `result.success`.
- Use `scope(...)` for isolation when composing runtime contexts.
