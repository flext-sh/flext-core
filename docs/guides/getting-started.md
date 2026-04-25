# Getting Started


<!-- TOC START -->
- [Overview](#overview)
- [Result Basics](#result-basics)
- [Settings Basics](#settings-basics)
- [Container Basics](#container-basics)
- [Dispatcher Walkthrough (Examples)](#dispatcher-walkthrough-examples)
- [Next Steps](#next-steps)
<!-- TOC END -->

## Overview

This quick start uses current `flext-core` APIs and runnable snippets.

## Result Basics

```python
from flext_core import p, r


def divide(a: float, b: float) -> p.Result[float]:
    if b == 0:
        return r[float].fail("division_by_zero")
    return r[float].ok(a / b)


assert divide(10, 2).success
assert divide(10, 0).failure
```

## Settings Basics

```python
from flext_core import FlextSettings

settings = FlextSettings.fetch_global()
snapshot = settings.model_dump()

assert "log_level" in snapshot
```

## Container Basics

```python
from flext_core import FlextContainer

container = FlextContainer()
_ = container.bind("project", "flext-core")

resolved = container.resolve("project")
assert resolved.success
assert resolved.value == "flext-core"
```

## Dispatcher Walkthrough (Examples)

```python
from examples.ex_04_flext_dispatcher import Ex04DispatchDsl

result = Ex04DispatchDsl.run()
assert result.success
assert result.value == "dispatcher-example"
```

## Next Steps

- Run additional examples in `examples/ex_*.py`.
- Use result pipelines (`map`, `flat_map`, `recover`) for business flows.
- Use `FlextContainer.scope(...)` for isolated runtime contexts.
