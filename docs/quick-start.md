# Quick Start


<!-- TOC START -->
- [Step 1: r[T] basics](#step-1-rt-basics)
- [Step 2: Container basics](#step-2-container-basics)
- [Step 3: Dispatcher example](#step-3-dispatcher-example)
<!-- TOC END -->

## Step 1: r[T] basics

```python
from flext_core import p, r


def ping(value: str) -> p.Result[str]:
    if not value:
        return r[str].fail("missing_value")
    return r[str].ok(f"pong:{value}")


assert ping("ok").success
assert ping("").failure
```

## Step 2: Container basics

```python
from flext_core import FlextContainer

container = FlextContainer()
_ = container.bind("app", "flext")
app = container.resolve("app")
assert app.success
assert app.value == "flext"
```

## Step 3: Dispatcher example

```python
from examples.ex_04_flext_dispatcher import Ex04DispatchDsl

result = Ex04DispatchDsl.run()
assert result.success
assert result.value == "dispatcher-example"
```
