# Quick Start

<!-- TOC START -->

- [Step 1: p.Result\[T\] basics](#step-1-presultt-basics)
- [Step 2: Container basics](#step-2-container-basics)
- [Step 3: Dispatcher example](#step-3-dispatcher-example)

<!-- TOC END -->

## Step 1: p.Result[T] basics

```python
from __future__ import annotations

from flext_core import p, r


def ping(value: str) -> p.Result[str]:
    """Answer a ping, rejecting empty values.

    Returns:
        The resulting ``p.Result[str]``.

    """
    if not value:
        return r[str].fail("missing_value")
    return r[str].ok(f"pong:{value}")


ok_ping = ping("ok")
missing_ping = ping("")
if not ok_ping.success:
    message = "Expected ping success"
    raise RuntimeError(message)
if not missing_ping.failure:
    message = "Expected empty ping failure"
    raise RuntimeError(message)
```

## Step 2: Container basics

```python
from flext_core import FlextContainer

container = FlextContainer()
_ = container.bind("app", "flext")
app = container.resolve("app")
expected_app = "flext"
if not app.success:
    message = "Expected bound app resolution success"
    raise RuntimeError(message)
if app.value != expected_app:
    message = "Unexpected resolved app value"
    raise RuntimeError(message)
```

## Step 3: Dispatcher example

```python
from examples.ex_04_flext_dispatcher import Ex04DispatchDsl

result = Ex04DispatchDsl.run()
expected_value = "pong:dispatcher-example"
if not result.success:
    message = "Expected dispatcher success"
    raise RuntimeError(message)
if result.value != expected_value:
    message = "Unexpected dispatcher value"
    raise RuntimeError(message)
```
