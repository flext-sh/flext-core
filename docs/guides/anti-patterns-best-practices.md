# Anti-Patterns and Best Practices

<!-- TOC START -->

- [Overview](#overview)
- [Common Anti-Patterns (Illustrative)](#common-anti-patterns-illustrative)
- [Best Practices (Executable)](#best-practices-executable)
  - [Prefer r\[T\] for Fallible Paths](#prefer-rt-for-fallible-paths)
  - [Prefer Current Settings API](#prefer-current-settings-api)
  - [Prefer Explicit Container Registration](#prefer-explicit-container-registration)
  - [Reuse Maintainer Examples](#reuse-maintainer-examples)

<!-- TOC END -->

## Overview

This guide separates intentionally wrong examples (text only) from executable
best-practice snippets.

## Common Anti-Patterns (Illustrative)

```text
Wrong pattern: relying on exception flow for business failures.

def process(data):
    if "email" not in data:
        raise ValueError("missing email")
    return data
```

```text
Wrong pattern: legacy API names in docs, e.g. FlextSettings.fetch_global().
Use FlextSettings.fetch_global() instead.
```

```text
Wrong pattern: assuming non-existent container.batch_register().
Use explicit bind/factory loops.
```

## Best Practices (Executable)

### Prefer r[T] for Fallible Paths

```python
from __future__ import annotations

from flext_core import p, r


def validate_payload(payload: dict[str, str]) -> p.Result[dict[str, str]]:
    """Require the email key before accepting a payload.

    Returns:
        The resulting ``p.Result[dict[str, str]]``.

    """
    if "email" not in payload:
        return r[dict[str, str]].fail("missing_email")
    return r[dict[str, str]].ok(payload)


valid_payload = validate_payload({"email": "a@b.com"})
empty_payload = validate_payload({})
if not valid_payload.success:
    message = "Expected complete payload success"
    raise RuntimeError(message)
if not empty_payload.failure:
    message = "Expected incomplete payload failure"
    raise RuntimeError(message)
```

### Prefer Current Settings API

```python
from flext_core import FlextSettings

settings = FlextSettings.fetch_global()
data = settings.model_dump()

if not isinstance(data, dict):
    message = "Expected dict settings snapshot"
    raise TypeError(message)
```

### Prefer Explicit Container Registration

```python
from flext_core import FlextContainer

container = FlextContainer()
_ = container.bind("service", "ready")

service = container.resolve("service")
expected_service = "ready"
if not service.success:
    message = "Expected bound service resolution success"
    raise RuntimeError(message)
if service.value != expected_service:
    message = "Unexpected resolved service value"
    raise RuntimeError(message)
```

### Reuse Maintainer Examples

```python
import io
from contextlib import redirect_stdout

from examples.ex_03_flext_logger import Ex03FlextLogger
from examples.ex_04_flext_dispatcher import Ex04DispatchDsl

stream = io.StringIO()
with redirect_stdout(stream):
    Ex03FlextLogger().run()
result = Ex04DispatchDsl.run()
expected_value = "pong:dispatcher-example"
if not result.success:
    message = "Expected dispatcher success"
    raise RuntimeError(message)
if result.value != expected_value:
    message = "Unexpected dispatcher value"
    raise RuntimeError(message)
```
