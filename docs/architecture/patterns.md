# Architecture Patterns

<!-- TOC START -->

- [Overview](#overview)
- [Result Composition Pattern](#result-composition-pattern)
- [Container Pattern](#container-pattern)
- [Dispatcher Pattern (examples-backed)](#dispatcher-pattern-examples-backed)

<!-- TOC END -->

## Overview

These patterns are executable references for core runtime behavior.

## Result Composition Pattern

```python
from __future__ import annotations

from flext_core import p, r


def validate_name(name: str) -> p.Result[str]:
    """Reject empty names with a failure result.

    Returns:
        The resulting ``p.Result[str]``.

    """
    if not name:
        return r[str].fail("name_required")
    return r[str].ok(name)


def to_slug(name: str) -> p.Result[str]:
    """Convert a name into its slug form.

    Returns:
        The resulting ``p.Result[str]``.

    """
    return r[str].ok(name.strip().lower().replace(" ", "-"))


slug = r[str].ok("Alice Doe").flat_map(validate_name).flat_map(to_slug)
expected_slug = "alice-doe"
if not slug.success:
    message = "Expected slug pipeline success"
    raise RuntimeError(message)
if slug.value != expected_slug:
    message = "Unexpected slug value"
    raise RuntimeError(message)
```

## Container Pattern

```python
from flext_core import FlextContainer

container = FlextContainer()
_ = container.bind("feature_flag", "enabled")

flag = container.resolve("feature_flag")
expected_flag = "enabled"
if not flag.success:
    message = "Expected bound value resolution success"
    raise RuntimeError(message)
if flag.value != expected_flag:
    message = "Unexpected resolved flag value"
    raise RuntimeError(message)
```

## Dispatcher Pattern (examples-backed)

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
