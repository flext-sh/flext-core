# Python Standards

<!-- TOC START -->

- [Core Rules](#core-rules)
- [Result Contract Example](#result-contract-example)
- [Settings Contract Example](#settings-contract-example)
- [Container Contract Example](#container-contract-example)

<!-- TOC END -->

## Core Rules

- Prefer explicit `p.Result[T]` for fallible business paths.
- Prefer Pydantic v2 models and `model_dump()`.
- Prefer container/service patterns that match current public APIs.

## Result Contract Example

```python
from __future__ import annotations

from flext_core import p, r


def parse_int(raw: str) -> p.Result[int]:
    """Parse an integer, mapping ``ValueError`` to a failure result.

    Returns:
        The resulting ``p.Result[int]``.

    """
    try:
        return r[int].ok(int(raw))
    except ValueError:
        return r[int].fail("invalid_int")


parsed_number = parse_int("42")
parsed_text = parse_int("x")
if not parsed_number.success:
    message = "Expected numeric parse success"
    raise RuntimeError(message)
if not parsed_text.failure:
    message = "Expected non-numeric parse failure"
    raise RuntimeError(message)
```

## Settings Contract Example

```python
from flext_core import FlextSettings

settings = FlextSettings.fetch_global()
snapshot = settings.model_dump()

if not isinstance(snapshot, dict):
    message = "Expected dict settings snapshot"
    raise TypeError(message)
```

## Container Contract Example

```python
from flext_core import FlextContainer

container = FlextContainer()
_ = container.bind("name", "flext")

resolved = container.resolve("name")
expected_name = "flext"
if not resolved.success:
    message = "Expected bound name resolution success"
    raise RuntimeError(message)
if resolved.value != expected_name:
    message = "Unexpected resolved name value"
    raise RuntimeError(message)
```
