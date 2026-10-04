# Development Standards

<!-- TOC START -->

- [Core Expectations](#core-expectations)
- [Example: Result-first workflow](#example-result-first-workflow)
- [Example: Runtime wiring](#example-runtime-wiring)

<!-- TOC END -->

## Core Expectations

- Use current public APIs only.
- Prefer `p.Result[T]` for fallible flows.
- Keep snippets runnable and self-contained.

## Example: Result-first workflow

```python
from __future__ import annotations

from flext_core import p, r


def ensure_non_empty(value: str) -> p.Result[str]:
    """Reject empty values with a failure result.

    Returns:
        The resulting ``p.Result[str]``.

    """
    if not value:
        return r[str].fail("empty_value")
    return r[str].ok(value)


kept_value = ensure_non_empty("ok")
rejected_value = ensure_non_empty("")
if not kept_value.success:
    message = "Expected non-empty value success"
    raise RuntimeError(message)
if not rejected_value.failure:
    message = "Expected empty value failure"
    raise RuntimeError(message)
```

## Example: Runtime wiring

```python
from flext_core import FlextContainer, FlextSettings

container = FlextContainer()
settings = FlextSettings.fetch_global()
_ = container.bind("settings", settings)

resolved = container.resolve("settings")
if not resolved.success:
    message = "Expected bound settings resolution success"
    raise RuntimeError(message)
```
