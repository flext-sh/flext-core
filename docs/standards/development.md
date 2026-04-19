# Development Standards

## Core Expectations

- Use current public APIs only.
- Prefer `p.Result[T]` for fallible flows.
- Keep snippets runnable and self-contained.

## Example: Result-first workflow

```python
from flext_core import p, r


def ensure_non_empty(value: str) -> p.Result[str]:
    if not value:
        return r[str].fail("empty_value")
    return r[str].ok(value)


assert ensure_non_empty("ok").success
assert ensure_non_empty("").failure
```

## Example: Runtime wiring

```python
from flext_core import FlextContainer, FlextSettings

container = FlextContainer()
settings = FlextSettings.fetch_global()
_ = container.bind("settings", settings)

resolved = container.resolve("settings")
assert resolved.success
```
