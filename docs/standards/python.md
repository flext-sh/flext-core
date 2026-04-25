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
from flext_core import p, r


def parse_int(raw: str) -> p.Result[int]:
    try:
        return r[int].ok(int(raw))
    except ValueError:
        return r[int].fail("invalid_int")


assert parse_int("42").success
assert parse_int("x").failure
```

## Settings Contract Example

```python
from flext_core import FlextSettings

settings = FlextSettings.fetch_global()
snapshot = settings.model_dump()

assert isinstance(snapshot, dict)
```

## Container Contract Example

```python
from flext_core import FlextContainer

container = FlextContainer()
_ = container.bind("name", "flext")

resolved = container.resolve("name")
assert resolved.success
assert resolved.value == "flext"
```
