# flext_core package README

## Purpose

`flext_core` provides base contracts for result flow, settings, container wiring,
and dispatcher-driven orchestration.

## Minimal Runtime Examples

### Result

```python
from flext_core import p, r


def safe_divide(a: float, b: float) -> p.Result[float]:
    if b == 0:
        return r[float].fail("division_by_zero")
    return r[float].ok(a / b)


assert safe_divide(10, 2).success
assert safe_divide(10, 0).failure
```

### Settings

```python
from flext_core import FlextSettings

settings = FlextSettings.fetch_global()
assert isinstance(settings.model_dump(), dict)
```

### Container

```python
from flext_core import FlextContainer

container = FlextContainer()
_ = container.bind("service", "ready")
resolved = container.resolve("service")

assert resolved.success
assert resolved.value == "ready"
```

### Dispatcher (examples-backed)

```python
from examples.ex_04_flext_dispatcher import _Ex04Exercise

_Ex04Exercise("src/flext_core/README.md").exercise()
```
