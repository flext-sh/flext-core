# flext_core package README

## Purpose

`flext_core` provides base contracts for result flow, settings, container wiring, and
dispatcher-driven orchestration.

## Minimal Runtime Examples

### Result

```python
from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)

from flext_core import p, r
from flext_tests import tm


def safe_divide(a: float, b: float) -> p.Result[float]:
    if b == 0:
        return r[float].fail("division_by_zero")
    return r[float].ok(a / b)


tm.ok(safe_divide(10, 2))
tm.fail(safe_divide(10, 0))
```

### Settings

```python
from flext_core import FlextSettings
from flext_tests import tm

settings = FlextSettings.fetch_global()
tm.that(settings.model_dump(), is_=dict)
```

### Container

```python
from flext_core import FlextContainer
from flext_tests import tm

container = FlextContainer()
_ = container.bind("service", "ready")
resolved = container.resolve("service")

tm.ok(resolved, eq="ready")
```

### Dispatcher (examples-backed)

```python
from examples.ex_04_flext_dispatcher import Ex04DispatchDsl
from flext_tests import tm

result = Ex04DispatchDsl.run()
tm.ok(result, eq="pong:dispatcher-example")
```
