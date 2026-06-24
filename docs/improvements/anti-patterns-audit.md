# Anti-Patterns Audit (Current)

## Scope

This page contains executable checks only.

## Result pattern sanity

```python
from __future__ import annotations

from flext_core import p, r


def normalize(value: str) -> p.Result[str]:
    if not value:
        return r[str].fail("empty")
    return r[str].ok(value.strip())


assert normalize(" x ").success
assert normalize("").failure
```

## examples-backed sanity

```python
import io
from contextlib import redirect_stdout

from examples.ex_03_flext_logger import Ex03FlextLogger

stream = io.StringIO()
with redirect_stdout(stream):
    Ex03FlextLogger().run()
```
