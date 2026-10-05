# Anti-Patterns Audit (Current)

<!-- TOC START -->

- [Scope](#scope)
- [Result pattern sanity](#result-pattern-sanity)
- [examples-backed sanity](#examples-backed-sanity)

<!-- TOC END -->

## Scope

This page contains executable checks only.

## Result pattern sanity

```python
from __future__ import annotations

from flext_core import p, r


def normalize(value: str) -> p.Result[str]:
    """Strip whitespace, rejecting empty input.

    Returns:
        The resulting ``p.Result[str]``.

    """
    if not value:
        return r[str].fail("empty")
    return r[str].ok(value.strip())


stripped_value = normalize(" x ")
empty_value = normalize("")
if not stripped_value.success:
    message = "Expected whitespace-only pad success"
    raise RuntimeError(message)
if not empty_value.failure:
    message = "Expected empty input failure"
    raise RuntimeError(message)
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
