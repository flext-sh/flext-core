# Architecture Decisions

<!-- TOC START -->

- [Active Decisions](#active-decisions)
- [Decision Check: Result Contract](#decision-check-result-contract)

<!-- TOC END -->

## Active Decisions

1. Use `r[T]` for explicit failure modeling.
2. Use Pydantic v2 models and `model_dump()`.
3. Use container + dispatcher as runtime composition primitives.

## Decision Check: Result Contract

```python
from __future__ import annotations

from flext_core import p, r


def to_int(raw: str) -> p.Result[int]:
    """Parse an integer, mapping ``ValueError`` to a failure result.

    Returns:
        The resulting ``p.Result[int]``.

    """
    try:
        return r[int].ok(int(raw))
    except ValueError:
        return r[int].fail("invalid_int")


parsed_ok = to_int("7")
parsed_ko = to_int("bad")
if not parsed_ok.success:
    message = "Expected valid integer success"
    raise RuntimeError(message)
if not parsed_ko.failure:
    message = "Expected invalid integer failure"
    raise RuntimeError(message)
```
