# Documentation Templates

<!-- TOC START -->

- [Template: API Example](#template-api-example)
- [Template: Runtime Example](#template-runtime-example)

<!-- TOC END -->

## Template: API Example

```python
from __future__ import annotations

from flext_core import p, r


def run_case(value: str) -> p.Result[str]:
    """Pass through non-empty values, rejecting missing ones.

    Returns:
        The resulting ``p.Result[str]``.

    """
    if not value:
        return r[str].fail("missing_value")
    return r[str].ok(value)


kept_case = run_case("ok")
missing_case = run_case("")
if not kept_case.success:
    message = "Expected provided value success"
    raise RuntimeError(message)
if not missing_case.failure:
    message = "Expected missing value failure"
    raise RuntimeError(message)
```

## Template: Runtime Example

```python
from flext_core import FlextContainer

container = FlextContainer()
_ = container.bind("template", "active")

result = container.resolve("template")
if not result.success:
    message = "Expected bound template resolution success"
    raise RuntimeError(message)
```
