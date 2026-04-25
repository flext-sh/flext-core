# Documentation Templates

<!-- TOC START -->
- [Template: API Example](#template-api-example)
- [Template: Runtime Example](#template-runtime-example)
<!-- TOC END -->

## Template: API Example

```python
from flext_core import p, r


def run_case(value: str) -> p.Result[str]:
    if not value:
        return r[str].fail("missing_value")
    return r[str].ok(value)


assert run_case("ok").success
assert run_case("").failure
```

## Template: Runtime Example

```python
from flext_core import FlextContainer

container = FlextContainer()
_ = container.bind("template", "active")

result = container.resolve("template")
assert result.success
```
