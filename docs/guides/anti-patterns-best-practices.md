# Anti-Patterns and Best Practices

<!-- TOC START -->
- [Overview](#overview)
- [Common Anti-Patterns (Illustrative)](#common-anti-patterns-illustrative)
- [Best Practices (Executable)](#best-practices-executable)
  - [Prefer r[T] for Fallible Paths](#prefer-rt-for-fallible-paths)
  - [Prefer Current Settings API](#prefer-current-settings-api)
  - [Prefer Explicit Container Registration](#prefer-explicit-container-registration)
  - [Reuse Maintainer Examples](#reuse-maintainer-examples)
<!-- TOC END -->

## Overview

This guide separates intentionally wrong examples (text only) from executable best-practice snippets.

## Common Anti-Patterns (Illustrative)

```text
Wrong pattern: relying on exception flow for business failures.

def process(data):
    if "email" not in data:
        raise ValueError("missing email")
    return data
```

```text
Wrong pattern: legacy API names in docs, e.g. FlextSettings.get_global().
Use FlextSettings.fetch_global() instead.
```

```text
Wrong pattern: assuming non-existent container.batch_register().
Use explicit bind/factory loops.
```

## Best Practices (Executable)

### Prefer r[T] for Fallible Paths

```python
from flext_core import p, r


def validate_payload(payload: dict[str, str]) -> p.Result[dict[str, str]]:
    if "email" not in payload:
        return r[dict[str, str]].fail("missing_email")
    return r[dict[str, str]].ok(payload)


assert validate_payload({"email": "a@b.com"}).success
assert validate_payload({}).failure
```

### Prefer Current Settings API

```python
from flext_core import FlextSettings

settings = FlextSettings.fetch_global()
data = settings.model_dump()

assert isinstance(data, dict)
```

### Prefer Explicit Container Registration

```python
from flext_core import FlextContainer

container = FlextContainer()
_ = container.bind("service", "ready")

service = container.resolve("service")
assert service.success
assert service.value == "ready"
```

### Reuse Maintainer Examples

```python
from examples.ex_03_flext_logger import Ex03LoggingDsl
from examples.ex_04_flext_dispatcher import Ex04DispatchDsl

Ex03LoggingDsl("docs/guides/anti-patterns-best-practices.md").exercise()
result = Ex04DispatchDsl.run()
assert result.success
assert result.value == "dispatcher-example"
```
