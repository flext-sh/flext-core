# Architecture Patterns

## Overview

These patterns are executable references for core runtime behavior.

## Result Composition Pattern

```python
from flext_core import p, r


def validate_name(name: str) -> p.Result[str]:
    if not name:
        return r[str].fail("name_required")
    return r[str].ok(name)


def to_slug(name: str) -> p.Result[str]:
    return r[str].ok(name.strip().lower().replace(" ", "-"))


slug = r[str].ok("Alice Doe").flat_map(validate_name).flat_map(to_slug)
assert slug.success
assert slug.value == "alice-doe"
```

## Container Pattern

```python
from flext_core import FlextContainer

container = FlextContainer()
_ = container.bind("feature_flag", "enabled")

flag = container.resolve("feature_flag")
assert flag.success
assert flag.value == "enabled"
```

## Dispatcher Pattern (examples-backed)

```python
from examples.ex_04_flext_dispatcher import _Ex04Exercise

_Ex04Exercise("docs/architecture/patterns.md").exercise()
```
