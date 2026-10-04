# Domain-Driven Design Guide

<!-- TOC START -->

- [Overview](#overview)
- [Value Validation with r\[T\]](#value-validation-with-rt)
- [Entity Command Flow](#entity-command-flow)
- [Use Maintainer DDD-Like Examples](#use-maintainer-ddd-like-examples)
- [DDD Checklist](#ddd-checklist)

<!-- TOC END -->

## Overview

This DDD guide focuses on practical boundaries and executable flows.

## Value Validation with r[T]

```python
from __future__ import annotations

from flext_core import p, r


def validate_sku(sku: str) -> p.Result[str]:
    """Enforce the minimum SKU length for the domain.

    Returns:
        The resulting ``p.Result[str]``.

    """
    minimum_sku_length = 3
    if not sku or len(sku) < minimum_sku_length:
        return r[str].fail("invalid_sku")
    return r[str].ok(sku)


valid_sku = validate_sku("ABC")
short_sku = validate_sku("A")
if not valid_sku.success:
    message = "Expected valid SKU success"
    raise RuntimeError(message)
if not short_sku.failure:
    message = "Expected short SKU failure"
    raise RuntimeError(message)
```

## Entity Command Flow

```python
from __future__ import annotations

from flext_core import p, r


def validate_sku(sku: str) -> p.Result[str]:
    """Enforce the minimum SKU length for the domain.

    Returns:
        The resulting ``p.Result[str]``.

    """
    minimum_sku_length = 3
    if not sku or len(sku) < minimum_sku_length:
        return r[str].fail("invalid_sku")
    return r[str].ok(sku)


def create_product(command: dict[str, str]) -> p.Result[dict[str, str]]:
    """Validate the SKU before recording product creation.

    Returns:
        The resulting ``p.Result[dict[str, str]]``.

    """
    sku_result = validate_sku(command.get("sku", ""))
    if sku_result.failure:
        return r[dict[str, str]].fail("product_validation_failed")
    return r[dict[str, str]].ok({"sku": sku_result.value, "status": "created"})


created = create_product({"sku": "SKU-123"})
if not created.success:
    message = "Expected valid product creation success"
    raise RuntimeError(message)
```

## Use Maintainer DDD-Like Examples

```python
import io
from contextlib import redirect_stdout

from examples.ex_11_flext_service import ExampleService
from examples.ex_12_flext_registry import Ex12RegistryDsl

ExampleService.run()
stream = io.StringIO()
with redirect_stdout(stream):
    Ex12RegistryDsl("docs/guides/domain-driven-design.md").exercise()
```

## DDD Checklist

- Keep domain validation deterministic.
- Model failures explicitly with `r[T]`.
- Keep orchestration in services/handlers, not in entities. de
