# Domain-Driven Design Guide

<!-- TOC START -->
- [Overview](#overview)
- [Value Validation with r[T]](#value-validation-with-rt)
- [Entity Command Flow](#entity-command-flow)
- [Use Maintainer DDD-Like Examples](#use-maintainer-ddd-like-examples)
- [DDD Checklist](#ddd-checklist)
<!-- TOC END -->

## Overview

This DDD guide focuses on practical boundaries and executable flows.

## Value Validation with r[T]

```python
from flext_core import p, r


def validate_sku(sku: str) -> p.Result[str]:
    minimum_sku_length = 3
    if not sku or len(sku) < minimum_sku_length:
        return r[str].fail("invalid_sku")
    return r[str].ok(sku)


assert validate_sku("ABC").success
assert validate_sku("A").failure
```

## Entity Command Flow

```python
from flext_core import p, r


def validate_sku(sku: str) -> p.Result[str]:
    minimum_sku_length = 3
    if not sku or len(sku) < minimum_sku_length:
        return r[str].fail("invalid_sku")
    return r[str].ok(sku)


def create_product(command: dict[str, str]) -> p.Result[dict[str, str]]:
    sku_result = validate_sku(command.get("sku", ""))
    if sku_result.failure:
        return r[dict[str, str]].fail("product_validation_failed")
    return r[dict[str, str]].ok({"sku": sku_result.value, "status": "created"})


created = create_product({"sku": "SKU-123"})
assert created.success
```

## Use Maintainer DDD-Like Examples

```python
from examples.ex_11_flext_service import ExampleService
from examples.ex_12_flext_registry import Ex12RegistryDsl

ExampleService.run()
Ex12RegistryDsl("docs/guides/domain-driven-design.md").exercise()
```

## DDD Checklist

- Keep domain validation deterministic.
- Model failures explicitly with `r[T]`.
- Keep orchestration in services/handlers, not in entities.
de