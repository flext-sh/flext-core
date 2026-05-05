# Testing Guide

<!-- TOC START -->
- [Overview](#overview)
- [Assert r[T] Outcomes](#assert-rt-outcomes)
- [Test Container Registration](#test-container-registration)
- [Reuse Official Example Tests](#reuse-official-example-tests)
- [Recommended Strategy](#recommended-strategy)
<!-- TOC END -->

## Overview

This guide shows practical test patterns aligned with current FLEXT APIs.

## Assert r[T] Outcomes

```python
from flext_core import p, r


def validate_username(username: str) -> p.Result[str]:
    if not username:
        return r[str].fail("username_required")
    return r[str].ok(username)


ok = validate_username("alice")
ko = validate_username("")

assert ok.success
assert ko.failure
```

## Test Container Registration

```python
from flext_core import FlextContainer

container = FlextContainer()
_ = container.bind("service", "ready")

resolved = container.resolve("service")
assert resolved.success
assert resolved.value == "ready"
```

## Reuse Official Example Tests

```python
from examples.ex_11_flext_service import ExampleService

ExampleService.run()
```

## Recommended Strategy

- Prefer public behavior assertions (`success`, `failure`, `value`, `error`).
- Keep tests deterministic and isolated.
- Reuse `examples/` as contract references when APIs evolve.
