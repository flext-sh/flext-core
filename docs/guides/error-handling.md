# Error Handling Guide

<!-- TOC START -->

- [Overview](#overview)
- [Basic Pattern](#basic-pattern)
- [Recovery Pattern](#recovery-pattern)
- [Error Mapping Pattern](#error-mapping-pattern)

<!-- TOC END -->

## Overview

Use `r[T]` to keep errors explicit and composable.

## Basic Pattern

```python
from __future__ import annotations

from flext_core import c, p, r


def parse_port(raw: str) -> p.Result[int]:
    """Parse a port within library bounds, mapping errors to failures.

    Returns:
        The resulting ``p.Result[int]``.

    """
    try:
        port = int(raw)
    except ValueError:
        return r[int].fail("invalid_port")
    if port < c.MIN_PORT or port > c.MAX_PORT:
        return r[int].fail("port_out_of_range")
    return r[int].ok(port)


parsed_port = parse_port("8080")
invalid_port = parse_port("bad")
if not parsed_port.success:
    message = "Expected in-range port success"
    raise RuntimeError(message)
if not invalid_port.failure:
    message = "Expected non-numeric port failure"
    raise RuntimeError(message)
```

## Recovery Pattern

```python
from flext_core import r

default_port = 80
result = r[int].fail("missing_value").recover(lambda _err: default_port)
if not result.success:
    message = "Expected recovery success"
    raise RuntimeError(message)
if result.value != default_port:
    message = "Unexpected recovery value"
    raise RuntimeError(message)
```

## Error Mapping Pattern

```python
from flext_core import r

mapped = r[str].fail("not_found").map_error(lambda msg: f"domain_error:{msg}")
expected_error = "domain_error:not_found"
if not mapped.failure:
    message = "Expected mapped failure"
    raise RuntimeError(message)
if mapped.error != expected_error:
    message = "Unexpected mapped error message"
    raise RuntimeError(message)
```
