# Error Handling Guide

## Overview

Use `r[T]` to keep errors explicit and composable.

## Basic Pattern

```python
from flext_core import p, r


def parse_port(raw: str) -> p.Result[int]:
    try:
        port = int(raw)
    except ValueError:
        return r[int].fail("invalid_port")
    if port < 1 or port > 65535:
        return r[int].fail("port_out_of_range")
    return r[int].ok(port)


assert parse_port("8080").success
assert parse_port("bad").failure
```

## Recovery Pattern

```python
from flext_core import r

result = r[int].fail("missing_value").recover(lambda _err: 80)
assert result.success
assert result.value == 80
```

## Error Mapping Pattern

```python
from flext_core import r

mapped = r[str].fail("not_found").map_error(lambda msg: f"domain_error:{msg}")
assert mapped.failure
assert mapped.error == "domain_error:not_found"
```
