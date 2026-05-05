# Railway Programming Audit

<!-- TOC START -->
- [Current Status](#current-status)
- [Audit Check: map + flat_map](#audit-check-map-flatmap)
- [Audit Check: recover](#audit-check-recover)
<!-- TOC END -->

## Current Status

Core railway behaviors are verified through executable examples and docs snippets.

## Audit Check: map + flat_map

```python
from flext_core import p, r

even_divisor = 2
increment = 2
starting_value = 4
expected_value = 6


def ensure_even(value: int) -> p.Result[int]:
    if value % even_divisor:
        return r[int].fail("not_even")
    return r[int].ok(value)


result = r[int].ok(starting_value).map(lambda n: n + increment).flat_map(ensure_even)
assert result.success
assert result.value == expected_value
```

## Audit Check: recover

```python
from flext_core import r

fallback = r[int].fail("missing").recover(lambda _err: 1)
assert fallback.success
assert fallback.value == 1
```
