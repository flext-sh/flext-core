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


def ensure_even(value: int) -> p.Result[int]:
    if value % 2:
        return r[int].fail("not_even")
    return r[int].ok(value)


result = r[int].ok(4).map(lambda n: n + 2).flat_map(ensure_even)
assert result.success
assert result.value == 6
```

## Audit Check: recover

```python
from flext_core import r

fallback = r[int].fail("missing").recover(lambda _err: 1)
assert fallback.success
assert fallback.value == 1
```
