# Railway-Oriented Programming with r[T]

## Overview

`r[T]` is FLEXT's result container for explicit success/failure flows.

- Success path: `r[T].ok(value)`
- Failure path: `r[T].fail(message, error_code=..., error_data=...)`

The goal is predictable composition without exception-driven control flow.

All snippets below are standalone and executable.

## Creating Results

```python
from flext_core import p, r

success: p.Result[str] = r[str].ok("hello")
failure: p.Result[str] = r[str].fail("cannot_continue", error_code="DOCS_EXAMPLE")

assert success.success
assert failure.failure
assert failure.error_code == "DOCS_EXAMPLE"
```

## Reading State Safely

```python
from flext_core import r

result = r[int].ok(42)
if result.success:
    assert result.value == 42

failed = r[int].fail("broken")
assert failed.failure
assert failed.error == "broken"
```

## flat_map Composition

Use `flat_map` when each step already returns `p.Result[...]`.

```python
from flext_core import p, r


def validate_email(email: str) -> p.Result[str]:
    if "@" not in email:
        return r[str].fail("invalid_email")
    return r[str].ok(email)


def normalize_email(email: str) -> p.Result[str]:
    return r[str].ok(email.strip().lower())


result = (
    r[str].ok(" USER@EXAMPLE.COM ").flat_map(validate_email).flat_map(normalize_email)
)
assert result.success
assert result.value == "user@example.com"
```

## map, filter, map_error, recover

```python
from flext_core import r

length_result = r[str].ok("flext").map(len)
assert length_result.value == 5

adult_result = r[int].ok(21).filter(lambda age: age >= 18)
assert adult_result.success

error_result = r[str].fail("bad_input").map_error(lambda err: f"validation:{err}")
assert error_result.error == "validation:bad_input"

recovered = r[str].fail("not_found").recover(lambda _err: "guest")
assert recovered.success
assert recovered.value == "guest"
```

## unwrap_or and unwrap_or_else

```python
from flext_core import r

ok_value = r[int].ok(7).unwrap_or(0)
fallback_value = r[int].fail("nope").unwrap_or(0)

assert ok_value == 7
assert fallback_value == 0

lazy_fallback = r[int].fail("still_nope").unwrap_or_else(lambda: 99)
assert lazy_fallback == 99
```

## lash for Failure Branching

Use `lash` to transform failures into alternate result flows.

```python
from flext_core import p, r


def fallback(_message: str) -> p.Result[int]:
    return r[int].ok(0)


result = r[int].fail("cannot_parse").lash(fallback)
assert result.success
assert result.value == 0
```

## traverse and with_resource

```python
from flext_core import p, r


def validate_item(item: str) -> p.Result[str]:
    if len(item) < 3:
        return r[str].fail(f"too_short:{item}")
    return r[str].ok(item)


failed = r.traverse(["abc", "ok", "xy"], validate_item)
assert failed.failure

passed = r.traverse(["abc", "ok1", "xyz"], validate_item)
assert passed.success
assert passed.value == ["abc", "ok1", "xyz"]
```

```python
from flext_core import p, r


def create_connection() -> dict[str, int]:
    return {"id": 10}


def use_connection(conn: dict[str, int]) -> p.Result[int]:
    return r[int].ok(conn["id"])


def close_connection(_conn: dict[str, int]) -> None:
    return None


resource_result = r.with_resource(create_connection, use_connection, close_connection)
assert resource_result.success
assert resource_result.value == 10
```

## Decorator Integration

### @d.railway

```python
from flext_core import d


@d.railway(error_code="PARSE_ERROR")
def parse_positive_number(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise ValueError("must_be_positive")
    return value


ok_result = parse_positive_number("5")
fail_result = parse_positive_number("-1")

assert ok_result.success
assert fail_result.failure
assert fail_result.error_code == "PARSE_ERROR"
```

### @d.retry + @d.railway

```python
from flext_core import d

attempts = {"count": 0}


@d.railway(error_code="RETRY_EXAMPLE")
@d.retry(max_attempts=3, delay_seconds=0.01, backoff_strategy="linear")
def flaky_operation() -> int:
    attempts["count"] += 1
    if attempts["count"] < 3:
        raise RuntimeError("transient_error")
    return 123


result = flaky_operation()
assert result.success
assert result.value == 123
assert attempts["count"] == 3
```

### @d.combined

`@d.combined` supports operation logging, optional DI injection, and optional railway wrapping.

```python
from flext_core import d


@d.combined(operation_name="sum_values", use_railway=True, track_perf=False)
def sum_values(values: list[int]) -> int:
    return sum(values)


result = sum_values([1, 2, 3])
assert result.success
assert result.value == 6
```

## Best Practices

- Return `p.Result[T]` from fallible operations.
- Use `flat_map` for steps that already return results.
- Use `map` only for pure value transformations.
- Use `map_error`/`recover` for explicit error strategy.
- Include `error_code` and `error_data` when failures cross boundaries.
