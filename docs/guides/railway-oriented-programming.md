# Railway-Oriented Programming with r[T]

<!-- TOC START -->
- [Overview](#overview)
- [Creating Results](#creating-results)
- [Reading State Safely](#reading-state-safely)
- [flat_map Composition](#flatmap-composition)
- [map, map_error, and recover](#map-maperror-and-recover)
- [map_or for Defaulted Reads](#mapor-for-defaulted-reads)
- [Factory Helpers](#factory-helpers)
- [unwrap_or and unwrap_or_else](#unwrapor-and-unwraporelse)
- [lash for Failure Branching](#lash-for-failure-branching)
- [FlextExceptions at Result Boundaries](#flextexceptions-at-result-boundaries)
- [Typed Exceptions with Metadata](#typed-exceptions-with-metadata)
- [None Handling Techniques](#none-handling-techniques)
- [Exception Propagation with Context](#exception-propagation-with-context)
- [traverse and with_resource](#traverse-and-withresource)
- [Decorator Integration](#decorator-integration)
  - [@d.railway](#drailway)
  - [@d.retry + @d.railway](#dretry-drailway)
  - [@d.combined](#dcombined)
- [Organizing Error Handling](#organizing-error-handling)
- [Best Practices](#best-practices)
<!-- TOC END -->

## Overview

`r[T]` is FLEXT's result container for explicit success and failure flows, while
`e` exposes the structured `FlextExceptions` DSL used both for raised typed
exceptions and for `fail_*` helpers that already return `p.Result[T]`.

- Success path: `r[T].ok(value)`
- Failure path: `r[T].fail(message, error_code=..., error_data=...)`
- Structured result failures: `e.fail_operation(...)`,
  `e.fail_not_found(...)`, `e.fail_validation(...)`
- Typed raised exceptions: `e.ValidationError(...)`, `e.TimeoutError(...)`,
  `e.NotFoundError(...)`

Canonical implementations and live examples:

- [`flext_core.result`](../../src/flext_core/result.py)
- [`flext_core.exceptions`](../../src/flext_core/exceptions.py)
- [`flext_core.decorators`](../../src/flext_core/decorators.py)
- [`examples/ex_01_flext_result.py`](../../examples/ex_01_flext_result.py)
- [`tests/unit/test_exceptions.py`](../../tests/unit/test_exceptions.py)

The goal is predictable composition without exception-driven control flow in
the core path, while still allowing typed exception propagation at the outer
boundaries.

All snippets below are standalone and executable.

## Creating Results

```python
"""Create success and failure results."""

from __future__ import annotations

from flext_core import p, r

success: p.Result[str] = r[str].ok("hello")
failure: p.Result[str] = r[str].fail("cannot_continue", error_code="DOCS_EXAMPLE")

expected_error_code = "DOCS_EXAMPLE"
if not success.success:
    message = "Expected success result"
    raise RuntimeError(message)
if not failure.failure:
    message = "Expected failure result"
    raise RuntimeError(message)
if failure.error_code != expected_error_code:
    message = "Unexpected error code"
    raise RuntimeError(message)
```

## Reading State Safely

```python
"""Read success and failure state safely."""

from __future__ import annotations

from flext_core import r

result = r[int].ok(42)
if result.success:
    expected_value = 42
    if result.value != expected_value:
        message = "Unexpected success value"
        raise RuntimeError(message)

failed = r[int].fail("broken")
expected_error = "broken"
if not failed.failure:
    message = "Expected failed result"
    raise RuntimeError(message)
if failed.error != expected_error:
    message = "Unexpected failure message"
    raise RuntimeError(message)
```

## flat_map Composition

Use `flat_map` when each step already returns `p.Result[...]`.

```python
"""Compose result-producing steps with flat_map."""

from __future__ import annotations

from flext_core import p, r


def validate_email(email: str) -> p.Result[str]:
    """Validate that an email contains the separator used in this example."""
    if "@" not in email:
        return r[str].fail("invalid_email")
    return r[str].ok(email)


def normalize_email(email: str) -> p.Result[str]:
    """Normalize whitespace and casing for a validated email."""
    return r[str].ok(email.strip().lower())


result = (
    r[str].ok(" USER@EXAMPLE.COM ").flat_map(validate_email).flat_map(normalize_email)
)
expected_email = "user@example.com"
if not result.success:
    message = "Expected normalized email success"
    raise RuntimeError(message)
if result.value != expected_email:
    message = "Unexpected normalized email value"
    raise RuntimeError(message)
```

## map, map_error, and recover

```python
"""Transform success and failure branches explicitly."""

from __future__ import annotations

from flext_core import r

length_result = r[str].ok("flext").map(len)
expected_length = 5
if length_result.value != expected_length:
    message = "Unexpected mapped length"
    raise RuntimeError(message)

error_result = r[str].fail("bad_input").map_error(lambda err: f"validation:{err}")
expected_error = "validation:bad_input"
if error_result.error != expected_error:
    message = "Unexpected mapped error"
    raise RuntimeError(message)

recovered = r[str].fail("not_found").recover(lambda _err: "guest")
expected_guest = "guest"
if not recovered.success:
    message = "Expected recovered success"
    raise RuntimeError(message)
if recovered.value != expected_guest:
    message = "Unexpected recovered value"
    raise RuntimeError(message)
```

Prefer `map` for pure value transformations and `map_error` when the failure
text needs to be normalized for the next boundary. `recover` converts a
failure into a success value; if the fallback itself is another
result-producing step, prefer `lash` instead.

## map_or for Defaulted Reads

`map_or` is the compact form used in many runtime call sites when a result must
be reduced to a plain value with a default.

```python
"""Reduce a result into a plain value with a default."""

from __future__ import annotations

from flext_core import r

success_value = r[int].ok(7).map_or(0)
failure_value = r[int].fail("missing").map_or(0)
length_value = r[str].ok("flext").map_or(0, len)

expected_success_value = 7
expected_failure_value = 0
expected_length_value = 5
if success_value != expected_success_value:
    message = "Unexpected success value from map_or"
    raise RuntimeError(message)
if failure_value != expected_failure_value:
    message = "Unexpected failure fallback from map_or"
    raise RuntimeError(message)
if length_value != expected_length_value:
    message = "Unexpected transformed value from map_or"
    raise RuntimeError(message)
```

## Factory Helpers

Use the result factory helpers when the boundary behavior is already known:
`create_from_callable` for exception-to-result adaptation and
`from_validation` for model parsing.

```python
"""Convert a callable and model validation into results."""

from __future__ import annotations

from flext_core import m, r


class User(m.BaseModel):
    """Example model validated through the result factory."""

    name: str
    age: int


callable_result = r[str].create_from_callable(lambda: "created")
validated = r[User].from_validation({"name": "Ada", "age": 30}, User)
invalid = r[User].from_validation({"name": "Ada", "age": "bad"}, User)

if not callable_result.success:
    message = "Expected callable factory success"
    raise RuntimeError(message)
if not validated.success:
    message = "Expected validation success"
    raise RuntimeError(message)
if not invalid.failure:
    message = "Expected validation failure"
    raise RuntimeError(message)
```

For ad-hoc local failures there is also `r[T].fail_op(...)`, but for
structured cross-boundary failures prefer the canonical `e.fail_*` helpers
shown below.

## unwrap_or and unwrap_or_else

```python
"""Extract defaulted values from results."""

from __future__ import annotations

from flext_core import r

ok_value = r[int].ok(7).unwrap_or(0)
fallback_value = r[int].fail("nope").unwrap_or(0)

expected_ok_value = 7
expected_fallback_value = 0
if ok_value != expected_ok_value:
    message = "Unexpected unwrap_or success value"
    raise RuntimeError(message)
if fallback_value != expected_fallback_value:
    message = "Unexpected unwrap_or fallback value"
    raise RuntimeError(message)

lazy_fallback = r[int].fail("still_nope").unwrap_or_else(lambda: 99)
expected_lazy_fallback = 99
if lazy_fallback != expected_lazy_fallback:
    message = "Unexpected unwrap_or_else fallback value"
    raise RuntimeError(message)
```

## lash for Failure Branching

Use `lash` to transform failures into alternate result flows.

```python
"""Recover a failed result with another result-producing branch."""

from __future__ import annotations

from flext_core import p, r


def fallback(_message: str) -> p.Result[int]:
    """Convert an error message into a deterministic fallback result."""
    return r[int].ok(0)


result = r[int].fail("cannot_parse").lash(fallback)
expected_value = 0
if not result.success:
    message = "Expected lash success"
    raise RuntimeError(message)
if result.value != expected_value:
    message = "Unexpected lash fallback value"
    raise RuntimeError(message)
```

## FlextExceptions at Result Boundaries

Use `e.fail_*` when the function already returns `p.Result[T]` but the failure
should carry canonical `error_code` and `error_data`.

```python
"""Use e.fail_* helpers at result-returning boundaries."""

from __future__ import annotations

from flext_core import e, p, r


def fetch_profile_name(user_id: str) -> p.Result[str]:
    """Return a structured not-found failure instead of a raw string."""
    expected_user_id = "u-1"
    if user_id != expected_user_id:
        return e.fail_not_found("user", user_id)
    return r[str].ok("Ada")


def parse_age(raw_value: str) -> p.Result[int]:
    """Normalize parser exceptions to a structured operation failure."""
    try:
        return r[int].ok(int(raw_value))
    except ValueError as exc:
        return e.fail_operation("parse age", exc)


missing = fetch_profile_name("u-2")
parsed = parse_age("30")
failed_parse = parse_age("bad")
expected_parsed_age = 30

if not missing.failure:
    message = "Expected not-found failure"
    raise RuntimeError(message)
missing_error_data = missing.error_data
if missing_error_data is None:
    message = "Expected not-found error data"
    raise RuntimeError(message)
if missing_error_data["resource_id"] != "u-2":
    message = "Unexpected not-found resource id"
    raise RuntimeError(message)
if parsed.value != expected_parsed_age:
    message = "Unexpected parsed age"
    raise RuntimeError(message)
if not failed_parse.failure:
    message = "Expected structured operation failure"
    raise RuntimeError(message)
failed_parse_error_data = failed_parse.error_data
if failed_parse_error_data is None:
    message = "Expected operation error data"
    raise RuntimeError(message)
if failed_parse_error_data["operation"] != "parse age":
    message = "Unexpected operation error payload"
    raise RuntimeError(message)
```

## Typed Exceptions with Metadata

Raised `e.*Error` instances keep structured fields, metadata, and optional
auto-generated correlation ids.

```python
"""Create a typed exception with structured metadata."""

from __future__ import annotations

from flext_core import e

error = e.ValidationError(
    "Invalid profile payload",
    field="email",
    value="bad",
    auto_correlation=True,
    context={"scope": "profile-service", "user_id": "u-1"},
)

if error.field != "email":
    message = "Unexpected validation field"
    raise RuntimeError(message)
if error.value != "bad":
    message = "Unexpected validation value"
    raise RuntimeError(message)
if error.correlation_id is None:
    message = "Expected auto-generated correlation id"
    raise RuntimeError(message)
scope = error.metadata.attributes.get("scope")
if scope != "profile-service":
    message = "Unexpected metadata scope"
    raise RuntimeError(message)
```

## None Handling Techniques

`None` should stay explicit. Keep optional business absence local, but convert
required `None` inputs into structured validation failures as early as
possible.

```python
"""Keep None semantics explicit."""

from __future__ import annotations

from flext_core import e, p, r


def display_name_or_guest(raw_name: str | None) -> str:
    """Business absence stays local and becomes a plain default."""
    if raw_name is None:
        return "guest"
    return raw_name.strip()


def require_email(raw_email: str | None) -> p.Result[str]:
    """Required input converts None into a structured validation failure."""
    if raw_email is None:
        return e.fail_validation("email", error="cannot be None")
    normalized = raw_email.strip().lower()
    if not normalized:
        return e.fail_validation("email", error="cannot be blank")
    return r[str].ok(normalized)


guest_name = display_name_or_guest(None)
email_result = require_email(" Ada@example.com ")
missing_result = require_email(None)
blank_result = require_email("  ")

if guest_name != "guest":
    message = "Unexpected guest fallback"
    raise RuntimeError(message)
if email_result.value != "ada@example.com":
    message = "Unexpected normalized email"
    raise RuntimeError(message)
if not missing_result.failure:
    message = "Expected None validation failure"
    raise RuntimeError(message)
missing_error_data = missing_result.error_data
if missing_error_data is None:
    message = "Expected validation error data for None"
    raise RuntimeError(message)
if missing_error_data["cause"] != "cannot be None":
    message = "Unexpected None validation cause"
    raise RuntimeError(message)
if not blank_result.failure:
    message = "Expected blank validation failure"
    raise RuntimeError(message)
```

## Exception Propagation with Context

When you must switch from a foreign exception to a typed FLEXT exception, use
`raise ... from exc` so the original cause chain is preserved.

```python
"""Wrap a foreign exception in a typed timeout error."""

from __future__ import annotations

from flext_core import e


def fetch_remote_profile() -> str:
    """Convert an infrastructure exception into a typed timeout error."""
    socket_message = "socket stalled"
    timeout_message = "Remote profile lookup timed out"
    try:
        raise RuntimeError(socket_message)
    except RuntimeError as exc:
        raise e.TimeoutError(
            timeout_message,
            operation="fetch profile",
            timeout_seconds=2.0,
            auto_correlation=True,
            context={"service": "profile-api"},
        ) from exc


captured: e.TimeoutError | None = None
try:
    fetch_remote_profile()
except e.TimeoutError as exc:
    captured = exc

if captured is None:
    message = "Expected timeout exception"
    raise RuntimeError(message)
if captured.operation != "fetch profile":
    message = "Unexpected timeout operation"
    raise RuntimeError(message)
if captured.__cause__ is None:
    message = "Expected preserved cause chain"
    raise RuntimeError(message)
service_name = captured.metadata.attributes.get("service")
if service_name != "profile-api":
    message = "Unexpected propagated metadata"
    raise RuntimeError(message)
if captured.correlation_id is None:
    message = "Expected propagated correlation id"
    raise RuntimeError(message)
```

This is the same shape used by retry-style boundaries: translate the foreign
exception once, enrich it with operation metadata, and preserve the original
cause for debugging.

## traverse and with_resource

```python
"""Traverse a sequence through a validating result function."""

from __future__ import annotations

from flext_core import p, r


def validate_item(item: str) -> p.Result[str]:
    """Reject strings shorter than the configured minimum length."""
    minimum_length = 3
    if len(item) < minimum_length:
        return r[str].fail(f"too_short:{item}")
    return r[str].ok(item)


failed = r.traverse(["abc", "ok", "xy"], validate_item)
if not failed.failure:
    message = "Expected traversal failure"
    raise RuntimeError(message)

passed = r.traverse(["abc", "ok1", "xyz"], validate_item)
expected_items = ["abc", "ok1", "xyz"]
if not passed.success:
    message = "Expected traversal success"
    raise RuntimeError(message)
if passed.value != expected_items:
    message = "Unexpected traversal value"
    raise RuntimeError(message)
```

```python
"""Wrap resource setup, use, and cleanup in a result."""

from __future__ import annotations

from flext_core import p, r


def create_connection() -> dict[str, int]:
    """Create the resource used by the operation callback."""
    resource_identifier = 10
    return {"id": resource_identifier}


def use_connection(conn: dict[str, int]) -> p.Result[int]:
    """Use the resource and return the extracted identifier as a result."""
    return r[int].ok(conn["id"])


def close_connection(_conn: dict[str, int]) -> None:
    """Close the resource used by the example."""


resource_result = r.with_resource(
    create_connection,
    use_connection,
    close_connection,
)
expected_identifier = 10
if not resource_result.success:
    message = "Expected resource success"
    raise RuntimeError(message)
if resource_result.value != expected_identifier:
    message = "Unexpected resource identifier"
    raise RuntimeError(message)
```

## Decorator Integration

### @d.railway

```python
"""Wrap an exception-raising function with the railway decorator."""

from __future__ import annotations

from flext_core import d


@d.railway(error_code="PARSE_ERROR")
def parse_positive_number(raw: str) -> int:
    """Parse a strictly positive integer value."""
    value = int(raw)
    minimum_positive_value = 0
    if value <= minimum_positive_value:
        message = "must_be_positive"
        raise ValueError(message)
    return value


ok_result = parse_positive_number("5")
fail_result = parse_positive_number("-1")

expected_error_code = "PARSE_ERROR"
if not ok_result.success:
    message = "Expected railway success"
    raise RuntimeError(message)
if not fail_result.failure:
    message = "Expected railway failure"
    raise RuntimeError(message)
if fail_result.error_code != expected_error_code:
    message = "Unexpected railway error code"
    raise RuntimeError(message)
```

### @d.retry + @d.railway

When retries exhaust, `@d.retry` raises `e.TimeoutError`; with outer
`@d.railway`, the exception is converted back into `p.Result[T]`.

```python
"""Combine retry and railway decorators."""

from __future__ import annotations

from flext_core import d

attempts = {"count": 0}


@d.railway(error_code="RETRY_EXAMPLE")
@d.retry(max_attempts=3, delay_seconds=0.01, backoff_strategy="linear")
def flaky_operation() -> int:
    """Fail twice before returning a stable value."""
    attempts["count"] += 1
    required_attempts = 3
    if attempts["count"] < required_attempts:
        message = "transient_error"
        raise RuntimeError(message)
    return 123


result = flaky_operation()
expected_value = 123
expected_attempts = 3
if not result.success:
    message = "Expected retry success"
    raise RuntimeError(message)
if result.value != expected_value:
    message = "Unexpected retry result value"
    raise RuntimeError(message)
if attempts["count"] != expected_attempts:
    message = "Unexpected retry attempt count"
    raise RuntimeError(message)
```

### @d.combined

`@d.combined` supports operation logging, optional DI injection, and optional
railway wrapping.

```python
"""Use the combined decorator with railway wrapping enabled."""

from __future__ import annotations

from flext_core import d


@d.combined(
    operation_name="sum_values",
    railway_enabled=True,
    track_perf=False,
)
def sum_values(values: list[int]) -> int:
    """Return the arithmetic sum of the provided values."""
    return sum(values)


sample_values = [1, 2, 3]
expected_total = 6
result = sum_values(sample_values)
if not result.success:
    message = "Expected combined decorator success"
    raise RuntimeError(message)
if result.value != expected_total:
    message = "Unexpected combined decorator value"
    raise RuntimeError(message)
```

## Organizing Error Handling

- Adapter and boundary functions should catch foreign exceptions once and
  convert them with `e.fail_*` or raise a typed `e.*Error`.
- Orchestration functions should mostly stay in `p.Result[T]` and compose with
  `flat_map`, `map`, `map_error`, `recover`, and `lash`.
- Normalize required `None` inputs at the first boundary that understands the
  business meaning, typically with `e.fail_validation(...)`.
- Preserve cause chains with `raise ... from exc` whenever you translate from a
  foreign exception to `e.*Error`.
- Collapse `p.Result[T]` into plain values only at the output edge with
  `map_or`, `unwrap_or`, or `unwrap_or_else`.

## Best Practices

- Return `p.Result[T]` from fallible operations in the core flow.
- Use `e.fail_*` for structured failures in result-returning boundaries.
- Raise typed `e.*Error` at imperative or transport boundaries and preserve the
  cause with `raise ... from exc`.
- Treat `None` as business semantics, not as a generic failure marker. Convert
  required `None` inputs early.
- Use `flat_map` for steps that already return results and `lash` when the
  recovery branch also returns a result.
- Use `map` only for pure value transformations.
- Use `map_error` and `recover` for explicit error strategy.
- Use `map_or` and `unwrap_or` only when the caller must collapse a result into
  a plain value.
- Include `error_code`, `error_data`, metadata, and correlation when a failure
  crosses a boundary.
