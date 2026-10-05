<!-- AUTO-GENERATED FILE — regenerate through `make gen` from the workspace root. -->
<!-- Source of truth: `<workspace-root>/docs/guides/using-flext-core.md`; adjust that workspace source, never this member projection. -->

# flext-core - Using flext-core

> Project profile: `flext-core`

<!-- TOC START -->

- [Aliases](#aliases)
- [Result flow](#result-flow)
- [Settings](#settings)
- [Container](#container)
- [Logging](#logging)
- [Service runtime](#service-runtime)
- [Good practices](#good-practices)
- [Bad practices](#bad-practices)
- [Related](#related)

<!-- TOC END -->

`flext_core` is the base package for result flow, settings, container wiring, logging,
and service runtime.

## Aliases

Import canonical aliases from the package root:

The examples below import only the aliases they consume from `flext_core`.

| Alias | Purpose                            |
| ----- | ---------------------------------- |
| `c`   | constants / constants namespace    |
| `d`   | decorators                         |
| `e`   | errors / exceptions                |
| `h`   | handlers                           |
| `m`   | models / Pydantic helpers          |
| `p`   | protocols                          |
| `r`   | result (`FlextResult`)             |
| `s`   | service / runtime (`FlextService`) |
| `t`   | typings                            |
| `u`   | utilities                          |
| `x`   | mixins / execution                 |

**Important:** `s` is the service/runtime alias. Settings classes (`FlextSettings`,
`FlextCliSettings`, `FlextTestsSettings`) have no short alias.

## Result flow

Use `r[T]` to construct explicit success or domain-failure results. Do not convert
unexpected runtime exceptions into success or ad-hoc error dictionaries.

```python
from __future__ import annotations

from math import isclose

from flext_core import p, r


def safe_divide(a: float, b: float) -> p.Result[float]:
    """Divide two floats, rejecting a zero divisor.

    Returns:
        The resulting ``p.Result[float]``.

    """
    if b == 0:
        return r[float].fail("division_by_zero")
    return r[float].ok(a / b)


quotient_result = safe_divide(10, 2)
zero_result = safe_divide(10, 0)
expected_quotient = 5.0
if not quotient_result.success:
    message = "Expected division success"
    raise RuntimeError(message)
if not isclose(quotient_result.value, expected_quotient):
    message = "Unexpected division quotient"
    raise RuntimeError(message)
if not zero_result.failure:
    message = "Expected zero divisor failure"
    raise RuntimeError(message)
```

## Settings

```python
from flext_core import FlextSettings

settings = FlextSettings.fetch_global()
snapshot = settings.model_dump()
if not isinstance(snapshot, dict):
    message = "Expected dict settings snapshot"
    raise TypeError(message)
```

Subprojects extend `FlextSettings` with their own `env_prefix`:

```python
from flext_core import FlextSettings, m


class GreetingSettings(FlextSettings):
    """Settings for the greeting demo with its own env prefix."""

    model_config = m.SettingsConfigDict(env_prefix="GREETING_", extra="forbid")
```

## Container

```python
from flext_core import FlextContainer, p

container = FlextContainer()
container.bind("service", "ready")
resolved: p.Result[str] = container.resolve("service", type_cls=str)

expected_service = "ready"
if not resolved.success:
    message = "Expected typed resolution success"
    raise RuntimeError(message)
if resolved.value != expected_service:
    message = "Unexpected resolved service value"
    raise RuntimeError(message)
```

## Logging

```python
from flext_core import u

logger = u.fetch_logger(__name__)
logger.info("user.created", user_id=42)
```

## Service runtime

```python
from typing import override

from flext_core import p, r, s


class GreetingService(s[str]):
    """Service greeting through its execute entry point."""

    @override
    def execute(self) -> p.Result[str]:
        return r[str].ok("Hello!")


runtime = GreetingService.fetch_global()
result = runtime.execute()
expected_greeting = "Hello!"
if not result.success:
    message = "Expected service execution success"
    raise RuntimeError(message)
if result.value != expected_greeting:
    message = "Unexpected service greeting"
    raise RuntimeError(message)
```

## Good practices

- Use aliases instead of importing nested modules directly.
- Use `r[T]` for fallible paths.
- Reset singletons in tests with `FlextSettings.reset_for_testing()` and
  `FlextContainer.reset_for_testing()`.
- Remember: `s` = service/runtime, never settings.

## Bad practices

Do not instantiate the base service to execute domain logic: its `execute()` raises
`NotImplementedError`. Implement the typed operation in a concrete service, and obtain
its singleton through `fetch_global()`.

## Related

- `flext-core/src/flext_core/README.md`
- Foundation API reference
