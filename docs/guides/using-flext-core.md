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
- [Ports and composition](#ports-and-composition)
- [Domain models](#domain-models)
- [Need -> capability -> benefit](#need-capability-benefit)
- [What flext-core must not absorb](#what-flext-core-must-not-absorb)
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
    """Divide ``a`` by ``b`` and fail explicitly on a zero divisor.

    Returns:
        The quotient, or a ``division_by_zero`` failure.

    """
    if b == 0:
        return r[float].fail("division_by_zero")
    return r[float].ok(a / b)


expected_quotient = 5.0
quotient = safe_divide(10, 2)
if not quotient.success or not isclose(quotient.value, expected_quotient):
    message = "Expected a successful quotient"
    raise RuntimeError(message)
if not safe_divide(10, 0).failure:
    message = "Expected division by zero to fail"
    raise RuntimeError(message)
```

## Settings

```python
from flext_core import FlextSettings

settings = FlextSettings.fetch_global()
if not isinstance(settings.model_dump(), dict):
    message = "Expected settings to dump to a mapping"
    raise TypeError(message)
```

Subprojects extend `FlextSettings` with their own `env_prefix`:

```python
from flext_core import FlextSettings, m


class GreetingSettings(FlextSettings):
    """Greeting settings read from ``GREETING_*`` environment variables."""

    model_config = m.SettingsConfigDict(env_prefix="GREETING_", extra="forbid")
```

## Container

```python
from flext_core import FlextContainer, p

container = FlextContainer()
container.bind("service", "ready")
resolved: p.Result[str] = container.resolve("service", type_cls=str)

expected_service = "ready"
if not resolved.success or resolved.value != expected_service:
    message = "Expected the bound service to resolve"
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
    """Service returning a fixed greeting."""

    @override
    def execute(self) -> p.Result[str]:
        """Return the greeting as a successful result."""
        return r[str].ok("Hello!")


expected_greeting = "Hello!"
runtime = GreetingService.fetch_global()
result = runtime.execute()
if not result.success or result.value != expected_greeting:
    message = "Expected the greeting service to succeed"
    raise RuntimeError(message)
```

## Ports and composition

An application composition root constructs adapters and passes them to its
services. A dependency uses `t.Port[p.X]`, with a runtime-checkable protocol and an
excluded field declaration, so service construction validates the supplied
collaborator without serializing it into the request or response.

Service `execute()` returns `p.Result[T]`. Project-specific bases add their own
settings and options through MRO; inheriting directly from `s` does not replace
those local specializations. `fetch_global()` builds the service without
arguments and is not a way to supply required ports. `with_settings()` creates an
isolated settings snapshot; it does not retain an injected settings object by
reference.

## Domain models

`m.Entity` and `m.AggregateRoot` follow Pydantic equality: two instances are equal
when they have the same type and field values, so two entities that share a
`unique_id` but differ in another field are unequal. Compare identity explicitly
with `a.unique_id == b.unique_id`, and key entity collections by `unique_id`. A
mutable entity is not hashable; only frozen models such as `m.Value` hash, by
their field values. Entities define no custom `__eq__`/`__hash__`, and consumers
that relied on identity equality are rewired fleet-wide by rule. This is the
Pydantic way the operator ruled on 2026-10-25 (tracker memory
`operator-ruling-2026-10-25-entity-equality-pydantic-way`; ADR-028 Decision 14).

## Need -> capability -> benefit

| Need               | Capability                                           | Benefit                                                     |
| ------------------ | ---------------------------------------------------- | ----------------------------------------------------------- |
| Service foundation | `s[T]` / `FlextService[T]`                           | Shared validation and lifecycle; `T` is the result payload  |
| Railway results    | `r[T]` / `FlextResult[T]`                            | One failure-preserving result implementation                |
| Typed ports        | `t.Port[p.X]`                                        | Constructor-level dependency injection, no service locator  |
| Data validation    | `m` presets and `u.validate_value`                   | Shared typed boundaries; strict presets for strict ingress  |
| Configuration      | `FlextConfig`, `FlextSettings`, `config`, `settings` | Shared contracts; application objects stay consumer-owned   |
| Handlers           | `h` / `FlextHandlers`                                | Typed operation metadata and registration                   |
| Lazy exports       | Declared root exports                                | Components load on first access through the generated init  |

## What flext-core must not absorb

| Public capability declaration | Accepted decision |
| --- | --- |
| This section directs consumers to generic workspace-analysis and structural-refactoring capabilities, not their implementation libraries. | [FLEXT ADR-028 Decision 15](https://github.com/flext-sh/flext/blob/0.12.0-dev/docs/architecture/adr/028-facade-boundary-orchestrators-declarative-enforcement.md#decision-15) |

- CLI frameworks and command rendering belong to `flext-cli`.
- Code generation, workspace analysis and structural refactoring, locks, and Git belong
  to `flext-infra` capability facades.
- Pytest fixtures and matchers belong to `flext-tests`.
- Domain adapters and LDAP/LDIF/Meltano semantics belong to their packages.
- Application authorization and use cases belong to the application.

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
