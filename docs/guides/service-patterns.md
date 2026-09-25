# Service Patterns

<!-- TOC START -->

- [Overview](#overview)
- [A service](#a-service)
- [Ports](#ports)
- [Composition root](#composition-root)
- [Settings and the runtime hook](#settings-and-the-runtime-hook)
- [Failures](#failures)
- [Forbidden forms](#forbidden-forms)
- [examples-backed service flows](#examples-backed-service-flows)

<!-- TOC END -->

## Overview

A service is one use case: a `FlextService` (`s`) subclass whose public methods return
`p.Result[...]` and whose collaborators are ports. The fleet contract is
[ADR-019](https://github.com/flext-sh/flext/blob/0.12.0-dev/docs/architecture/adr/019-service-contract-ports-operations.md).
Typed operations and the CLI derived from them arrive in later slices of that ADR; this
guide covers what the kernel enforces today.

| Layer            | Where                         | Role                                                          |
| ---------------- | ----------------------------- | ------------------------------------------------------------- |
| Declarations     | `c → t → p → m → u`           | Data, types, port protocols, models, helpers                  |
| Project base     | `base.py`                     | `Flext<X>ServiceBase(s[...])` and its runtime hook            |
| Use cases        | `services/`                   | One service per use case; imports declarations and `p` ports  |
| Adapters         | A package outside `services/` | Implement ports with the owner's libraries                    |
| Composition root | `api.py`                      | The only place that builds adapters and port-bearing services |

## A service

```python
from __future__ import annotations

from typing import Annotated

from flext_core import m, p, r, s


class CreateUserService(s[str]):
    """Create a user and return its name."""

    username: Annotated[str, m.Field(description="Username for the create flow.")] = ""

    def execute(self) -> p.Result[str]:
        if not self.username:
            return r[str].fail("username_required")
        return r[str].ok(self.username)


assert CreateUserService(username="alice").execute().value == "alice"
assert CreateUserService().execute().failure
```

Fields are ports or business parameters the root fills from `config` and `settings`. The
service module imports `p`, `t` and `m` at runtime: Pydantic resolves field annotations
when the class is created, and `model_rebuild` is forbidden.

## Ports

A port is a `@runtime_checkable` Protocol extending `p.Base` with the minimal capability
the service consumes, published in the project's `p` as `p.<Ns>.<Name>`. The service
declares it with `t.Port`:

```python
from __future__ import annotations

from typing import Protocol, runtime_checkable

from flext_core import m, p, r, s, t


@runtime_checkable
class Clock(p.Base, Protocol):
    """Current time in seconds."""

    def now(self) -> int: ...


class FixedClock:
    """Adapter that always returns the same instant."""

    def now(self) -> int:
        return 42


class StampService(s[int]):
    """Stamp an event with the time its clock reads."""

    clock: t.Port[Clock] = m.Field(exclude=True, description="Clock the stamp reads.")

    def execute(self) -> p.Result[int]:
        return r[int].ok(self.clock.now())


assert StampService(clock=FixedClock()).execute().value == 42
assert "clock" not in StampService.model_json_schema()["properties"]
try:
    StampService.model_validate({"clock": "not a clock"})
except m.ValidationError:
    pass
else:
    raise AssertionError
```

- `t.Port[P]` is `Annotated[P, SkipJsonSchema()]`. Pydantic validates the value with
  `isinstance` on construction and on assignment, and the port never enters the JSON
  Schema. `exclude=True` stays on the field: field metadata cannot live in the alias.
- The port type is a plain Protocol class. A subscripted generic or a concrete class is
  rejected with `TypeError` when the service class is created; Pydantic rejects a
  Protocol that is not runtime-checkable.
- A port has no `None` default and no `default_factory` that builds infrastructure. The
  runtime seeds of `x` are the only typed absence (see below).

## Composition root

The project's `api.py` is the only module that builds adapters and passes them to
services (pure dependency injection). One adapter shared by two services is a variable
passed to both constructors:

```python notest
from __future__ import annotations

clock = SystemClock()
stamps = StampService(clock=clock)
audit = AuditService(clock=clock)
```

- Constructing an adapter performs no I/O; it connects on first use, so importing
  `api.py` touches no infrastructure.
- `fetch_global()` builds the per-class singleton with no arguments, so it serves only
  services without ports. A service with a required port raises `ValidationError` there.
- `FlextContainer` is the registry of the core runtime (settings, context, command bus,
  logger). Services and adapters never call it.

## Settings and the runtime hook

The project base declares its settings class once, in a classmethod the core reads
through `p.RuntimeBootstrapProvider`; `FlextService` itself declares no hook:

```python notest
from __future__ import annotations


class FlextBillingServiceBase(s[m.Billing.Invoice]):
    """Service base of the billing project."""

    @classmethod
    def runtime_bootstrap_options(cls) -> m.RuntimeBootstrapOptions:
        return m.RuntimeBootstrapOptions(settings_type=FlextBillingSettings)
```

- `u.resolve_runtime_options(component)` merges the hook's options with the instance
  seeds `runtime_settings`, `settings_type`, `settings_overrides` and `initial_context`;
  a seed set on the instance wins. It also accepts an `m.RuntimeBootstrapOptions`, which
  it returns unchanged; any other source raises `TypeError`.
- `u.build_service_runtime(source)` builds the validated `m.ServiceRuntime`: injected or
  loaded settings, the context, a scope of the shared container, and the container's
  command bus as dispatcher. Failing to resolve the dispatcher raises with its cause.
- Seeds and ports are validated: `runtime_settings` must satisfy `p.Settings` and
  `initial_context` must satisfy `p.Context`. Runtime seeds never enter the JSON Schema.
- A service never reads a global `settings` or `config`; the root passes values through
  fields or the runtime. A base never redeclares `__init__` nor uses
  `settings or X.fetch_global()`.

## Failures

The first failure propagates with its cause: inside a result boundary as
`r.fail(msg, exception=exc)` or `e.fail_*`, outside as a typed `e.*` exception raised
`from` the cause. `unwrap()` and `.value` chain the carried exception:

```python
from __future__ import annotations

from flext_core import r

failure = r[int].fail("lookup failed", exception=KeyError("user-7"))
try:
    failure.unwrap()
except RuntimeError as exc:
    assert isinstance(exc.__cause__, KeyError)
else:
    raise AssertionError
```

Never replace a failure with `None`, `""`, `{}`, a default, `ok(True)`, a skipped item
or `unwrap_or(sentinel)`.

## Forbidden forms

1. `SkipValidation` on a dependency, including `Annotated[..., t.SkipValidation]`.
2. `getattr(x, "member", default)` on a typed member, or attribute-name probes.
3. Hand-rolled singletons (`fetch_instance`, `fetch_global_instance`, a local
   `_instance`).
4. `settings or X.fetch_global()`, or `__init__` redeclared to inject settings.
5. A service calling another service's `fetch_global()`, or building infrastructure,
   including `PrivateAttr(default_factory=…)`.
6. `except …: return <default>`, or `contextlib.suppress` around an owner call.
7. `model_copy(update=)` to build an object that must be validated.
8. Compatibility aliases, or old and new paths side by side.

## examples-backed service flows

```python
from examples.ex_11_flext_service import ExampleService

ExampleService.run()
```
