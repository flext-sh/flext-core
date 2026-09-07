<!-- AUTO-GENERATED FILE — regenerate through `make gen APPLY=Y` from the workspace root. -->
<!-- Source of truth: `docs/guides/using-flext-core.md`; adjust that source, never this projection. -->

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

`flext_core` is the base package for result flow, settings, container wiring, logging, and service runtime.

## Aliases

Import canonical aliases from the package root:

```python

```

| Alias | Purpose |
| ------- | --------- |
| `c` | constants / constants namespace |
| `d` | decorators |
| `e` | errors / exceptions |
| `h` | handlers |
| `m` | models / Pydantic helpers |
| `p` | protocols |
| `r` | result (`FlextResult`) |
| `s` | service / runtime (`FlextService`) |
| `t` | typings |
| `u` | utilities |
| `x` | mixins / execution |

**Important:** `s` is the service/runtime alias. Settings classes (`FlextSettings`, `FlextCliSettings`,
`FlextTestsSettings`) have no short alias.

## Result flow

Fallible paths return `r[T]`. Avoid raw exceptions or ad-hoc error dicts for control flow.

```python
from __future__ import annotations

from flext_core import r


def safe_divide(a: float, b: float) -> r[float]:
    if b == 0:
        return r[float].fail("division_by_zero")
    return r.ok(a / b)


assert safe_divide(10, 2).success
assert safe_divide(10, 2).value == 5.0
assert safe_divide(10, 0).failure
```

## Settings

```python
from flext_core import FlextSettings

settings = FlextSettings.fetch_global()
assert isinstance(settings.model_dump(), dict)
```

Subprojects extend `FlextSettings` with their own `env_prefix`:

```python
from flext_core import FlextSettings, m


class FlextCliSettings(FlextSettings):
    model_config = m.SettingsConfigDict(env_prefix="FLEXT_CLI_", extra="ignore")
```

## Container

```python
from flext_core import FlextContainer, p

container = FlextContainer()
container.bind("service", "ready")
resolved: p.Result[str] = container.resolve("service", type_cls=str)

assert resolved.success
assert resolved.value == "ready"
```

## Logging

```python
from flext_core import u

logger = u.fetch_logger(__name__)
logger.info("user.created", user_id=42)
```

## Service runtime

```python
from flext_core import s, FlextSettings

settings = FlextSettings.fetch_global()
runtime = s(settings=settings)
```

## Good practices

- Use aliases instead of importing nested modules directly.
- Use `r[T]` for fallible paths.
- Reset singletons in tests with `FlextSettings.reset_for_testing()` and `FlextContainer.reset_for_testing()`.
- Remember: `s` = service/runtime, never settings.

## Bad practices

```python

```

## Related

- `.agents/skills/using-flext-core/SKILL.md`
- `.agents/skills/coding-standards/SKILL.md`
- `flext-core/src/flext_core/README.md`
