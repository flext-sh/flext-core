# Dependency Injection Advanced

<!-- TOC START -->

- [Overview](#overview)
- [Reusing Official Example Code](#reusing-official-example-code)
- [Registration Rules](#registration-rules)
- [Core Container Operations](#core-container-operations)
- [Scoped Containers](#scoped-containers)
- [Factory Auto-Registration](#factory-auto-registration)
- [Best Practices](#best-practices)

<!-- TOC END -->

## Overview

This guide focuses on real `FlextContainer` usage with the current API. Examples are
backed by executable code from the `examples/` package.

Services do not use the container. A service declares each collaborator as a port
(`t.Port[p.X]`) and the project's `api.py` passes adapters to its constructor; see
[Service Patterns](service-patterns.md). `FlextContainer` is the registry of the core
runtime (settings, context, command bus, logger) and the tool for infrastructure code
that composes that runtime. It has no protocol-keyed binding and no string-keyed
`provide`/`wire` bridge: pure dependency injection through constructors is the
composition model.

## Reusing Official Example Code

Use the canonical container example as the reference path.

```python
from examples.ex_08_flext_container import Ex08FlextContainer

demo = Ex08FlextContainer("docs/guides/dependency-injection-advanced.md")
demo.exercise()
```

The `Ex08FlextContainer` flow exercises binding, factories, resources, the registration
rules, resolution, and scoped containers.

## Registration Rules

The container keeps one mapping of services, factories and resources, and every write
(`bind`, `factory`, `resource`, a `m.ServiceRegistrationSpec`, a scope declaration or
factory auto-registration) passes one private write path. That path raises
`e.ValidationError` instead of ignoring the write when:

- the name is empty;
- the name is already registered, whatever its kind (service, factory or resource);
- the name is reserved for the core runtime (`c.CONTAINER_RESERVED_NAMES`: `settings`,
  `logger`, `context`, `command_bus`);
- the value fails its record validation, for example a factory that is not callable.

The reserved core services resolve normally but stay out of `has`, `names` and `drop`.
To replace a public registration, `drop` it first.

```python
from flext_core import FlextContainer, c, e

container = FlextContainer.shared().scope()
_ = container.bind("feature_flag", True)

try:
    _ = container.bind("feature_flag", False)
except e.ValidationError as exc:
    assert "feature_flag" in str(exc)
else:
    raise AssertionError

try:
    _ = container.bind(c.ServiceName.LOGGER, "not a logger")
except e.ValidationError as exc:
    assert "reserved" in str(exc)
else:
    raise AssertionError

assert container.resolve("feature_flag").value is True
assert container.resolve(c.ServiceName.LOGGER).success
assert not container.has(c.ServiceName.LOGGER)
```

## Core Container Operations

```python
from flext_core import FlextContainer, u

container = FlextContainer.shared().scope()

_ = container.bind("app_name", "flext-core")
_ = container.factory("module_logger", lambda: u.fetch_logger(__name__))

app_name = container.resolve("app_name")
logger = container.resolve("module_logger")

assert app_name.value == "flext-core"
assert logger.success
assert set(container.names()) >= {"app_name", "module_logger"}
```

A factory and a resource are invoked on every `resolve`; a callable that raises, or
returns a value that is not a registerable service, yields a failed result carrying the
cause.

## Scoped Containers

`scope(...)` builds an isolated container that inherits the public registrations of its
parent. Registrations declared by the scope's `m.ServiceRegistrationSpec` replace
inherited names, and the scope binds its own core services to its own settings and
context. Writes to the scope never reach the parent.

```python
from flext_core import FlextContainer, m

root = FlextContainer.shared().scope()
_ = root.bind("tenant", "default")

scoped = root.scope(
    subproject="tenant_a",
    registration=m.ServiceRegistrationSpec(services={"tenant": "tenant_a"}),
)

assert scoped.resolve("tenant").value == "tenant_a"
assert root.resolve("tenant").value == "default"
assert scoped.resolve("settings").value is scoped.settings
assert scoped.context.get("subproject").value == "tenant_a"
```

## Factory Auto-Registration

`FlextContainer.shared(auto_register_factories=True)` registers every `@d.factory()`
function of the calling module through the same write path. A caller that cannot be
resolved to a module imported in `sys.modules` raises `e.ValidationError`; a duplicate
or reserved factory name raises as for any other write.

## Best Practices

- Keep service names stable and explicit; never reuse a name without `drop`.
- Prefer `bind` for concrete instances and `factory` for deferred construction.
- Let a registration error propagate: it names the rule the write broke.
- Use `scope(...)` for isolation when composing runtime contexts.
- Never resolve a service's collaborator from the container inside the service; pass it
  as a port from the composition root.
