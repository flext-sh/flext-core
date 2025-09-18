# Getting Started

> Updated for the 1.0.0 modernization plan (unified dispatcher, context-first observability, aligned configuration/domain services).

---

## Prerequisites

- Python 3.13 or newer
- Poetry (preferred) or pip for dependency management
- Git for source checkout

Verify your toolchain:

```bash
python --version
poetry --version
```

---

## Installation

```bash
git clone https://github.com/flext-sh/flext-core.git
cd flext-core
make setup  # installs dependencies and pre-commit hooks
```

Validate the install:

```bash
poetry run python - <<'PY'
from flext_core import FlextResult
print("FlextResult ready:", FlextResult[str].ok("✅"))
PY
```

---

## First Workflow

The snippet below wires together the modernization pillars:

```python
from flext_core import (
    FlextConfig,
    FlextContainer,
    FlextContext,
    FlextDispatcher,
    FlextLogger,
    FlextResult,
)

class AppConfig(FlextConfig):
    greeting: str = "hello"

config = AppConfig()
container = FlextContainer.get_global()
container.register(AppConfig.__name__, config)
container.register("logger", FlextLogger("getting-started"))

dispatcher = FlextDispatcher()

class HelloHandler:
    def handle(self, message: dict[str, str]) -> FlextResult[str]:
        FlextContext.Request.set_user_id(message.get("user", "anonymous"))
        greeting = container.get(AppConfig.__name__).unwrap().greeting
        return FlextResult[str].ok(f"{greeting}, {message['user']}")

dispatcher.register_command(dict, HelloHandler())
print(dispatcher.dispatch({"user": "FLEXT"}).unwrap())
```

Key takeaways:

- Configuration is centralised via `FlextConfig` and injected through the container.
- The dispatcher handles registration and dispatch; downstream services only depend on the handler contract.
- `FlextContext` stores request metadata that loggers (and other services) can enrich automatically.

---

## Next Steps

1. Read the [Architecture](architecture.md) guide for the modernization pillars and layering decisions.
2. Browse the [API Reference](api-reference.md) to understand the public surface guaranteed in the 1.x line.
3. Explore real code in `examples/` (see the updated [Examples README](../examples/README.md)).
4. Review the [Development](development.md) playbook before contributing.

With these steps complete you are ready to help shepherd the 1.0.0 modernization release.
