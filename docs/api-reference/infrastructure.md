# Infrastructure Layer API Reference

This section summarizes the infrastructure layer components that provide configuration, logging, and context for dispatcher and domain code.

## Configuration Management

### FlextConfig — Layered Configuration
Layered configuration system supporting environment variables, files, and programmatic overrides with type-safe access.

```python
from flext_core import FlextConfig

config = FlextConfig(
    config_files=["config.toml", "secrets.env"],
    overrides={"debug": True},
)

api_key = config.get("api.key", required=True)
debug_mode = config.get("debug", default=False)
```

## Logging and Observability

### FlextLogger — Structured Logging
Structured logging with correlation metadata and DI-friendly construction.

```python
from flext_core import FlextLogger

logger = FlextLogger(__name__)
logger.info("Application started")

with logger.context(operation="user_creation", user_id="user_123"):
    logger.info("Creating user")
```

### FlextContext — Request and Operation Context
Context object that carries correlation IDs, timing metadata, and arbitrary tags through dispatcher pipelines.

```python
from flext_core import FlextContext

context = FlextContext.create(
    operation_id="op_123",
    user_id="user_456",
    metadata={"source": "api", "version": "1.0"},
)
logger = context.get_logger(__name__)
logger.info("Handling request", extra=context.to_log_context())
```

Infrastructure components keep cross-cutting concerns consistent and testable without polluting domain or application code.
