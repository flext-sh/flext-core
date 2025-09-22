# Accessing FlextConfig Parameters with Pydantic 2.11

## Overview

FlextConfig is a standard Pydantic `BaseSettings` model. Previous releases exposed
helper methods such as `config.get(...)` or mixin utilities that wrapped
`model_dump()` lookups. Those wrappers have been removed in favor of the
first-class attribute semantics provided by Pydantic 2.11. Applications should now
read or update configuration values directly on the model instance using native
Python attribute access.

```python
from flext_core import FlextConfig

config = FlextConfig.get_global_instance()
print(config.environment)  # direct attribute access
```

## Reading Configuration Values

Every field defined on `FlextConfig` is available as an attribute. Use attribute
access for known fields or `getattr` when the field name is determined at
runtime.

```python
# Static access
if config.debug:
    enable_verbose_logging()

# Dynamic lookup
field_name = "timeout_seconds"
current_timeout = getattr(config, field_name)
```

`model_dump()` remains available when a dictionary view of the configuration is
required for serialization or logging.

```python
as_dict = config.model_dump()
```

## Updating Configuration Values

Pydantic validation runs automatically during assignment because
`validate_assignment=True` in the `model_config`. You can therefore mutate
individual fields safely or create derived copies with
`model_copy(update=...)`.

```python
# In-place updates with validation
config.debug = True
setattr(config, "timeout_seconds", 60)

# Immutable style using model_copy
updated_config = config.model_copy(update={"timeout_seconds": 45})
```

Both approaches respect field types and validators defined on `FlextConfig`.

## Singleton Convenience

The singleton helper remains available for obtaining the process-wide
configuration instance.

```python
config = FlextConfig.get_global_instance()
```

Once retrieved, interact with the instance using the same native attribute
semantics shown above.

## Benefits of the Native Approach

1. **Simpler API** – aligns with idiomatic Pydantic models and removes redundant
   wrappers.
2. **Type Safety** – leverages the validated attribute access already provided by
   `BaseSettings`.
3. **Flexibility** – works seamlessly with `model_dump()`, `model_copy()`, and
   `getattr`/`setattr` without custom helper functions.
