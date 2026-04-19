# Development Guide

## Overview

This page summarizes the daily development loop using executable examples.

## 1. Build a Result Pipeline

```python
from flext_core import p, r


def normalize_email(email: str) -> p.Result[str]:
    if "@" not in email:
        return r[str].fail("invalid_email")
    return r[str].ok(email.strip().lower())


result = r[str].ok(" USER@EXAMPLE.COM ").flat_map(normalize_email)
assert result.success
assert result.value == "user@example.com"
```

## 2. Configure Runtime Settings

```python
from flext_core import FlextSettings

settings = FlextSettings.fetch_global(overrides={"debug": True, "log_level": "DEBUG"})
assert settings.debug is True
assert settings.log_level == "DEBUG"
```

## 3. Wire Services Through Container

```python
from flext_core import FlextContainer, u

container = FlextContainer()
_ = container.factory("logger", lambda: u.fetch_logger(__name__))
logger_result = container.resolve("logger")

assert logger_result.success
```

## 4. Validate Against Examples

```python
from examples.ex_02_flext_settings import Ex02FlextSettings
from examples.ex_12_flext_registry import Ex12RegistryDsl

Ex02FlextSettings("docs/guides/development.md").exercise()
Ex12RegistryDsl("docs/guides/development.md").exercise()
```
