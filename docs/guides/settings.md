# FLEXT Settings Guide

<!-- TOC START -->

- [Overview](#overview)
- [Basic Usage](#basic-usage)
- [Singleton Access](#singleton-access)
- [Safe Override Application](#safe-override-application)
- [Context-Specific Settings](#context-specific-settings)
- [Custom Settings Models](#custom-settings-models)
- [Environment Variables](#environment-variables)
- [Best Practices](#best-practices)

<!-- TOC END -->

## Overview

`FlextSettings` is the canonical runtime configuration model in `flext-core`. It is a
Pydantic v2 settings model with:

- Typed fields
- Environment resolution
- Singleton access via `fetch_global()`
- Override helpers (`clone`, `fetch_global(overrides=...)`, `update_global`)

All snippets below are standalone and executable.

## Basic Usage

Create settings with explicit overrides.

```python
from flext_core import FlextSettings

settings = FlextSettings(log_level="INFO", debug=False, trace=False)
expected_level = "INFO"
if settings.log_level != expected_level:
    message = "Unexpected log level value"
    raise RuntimeError(message)
if settings.debug is not False:
    message = "Expected debug override to stay disabled"
    raise RuntimeError(message)
```

Read `log_level` directly; `debug` and `trace` are independent typed flags on the model.

```python
from flext_core import FlextSettings

settings = FlextSettings(log_level="INFO", debug=True, trace=False)
expected_level = "INFO"
if settings.log_level != expected_level:
    message = "Unexpected log level value"
    raise RuntimeError(message)
if settings.debug is not True:
    message = "Expected debug override to be enabled"
    raise RuntimeError(message)
if settings.trace is not False:
    message = "Expected trace override to stay disabled"
    raise RuntimeError(message)
```

## Singleton Access

Use `fetch_global()` for canonical global access.

```python
from flext_core import FlextSettings

base = FlextSettings.fetch_global()
derived = FlextSettings.fetch_global(overrides={"debug": True})

if not isinstance(base, FlextSettings):
    message = "Expected base settings type"
    raise TypeError(message)
if not isinstance(derived, FlextSettings):
    message = "Expected derived settings type"
    raise TypeError(message)
if derived.debug is not True:
    message = "Expected debug override in derived settings"
    raise RuntimeError(message)
```

## Safe Override Application

Use `clone` to derive a modified copy without mutating the original.

```python
from flext_core import FlextSettings

settings = FlextSettings.fetch_global(overrides={"debug": False})
updated = settings.clone(debug=True)

if updated is settings:
    message = "Expected clone to produce a new instance"
    raise RuntimeError(message)
if updated.debug is not True:
    message = "Expected debug override in cloned settings"
    raise RuntimeError(message)
if settings.debug is not False:
    message = "Expected original settings to stay unchanged"
    raise RuntimeError(message)
```

## Context-Specific Settings

Use `fetch_global(overrides=...)` to derive worker/request-level configuration from the
global singleton.

```python
from flext_core import FlextSettings

worker_settings = FlextSettings.fetch_global(
    overrides={"debug": True, "log_level": "DEBUG"},
)

expected_level = "DEBUG"
if worker_settings.debug is not True:
    message = "Expected debug override in worker settings"
    raise RuntimeError(message)
if worker_settings.log_level != expected_level:
    message = "Unexpected worker log level"
    raise RuntimeError(message)
```

## Custom Settings Models

Subclass `FlextSettings` to define bounded domain settings; base fields are inherited.

```python
from __future__ import annotations

from flext_core import FlextSettings, m


class DocsDemoSettings(FlextSettings):
    """Settings subclass with a demo-scoped env prefix."""

    model_config = m.ConfigDict(env_prefix="FLEXT_DOCS_DEMO_", extra="ignore")
    feature_enabled: bool = True


docs_settings = DocsDemoSettings()
expected_level = "INFO"

if not isinstance(docs_settings, DocsDemoSettings):
    message = "Expected subclass instance"
    raise TypeError(message)
if not isinstance(docs_settings, FlextSettings):
    message = "Expected FlextSettings compatibility"
    raise TypeError(message)
if docs_settings.feature_enabled is not True:
    message = "Expected default feature flag to stay enabled"
    raise RuntimeError(message)
if docs_settings.log_level != expected_level:
    message = "Unexpected inherited log level"
    raise RuntimeError(message)
```

## Environment Variables

FLEXT settings are environment-aware through Pydantic settings.

```bash
export FLEXT_LOG_LEVEL=DEBUG
export FLEXT_DEBUG=true
```

Then in code:

```python
from flext_core import FlextSettings

settings = FlextSettings()
if not isinstance(settings.log_level, str):
    message = "Expected string log level"
    raise TypeError(message)
if not isinstance(settings.debug, bool):
    message = "Expected boolean debug flag"
    raise TypeError(message)
```

## Best Practices

- Read settings from `FlextSettings.fetch_global()` in application entrypoints.
- Use typed fields instead of ad-hoc dictionaries.
- Use `fetch_global(overrides=...)` or `clone(...)` for per-worker configuration.
- Subclass `FlextSettings` for bounded domains.
- Keep secrets in environment variables, not hardcoded in source.
