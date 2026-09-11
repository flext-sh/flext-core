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

`FlextSettings` is the canonical runtime configuration model in `flext-core`.
It is a Pydantic v2 settings model with:

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
assert settings.log_level == "INFO"
assert settings.debug is False
```

Read `log_level` directly; `debug` and `trace` are independent typed flags on the model.

```python
from flext_core import FlextSettings

settings = FlextSettings(log_level="INFO", debug=True, trace=False)
assert settings.log_level == "INFO"
assert settings.debug is True
assert settings.trace is False
```

## Singleton Access

Use `fetch_global()` for canonical global access.

```python
from flext_core import FlextSettings

base = FlextSettings.fetch_global()
derived = FlextSettings.fetch_global(overrides={"debug": True})

assert isinstance(base, FlextSettings)
assert isinstance(derived, FlextSettings)
assert derived.debug is True
```

## Safe Override Application

Use `clone` to derive a modified copy without mutating the original.

```python
from flext_core import FlextSettings

settings = FlextSettings.fetch_global(overrides={"debug": False})
updated = settings.clone(debug=True)

assert updated is not settings
assert updated.debug is True
assert settings.debug is False
```

## Context-Specific Settings

Use `fetch_global(overrides=...)` to derive worker/request-level configuration from the global singleton.

```python
from flext_core import FlextSettings

worker_settings = FlextSettings.fetch_global(
    overrides={"debug": True, "log_level": "DEBUG"}
)

assert worker_settings.debug is True
assert worker_settings.log_level == "DEBUG"
```

## Custom Settings Models

Subclass `FlextSettings` to define bounded domain settings; base fields are inherited.

```python
from __future__ import annotations

from flext_core import FlextSettings, m


class DocsDemoSettings(FlextSettings):
    model_config = m.ConfigDict(env_prefix="FLEXT_DOCS_DEMO_", extra="ignore")
    feature_enabled: bool = True


docs_settings = DocsDemoSettings()

assert isinstance(docs_settings, DocsDemoSettings)
assert isinstance(docs_settings, FlextSettings)
assert docs_settings.feature_enabled is True
assert docs_settings.log_level == "INFO"
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
assert isinstance(settings.log_level, str)
assert isinstance(settings.debug, bool)
```

## Best Practices

- Read settings from `FlextSettings.fetch_global()` in application entrypoints.
- Use typed fields instead of ad-hoc dictionaries.
- Use `fetch_global(overrides=...)` or `clone(...)` for per-worker configuration.
- Subclass `FlextSettings` for bounded domains.
- Keep secrets in environment variables, not hardcoded in source.
