# FLEXT Settings Guide

<!-- TOC START -->
- [Overview](#overview)
- [Basic Usage](#basic-usage)
- [Singleton Access](#singleton-access)
- [Safe Override Application](#safe-override-application)
- [Context-Specific Settings](#context-specific-settings)
- [Namespace Registration](#namespace-registration)
- [Environment Variables](#environment-variables)
- [Best Practices](#best-practices)
<!-- TOC END -->

## Overview

`FlextSettings` is the canonical runtime configuration model in `flext-core`.
It is a Pydantic v2 settings model with:

- Typed fields
- Environment resolution
- Singleton access via `fetch_global()`
- Namespace and context override helpers

All snippets below are standalone and executable.

## Basic Usage

Create settings with explicit overrides.

```python
from flext_core import FlextSettings

settings = FlextSettings(log_level="INFO", debug=False, trace=False)
assert settings.log_level == "INFO"
assert settings.debug is False
```

Use `effective_log_level` to resolve runtime logging level when `debug`/`trace` are enabled.

```python
from flext_core import FlextSettings

settings = FlextSettings(log_level="INFO", debug=True, trace=False)
assert settings.effective_log_level in {"DEBUG", "TRACE", "INFO"}
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

Use `apply_override` when setting values dynamically.

```python
from flext_core import FlextSettings

settings = FlextSettings.fetch_global(overrides={"debug": False})
updated = settings.apply_override("debug", True)
invalid = settings.apply_override("does_not_exist", "value")

assert updated is True
assert invalid is False
assert settings.debug is True
```

## Context-Specific Settings

Use context overrides for worker/request-level configuration changes.

```python
from flext_core import FlextSettings

FlextSettings.register_context_overrides("worker-1", debug=True, log_level="DEBUG")
worker_settings = FlextSettings.for_context("worker-1")

assert worker_settings.debug is True
assert worker_settings.log_level == "DEBUG"
```

## Namespace Registration

Register namespaced settings with `auto_register`.

```python
from flext_core import FlextSettings, m


@FlextSettings.auto_register("docs_demo")
class DocsDemoSettings(FlextSettings):
    model_config: ClassVar[m.ConfigDict] = m.ConfigDict(
        env_prefix="FLEXT_DOCS_DEMO_", extra="ignore"
    )
    feature_enabled: bool = True


root_settings = FlextSettings.fetch_global()
docs_settings = root_settings.fetch_namespace("docs_demo", DocsDemoSettings)

assert isinstance(docs_settings, DocsDemoSettings)
assert docs_settings.feature_enabled is True
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
- Use `register_context_overrides` and `for_context` for per-worker configuration.
- Use namespaced settings for bounded domains.
- Keep secrets in environment variables, not hardcoded in source.
