# Troubleshooting Guide

<!-- TOC START -->
- [Overview](#overview)
- [1. Validate Settings Shape](#1-validate-settings-shape)
- [2. Validate Container Resolution](#2-validate-container-resolution)
- [3. Validate Result Flow](#3-validate-result-flow)
- [4. Re-run Canonical Examples](#4-re-run-canonical-examples)
- [5. Debug Logger Context](#5-debug-logger-context)
- [Checklist](#checklist)
<!-- TOC END -->

## Overview

This page lists practical checks for common runtime issues in `flext-core`.
All snippets are executable and aligned to the current public API.

## 1. Validate Settings Shape

```python
from flext_core import FlextSettings

settings = FlextSettings.fetch_global()
snapshot = settings.model_dump()

assert isinstance(snapshot, dict)
assert "log_level" in snapshot
```

## 2. Validate Container Resolution

```python
from flext_core import FlextContainer

container = FlextContainer()
_ = container.bind("service_name", "demo")

resolved = container.resolve("service_name")
assert resolved.success
assert resolved.value == "demo"
```

## 3. Validate Result Flow

```python
from flext_core import p, r


def parse_positive(value: int) -> p.Result[int]:
    if value <= 0:
        return r[int].fail("value_must_be_positive")
    return r[int].ok(value)


ok_result = parse_positive(1)
fail_result = parse_positive(0)

assert ok_result.success
assert fail_result.failure
```

## 4. Re-run Canonical Examples

When behavior drifts, execute the official examples used by maintainers.

```python
from examples.ex_02_flext_settings import Ex02FlextSettings
from examples.ex_08_flext_container import Ex08FlextContainer

Ex02FlextSettings("docs/guides/troubleshooting.md").exercise()
Ex08FlextContainer("docs/guides/troubleshooting.md").exercise()
```

## 5. Debug Logger Context

```python
from flext_core import FlextLogger

_ = FlextLogger.bind_global_context(component="troubleshooting")
logger = FlextLogger.create_module_logger(__name__)
_ = logger.info("diagnostic_event")
_ = FlextLogger.clear_global_context()
```

## Checklist

- Check `FlextSettings.fetch_global()` returns expected values.
- Check `container.resolve(name)` success/failure explicitly.
- Check result pipelines with `r[T]` contracts before integration steps.
- Reuse `examples/ex_*.py` as the source of truth for API usage.
