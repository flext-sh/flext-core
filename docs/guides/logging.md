# Logging Guide

<!-- TOC START -->

- [Overview](#overview)
- [Global Context](#global-context)
- [Scoped Context](#scoped-context)
- [Context Binding](#context-binding)
- [Request Handler Pattern](#request-handler-pattern)
- [Typed Processing Stages](#typed-processing-stages)
- [Best Practices](#best-practices)

<!-- TOC END -->

## Overview

FLEXT logging is built on `structlog` through `FlextUtilitiesLogging`. The logger
supports:

- Global context for app-wide metadata
- Scoped context for request/operation data
- Level-specific context for verbose diagnostics

All examples below are standalone and executable in markdown code-fence tests.

## Global Context

Use global context for metadata that should be present in all messages.

```python
import io
import time
from contextlib import redirect_stdout

from flext_core import FlextUtilitiesLogging

_ = FlextUtilitiesLogging.bind_global_context(service="flext-core", environment="dev")
stream = io.StringIO()
with redirect_stdout(stream):
    logger = FlextUtilitiesLogging.create_module_logger(__name__)
    _ = logger.info("application_started")
    deadline = time.monotonic() + 0.25
    while (
        time.monotonic() < deadline and "application_started" not in stream.getvalue()
    ):
        time.sleep(0.01)
    if "application_started" not in stream.getvalue():
        message = "Expected application_started log record"
        raise RuntimeError(message)

_ = FlextUtilitiesLogging.unbind_global_context("service", "environment")
```

Use `unbind_global_context` when you want to remove selected keys, or
`clear_global_context` when you want a full reset.

```python
from flext_core import FlextUtilitiesLogging

_ = FlextUtilitiesLogging.bind_global_context(trace_id="trace-001")
_ = FlextUtilitiesLogging.clear_global_context()
```

## Scoped Context

Use `bind_context` to attach context to a logical scope (for example, a request id).

```python
import io
import time
from contextlib import redirect_stdout

from flext_core import FlextUtilitiesLogging

scope = "request"
_ = FlextUtilitiesLogging.bind_context(
    scope=scope,
    request_id="req-123",
    user_id="u-42",
)

stream = io.StringIO()
with redirect_stdout(stream):
    logger = FlextUtilitiesLogging.create_module_logger(__name__)
    _ = logger.info("request_started")
    deadline = time.monotonic() + 0.25
    while time.monotonic() < deadline and "request_started" not in stream.getvalue():
        time.sleep(0.01)
    if "request_started" not in stream.getvalue():
        message = "Expected request_started log record"
        raise RuntimeError(message)

_ = FlextUtilitiesLogging.clear_scope(scope)
```

`clear_scope` removes context associated with that scope.

## Context Binding

Use global context to enrich related log lines and clear it when the scope ends.

```python
import io
import time
from contextlib import redirect_stdout

from flext_core import FlextUtilitiesLogging

_ = FlextUtilitiesLogging.bind_global_context(
    internal_state="cache-miss",
    debug_trace="trace-xyz",
)

stream = io.StringIO()
with redirect_stdout(stream):
    logger = FlextUtilitiesLogging.create_module_logger(__name__)
    _ = logger.debug("debug_message")
    _ = logger.info("info_message")
    deadline = time.monotonic() + 0.25
    while time.monotonic() < deadline and "info_message" not in stream.getvalue():
        time.sleep(0.01)
    if "info_message" not in stream.getvalue():
        message = "Expected info_message log record"
        raise RuntimeError(message)

_ = FlextUtilitiesLogging.clear_global_context()
```

## Request Handler Pattern

The typical flow is: bind request context, log, then clear scope in `finally`.

```python
from __future__ import annotations

import io
import time
from contextlib import redirect_stdout

from flext_core import FlextUtilitiesLogging


def handle_request(request_id: str, user_id: str) -> None:
    """Log one request inside its correlation scope and clean up after.

    Raises:
        RuntimeError: When the expected log record never appears.

    """
    scope = "request"
    _ = FlextUtilitiesLogging.bind_context(
        scope=scope,
        request_id=request_id,
        user_id=user_id,
    )
    try:
        stream = io.StringIO()
        with redirect_stdout(stream):
            logger = FlextUtilitiesLogging.create_module_logger(__name__)
            _ = logger.info("request_processing")
            deadline = time.monotonic() + 0.25
            while (
                time.monotonic() < deadline
                and "request_processing" not in stream.getvalue()
            ):
                time.sleep(0.01)
            if "request_processing" not in stream.getvalue():
                message = "Expected request_processing log record"
                raise RuntimeError(message)
    finally:
        _ = FlextUtilitiesLogging.clear_scope(scope)


handle_request("req-99", "u-10")
```

## Typed Processing Stages

Register custom event transformations with `m.StructlogOptions.processing_stages`
and pass those options to `u.configure_structlog`. Each stage implements
`p.LoggingStage`: it receives the output logger, method name, and
`t.LoggingEvent`, and returns a typed event mapping. Stages execute in registration
order after the built-in event processors and before the final renderer. Nested
JSON context and exception information remain available in the event.

Explicit options reconfigure subsequently created loggers, including after
default logging initialization. Bound loggers already cached retain their
processor chain; create a fresh public logger after changing registration.
Calling `u.configure_structlog()` without options initializes defaults only once.
Invalid noncallable registrations fail model validation. A stage failure stops
processing and its original exception propagates through public logging methods,
including `exception` and `trace`. Successful emissions retain the public
`r[bool]` result contract. Invalid trace formatting also raises its native error.

## Best Practices

- Use global context for stable metadata (service, environment, version).
- Use scoped context for per-request or per-operation values.
- Use level context for diagnostic-only fields.
- Always clear scoped context in `finally` blocks.
- Keep context keys small and deterministic.
