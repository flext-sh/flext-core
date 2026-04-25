# Logging Guide


<!-- TOC START -->
- [Overview](#overview)
- [Global Context](#global-context)
- [Scoped Context](#scoped-context)
- [Context Binding](#context-binding)
- [Request Handler Pattern](#request-handler-pattern)
- [Best Practices](#best-practices)
<!-- TOC END -->

## Overview

FLEXT logging is built on `structlog` through `FlextLogger`. The logger supports:

- Global context for app-wide metadata
- Scoped context for request/operation data
- Level-specific context for verbose diagnostics

All examples below are standalone and executable in markdown code-fence tests.

## Global Context

Use global context for metadata that should be present in all messages.

```python
from flext_core import FlextLogger

_ = FlextLogger.bind_global_context(service="flext-core", environment="dev")
logger = FlextLogger.create_module_logger(__name__)
_ = logger.info("application_started")

_ = FlextLogger.unbind_global_context("service", "environment")
```

Use `unbind_global_context` when you want to remove selected keys, or `clear_global_context` when you want a full reset.

```python
from flext_core import FlextLogger

_ = FlextLogger.bind_global_context(trace_id="trace-001")
_ = FlextLogger.clear_global_context()
```

## Scoped Context

Use `bind_context` to attach context to a logical scope (for example, a request id).

```python
from flext_core import FlextLogger

scope = "request"
_ = FlextLogger.bind_context(scope=scope, request_id="req-123", user_id="u-42")

logger = FlextLogger.create_module_logger(__name__)
_ = logger.info("request_started")

_ = FlextLogger.clear_scope(scope)
```

`clear_scope` removes context associated with that scope.

## Context Binding

Use global context to enrich related log lines and clear it when the scope ends.

```python
from flext_core import FlextLogger

_ = FlextLogger.bind_global_context(
    internal_state="cache-miss",
    debug_trace="trace-xyz",
)

logger = FlextLogger.create_module_logger(__name__)
_ = logger.debug("debug_message")
_ = logger.info("info_message")

_ = FlextLogger.clear_global_context()
```

## Request Handler Pattern

The typical flow is: bind request context, log, then clear scope in `finally`.

```python
from flext_core import FlextLogger


def handle_request(request_id: str, user_id: str) -> None:
    scope = "request"
    _ = FlextLogger.bind_context(scope=scope, request_id=request_id, user_id=user_id)
    logger = FlextLogger.create_module_logger(__name__)
    try:
        _ = logger.info("request_processing")
    finally:
        _ = FlextLogger.clear_scope(scope)


handle_request("req-99", "u-10")
```

## Best Practices

- Use global context for stable metadata (service, environment, version).
- Use scoped context for per-request or per-operation values.
- Use level context for diagnostic-only fields.
- Always clear scoped context in `finally` blocks.
- Keep context keys small and deterministic.
