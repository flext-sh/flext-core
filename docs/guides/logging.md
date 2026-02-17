# Logging Guide

## Overview

FLEXT provides a comprehensive structured logging system built on `structlog` with automatic context propagation, scoped contexts, and level-based context management. This guide explains the 3-tier context system and how to use it effectively.

## Canonical Rules

- Follow root governance in `CLAUDE.md`.
- Prefer structured examples that preserve context keys and correlation IDs.
- Keep cross-links in sync with guide and API reference sections.

## 3-Tier Context System

FLEXT logging implements a 3-tier context system that allows you to manage logging context at different levels of granularity:

### 1. Global Context

**Application-wide context via structlog contextvars**

Global context is managed using `structlog.contextvars` and is automatically propagated to all log messages across the entire application. This is the broadest level of context and is ideal for application-wide metadata such as:

- Application version
- Deployment environment
- Service instance ID
- Global correlation IDs

**Usage:**

```python
from flext_core.loggings import FlextLogger

# Bind global context
FlextLogger.Context.bind_global_context(
    app_version="1.0.0",
    environment="production",
    service_id="flext-api-001"
)

# All subsequent log messages will include this context
logger = FlextLogger.create_module_logger(__name__)
logger.info("Application started")  # Includes global context
```

**Unbinding:**

```python
# Unbind specific keys
FlextLogger.Context.unbind_global_context("app_version")

# Unbind all global context
FlextLogger.Context.unbind_global_context()
```

### 2. Scoped Context

**APPLICATION, REQUEST, OPERATION scopes**

Scoped contexts provide isolation for different execution scopes. Each scope maintains its own context dictionary, allowing you to have different context values for different parts of your application.

**Available Scopes:**

- **APPLICATION**: Application-level context (long-lived, shared across requests)
- **REQUEST**: Request-level context (per HTTP request, API call, etc.)
- **OPERATION**: Operation-level context (per business operation, handler execution, etc.)

**Usage:**

```python
from flext_core.loggings import FlextLogger
from flext_core.constants import c

# Bind context to APPLICATION scope
FlextLogger.Context.bind_context(
    scope=c.Context.SCOPE_APPLICATION,
    user_id="user-123",
    tenant_id="tenant-456"
)

# Bind context to REQUEST scope
FlextLogger.Context.bind_context(
    scope=c.Context.SCOPE_REQUEST,
    request_id="req-789",
    http_method="POST",
    endpoint="/api/users"
)

# Bind context to OPERATION scope
FlextLogger.Context.bind_context(
    scope=c.Context.SCOPE_OPERATION,
    operation_id="op-abc",
    handler_name="CreateUserHandler"
)

# Log messages will include context from all active scopes
logger = FlextLogger.create_module_logger(__name__)
logger.info("Processing user creation")  # Includes all scoped contexts
```

**Unbinding:**

```python
# Unbind specific keys from a scope
FlextLogger.Context.unbind_context(
    scope=c.Context.SCOPE_REQUEST,
    keys=["request_id"]
)

# Unbind all context from a scope
FlextLogger.Context.unbind_context(
    scope=c.Context.SCOPE_OPERATION
)
```

### 3. Level Context

**DEBUG-only verbose logging**

Level contexts allow you to add additional context that is only included in log messages at specific log levels. This is useful for verbose debugging information that you don't want to include in production logs.

**Usage:**

```python
from flext_core.loggings import FlextLogger
import logging

# Bind context for DEBUG level only
FlextLogger.Context.bind_context_for_level(
    level=logging.DEBUG,
    internal_state="detailed-state-info",
    debug_trace="trace-123"
)

# This context will only appear in DEBUG level messages
logger = FlextLogger.create_module_logger(__name__)
logger.debug("Debug message")  # Includes level context
logger.info("Info message")     # Does NOT include level context
```

**Unbinding:**

```python
# Unbind specific keys from a level
FlextLogger.Context.unbind_context_for_level(
    level=logging.DEBUG,
    keys=["internal_state"]
)

# Unbind all context from a level
FlextLogger.Context.unbind_context_for_level(level=logging.DEBUG)
```

## Context Precedence

When multiple context sources are active, they are merged in the following order (later sources override earlier ones):

1. **Global Context** (lowest priority)
2. **Scoped Contexts** (APPLICATION, then REQUEST, then OPERATION)
3. **Level Context** (only for matching log levels)
4. **Message-specific context** (highest priority, passed directly to log methods)

## Automatic Context Propagation

All context is automatically propagated to log messages. You don't need to manually pass context to each log call:

```python
# Set context once
FlextLogger.Context.bind_context(
    scope=c.Context.SCOPE_REQUEST,
    request_id="req-123"
)

# All log messages in this scope automatically include request_id
logger = FlextLogger.create_module_logger(__name__)
logger.info("Processing started")  # Automatically includes request_id
logger.warning("Validation failed")  # Automatically includes request_id
logger.error("Operation failed")  # Automatically includes request_id
```

## Best Practices

1. **Use Global Context** for application-wide metadata that doesn't change
2. **Use Scoped Context** for request/operation-specific data that should be isolated
3. **Use Level Context** for verbose debugging information that shouldn't appear in production
4. **Clean up context** when scopes end (e.g., unbind REQUEST context after request completes)
5. **Use context managers** for automatic cleanup when available

## Example: Request Handler Pattern

```python
from flext_core.loggings import FlextLogger
from flext_core.constants import c

class UserHandler:
    def __init__(self):
        self.logger = FlextLogger.create_module_logger(__name__)

    def handle_request(self, request_id: str, user_id: str):
        # Bind REQUEST scope context
        FlextLogger.Context.bind_context(
            scope=c.Context.SCOPE_REQUEST,
            request_id=request_id,
            user_id=user_id
        )

        try:
            self.logger.info("Request started")
            # ... process request ...
            self.logger.info("Request completed")
        finally:
            # Clean up REQUEST scope context
            FlextLogger.Context.unbind_context(
                scope=c.Context.SCOPE_REQUEST
            )
```

## Auto-Configuration

FLEXT automatically configures structlog on first logger creation. You don't need to manually call `FlextRuntime.configure_structlog()` unless you need custom configuration:

```python
# Automatic configuration - no manual setup required
logger = FlextLogger.create_module_logger(__name__)
logger.info("This works automatically!")
```

For custom configuration, you can still call `FlextRuntime.configure_structlog()` explicitly before creating loggers.

## See Also

- [Service Patterns Guide](./service-patterns.md) - Using logging in services
- [Error Handling Guide](./error-handling.md) - Logging errors and exceptions
- [Testing Guide](./testing.md) - Testing with structured logging
