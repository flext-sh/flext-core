# Extending FlextExceptions for Ecosystem Projects

**Version**: 1.0.0 | **Status**: Production Ready | **Last Updated**: 2025-10-02

This guide shows how domain libraries and application projects can extend the FlextExceptions hierarchy to create domain-specific exception types with structured error handling.

---

## Table of Contents

- [Extending FlextExceptions for Ecosystem Projects](#extending-flextexceptions-for-ecosystem-projects)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Why Extend FlextExceptions](#why-extend-flextexceptions)
  - [Basic Extension Pattern](#basic-extension-pattern)
    - [Step 1: Create Domain Exception Namespace](#step-1-create-domain-exception-namespace)
    - [Step 2: Export from Domain Library](#step-2-export-from-domain-library)
    - [Step 3: Use in Domain Code](#step-3-use-in-domain-code)
  - [Domain Library Examples](#domain-library-examples)
    - [flext-api: HTTP Exceptions](#flext-api-http-exceptions)
    - [flext-ldap: LDAP Exceptions](#flext-ldap-ldap-exceptions)
    - [flext-cli: CLI Exceptions](#flext-cli-cli-exceptions)
    - [flext-meltano: Pipeline Exceptions](#flext-meltano-pipeline-exceptions)
  - [Error Code Assignment](#error-code-assignment)
    - [Reserved Error Code Ranges](#reserved-error-code-ranges)
    - [Custom Error Code Pattern](#custom-error-code-pattern)
  - [Custom Context Fields](#custom-context-fields)
    - [Adding Domain-Specific Context](#adding-domain-specific-context)
    - [Context Best Practices](#context-best-practices)
  - [Best Practices](#best-practices)
    - [1. Inherit from Appropriate Base](#1-inherit-from-appropriate-base)
    - [2. Always Call super().**init**()](#2-always-call-superinit)
    - [3. Provide Meaningful Error Messages](#3-provide-meaningful-error-messages)
    - [4. Use Context for Debugging](#4-use-context-for-debugging)
    - [5. Export from Domain Library Root](#5-export-from-domain-library-root)
  - [Testing Your Exceptions](#testing-your-exceptions)
    - [Test Exception Instantiation](#test-exception-instantiation)
    - [Test Exception Catching](#test-exception-catching)
    - [Test Context Propagation](#test-context-propagation)
    - [Test Error Code Assignment](#test-error-code-assignment)
  - [Summary](#summary)

---

## Overview

FlextExceptions provides a foundation exception hierarchy with:

- **Error codes** for programmatic handling
- **Correlation IDs** for distributed tracing
- **Structured context** for debugging
- **Backward compatibility** with Python exceptions

Domain libraries extend this foundation to add domain-specific exceptions while maintaining the structured error handling patterns.

---

## Why Extend FlextExceptions

**Benefits for Domain Libraries**:

1. **Consistent error handling** across the entire FLEXT ecosystem
2. **Automatic correlation tracking** for distributed operations
3. **Structured logging** with error codes and context
4. **Type-safe error handling** with domain-specific exceptions
5. **Ecosystem compatibility** - works with all FLEXT tools

**Benefits for Users**:

1. **Predictable error handling** - same patterns across all FLEXT libraries
2. **Better debugging** - error codes and correlation IDs trace issues
3. **Programmatic handling** - catch specific error types, not generic exceptions
4. **Rich context** - domain-specific fields provide detailed debugging info

---

## Basic Extension Pattern

### Step 1: Create Domain Exception Namespace

```python
# src/flext_<domain>/exceptions.py
from flext_core import FlextExceptions

class FlextDomainExceptions:
    """Domain-specific exceptions extending FlextExceptions foundation."""

    # Inherit from base exception types
    class DomainError(FlextExceptions.BaseError):
        """Base exception for domain operations."""

        def __init__(
            self,
            message: str,
            *,
            domain_field: str | None = None,
            operation: str | None = None,
            **kwargs: object,
        ) -> None:
            self.domain_field = domain_field
            self.operation = operation

            # Build context with domain fields
            context = kwargs.get("context", {})
            if isinstance(context, dict):
                context.update({
                    "domain_field": domain_field,
                    "operation": operation,
                })

            # Call parent with error code
            super().__init__(
                message,
                code="DOMAIN_ERROR_001",
                context=context,
                **kwargs
            )
```

### Step 2: Export from Domain Library

```python
# src/flext_<domain>/__init__.py
from flext_<domain>.exceptions import FlextDomainExceptions

__all__ = [
    # ... other exports
    "FlextDomainExceptions",
]
```

### Step 3: Use in Domain Code

```python
# src/flext_<domain>/service.py
from flext_<domain> import FlextDomainExceptions

class DomainService:
    def validate_operation(self, data: dict) -> FlextResult[None]:
        if not data.get("required_field"):
            raise FlextDomainExceptions.DomainError(
                "Required field missing",
                domain_field="required_field",
                operation="validate_operation",
            )
        return FlextResult[None].ok(None)
```

---

## Domain Library Examples

### flext-api: HTTP Exceptions

**File**: `flext-api/src/flext_api/exceptions.py`

```python
from flext_core import FlextExceptions

class FlextApiExceptions:
    """HTTP/API specific exceptions with status codes."""

    class ApiError(FlextExceptions.ValidationError):
        """Base API error with HTTP status code."""

        def __init__(
            self,
            message: str,
            *,
            status_code: int = 500,
            endpoint: str | None = None,
            method: str | None = None,
            **kwargs: object,
        ) -> None:
            self.status_code = status_code
            self.endpoint = endpoint
            self.method = method

            context = kwargs.get("context", {})
            if isinstance(context, dict):
                context.update({
                    "status_code": status_code,
                    "endpoint": endpoint,
                    "method": method,
                })

            super().__init__(
                message,
                context=context,
                **kwargs
            )

    class BadRequestError(ApiError):
        """HTTP 400 Bad Request."""

        def __init__(self, message: str, **kwargs: object) -> None:
            super().__init__(message, status_code=400, **kwargs)

    class UnauthorizedError(ApiError):
        """HTTP 401 Unauthorized."""

        def __init__(self, message: str, **kwargs: object) -> None:
            super().__init__(message, status_code=401, **kwargs)

    class NotFoundError(ApiError):
        """HTTP 404 Not Found."""

        def __init__(self, message: str, **kwargs: object) -> None:
            super().__init__(message, status_code=404, **kwargs)

    class RateLimitError(ApiError):
        """HTTP 429 Too Many Requests."""

        def __init__(
            self,
            message: str,
            *,
            retry_after: int | None = None,
            **kwargs: object
        ) -> None:
            self.retry_after = retry_after
            context = kwargs.get("context", {})
            if isinstance(context, dict):
                context["retry_after"] = retry_after
            super().__init__(message, status_code=429, context=context, **kwargs)

# Usage example
def fetch_resource(resource_id: str) -> FlextResult[FlextTypes.Dict]:
    response = make_request(f"/api/resources/{resource_id}")

    if response.status_code == 404:
        raise FlextApiExceptions.NotFoundError(
            f"Resource {resource_id} not found",
            endpoint=f"/api/resources/{resource_id}",
            method="GET",
        )

    return FlextResult[FlextTypes.Dict].ok(response.json())
```

---

### flext-ldap: LDAP Exceptions

**File**: `flext-ldap/src/flext_ldap/exceptions.py`

```python
from flext_core import FlextExceptions

class FlextLdapExceptions:
    """LDAP-specific exceptions with DN and filter context."""

    class LdapError(FlextExceptions.OperationError):
        """Base LDAP error with DN and filter context."""

        def __init__(
            self,
            message: str,
            *,
            dn: str | None = None,
            ldap_filter: str | None = None,
            ldap_code: int | None = None,
            **kwargs: object,
        ) -> None:
            self.dn = dn
            self.ldap_filter = ldap_filter
            self.ldap_code = ldap_code

            context = kwargs.get("context", {})
            if isinstance(context, dict):
                context.update({
                    "dn": dn,
                    "ldap_filter": ldap_filter,
                    "ldap_code": ldap_code,
                })

            super().__init__(
                message,
                context=context,
                **kwargs
            )

    class EntryNotFoundError(LdapError):
        """LDAP entry not found (NO_SUCH_OBJECT)."""

        def __init__(self, message: str, dn: str, **kwargs: object) -> None:
            super().__init__(message, dn=dn, ldap_code=32, **kwargs)

    class EntryAlreadyExistsError(LdapError):
        """LDAP entry already exists (ENTRY_ALREADY_EXISTS)."""

        def __init__(self, message: str, dn: str, **kwargs: object) -> None:
            super().__init__(message, dn=dn, ldap_code=68, **kwargs)

    class InvalidCredentialsError(LdapError):
        """LDAP authentication failed (INVALID_CREDENTIALS)."""

        def __init__(self, message: str, **kwargs: object) -> None:
            super().__init__(message, ldap_code=49, **kwargs)

    class InvalidDnSyntaxError(LdapError):
        """LDAP DN syntax invalid (INVALID_DN_SYNTAX)."""

        def __init__(self, message: str, dn: str, **kwargs: object) -> None:
            super().__init__(message, dn=dn, ldap_code=34, **kwargs)

# Usage example
def get_ldap_entry(connection: LDAPConnection, dn: str) -> FlextResult[FlextTypes.Dict]:
    try:
        entry = connection.search(dn, search_scope="BASE")
        if not entry:
            raise FlextLdapExceptions.EntryNotFoundError(
                f"LDAP entry not found: {dn}",
                dn=dn,
            )
        return FlextResult[FlextTypes.Dict].ok(entry)
    except ldap3.LDAPNoSuchObjectResult:
        raise FlextLdapExceptions.EntryNotFoundError(
            f"LDAP entry does not exist: {dn}",
            dn=dn,
        ) from None
```

---

### flext-cli: CLI Exceptions

**File**: `flext-cli/src/flext_cli/exceptions.py`

```python
from flext_core import FlextExceptions

class FlextCliExceptions:
    """CLI-specific exceptions for command processing."""

    class CliError(FlextExceptions.ValidationError):
        """Base CLI error with command context."""

        def __init__(
            self,
            message: str,
            *,
            command: str | None = None,
            exit_code: int = 1,
            **kwargs: object,
        ) -> None:
            self.command = command
            self.exit_code = exit_code

            context = kwargs.get("context", {})
            if isinstance(context, dict):
                context.update({
                    "command": command,
                    "exit_code": exit_code,
                })

            super().__init__(
                message,
                context=context,
                **kwargs
            )

    class InvalidArgumentError(CliError):
        """Invalid command-line argument."""

        def __init__(
            self,
            message: str,
            *,
            argument: str,
            expected: str | None = None,
            **kwargs: object
        ) -> None:
            self.argument = argument
            self.expected = expected

            context = kwargs.get("context", {})
            if isinstance(context, dict):
                context.update({
                    "argument": argument,
                    "expected": expected,
                })

            super().__init__(message, context=context, exit_code=2, **kwargs)

    class MissingRequiredArgumentError(CliError):
        """Required argument not provided."""

        def __init__(
            self,
            message: str,
            *,
            required_args: FlextTypes.StringList,
            **kwargs: object
        ) -> None:
            self.required_args = required_args

            context = kwargs.get("context", {})
            if isinstance(context, dict):
                context["required_args"] = required_args

            super().__init__(message, context=context, exit_code=2, **kwargs)

# Usage example
def validate_cli_args(args: dict) -> FlextResult[FlextTypes.Dict]:
    if "config" not in args:
        raise FlextCliExceptions.MissingRequiredArgumentError(
            "Missing required argument: --config",
            required_args=["config"],
            command="flext-cli run",
        )
    return FlextResult[FlextTypes.Dict].ok(args)
```

---

### flext-meltano: Pipeline Exceptions

**File**: `flext-meltano/src/flext_meltano/exceptions.py`

```python
from flext_core import FlextExceptions

class FlextMeltanoExceptions:
    """Meltano/Singer pipeline exceptions."""

    class PipelineError(FlextExceptions.OperationError):
        """Base pipeline error with stage context."""

        def __init__(
            self,
            message: str,
            *,
            pipeline_name: str | None = None,
            stage: str | None = None,
            tap_name: str | None = None,
            target_name: str | None = None,
            **kwargs: object,
        ) -> None:
            self.pipeline_name = pipeline_name
            self.stage = stage
            self.tap_name = tap_name
            self.target_name = target_name

            context = kwargs.get("context", {})
            if isinstance(context, dict):
                context.update({
                    "pipeline_name": pipeline_name,
                    "stage": stage,
                    "tap_name": tap_name,
                    "target_name": target_name,
                })

            super().__init__(
                message,
                context=context,
                **kwargs
            )

    class TapExtractionError(PipelineError):
        """Singer tap extraction failed."""

        def __init__(
            self,
            message: str,
            *,
            tap_name: str,
            stream_name: str | None = None,
            **kwargs: object
        ) -> None:
            self.stream_name = stream_name
            context = kwargs.get("context", {})
            if isinstance(context, dict):
                context["stream_name"] = stream_name

            super().__init__(
                message,
                tap_name=tap_name,
                stage="extraction",
                context=context,
                **kwargs
            )

    class TargetLoadError(PipelineError):
        """Singer target load failed."""

        def __init__(
            self,
            message: str,
            *,
            target_name: str,
            batch_id: str | None = None,
            **kwargs: object
        ) -> None:
            self.batch_id = batch_id
            context = kwargs.get("context", {})
            if isinstance(context, dict):
                context["batch_id"] = batch_id

            super().__init__(
                message,
                target_name=target_name,
                stage="loading",
                context=context,
                **kwargs
            )

# Usage example
def run_tap(tap_name: str, config: dict) -> FlextResult[FlextTypes.Dict]:
    try:
        result = execute_tap(tap_name, config)
        return FlextResult[FlextTypes.Dict].ok(result)
    except Exception as e:
        raise FlextMeltanoExceptions.TapExtractionError(
            f"Tap extraction failed: {e}",
            tap_name=tap_name,
            stream_name=config.get("stream"),
        ) from e
```

---

## Error Code Assignment

### Reserved Error Code Ranges

FlextExceptions reserves error code ranges for different domains:

| Range                         | Domain                        | Example                   |
| ----------------------------- | ----------------------------- | ------------------------- |
| `VALIDATION_ERROR_001-099`    | Core validation errors        | `VALIDATION_ERROR_002`    |
| `CONFIGURATION_ERROR_001-099` | Core config errors            | `CONFIGURATION_ERROR_003` |
| `OPERATION_ERROR_001-099`     | Core operation errors         | `OPERATION_ERROR_001`     |
| `API_ERROR_001-099`           | flext-api HTTP errors         | `API_ERROR_404`           |
| `LDAP_ERROR_001-099`          | flext-ldap LDAP errors        | `LDAP_ERROR_032`          |
| `CLI_ERROR_001-099`           | flext-cli command errors      | `CLI_ERROR_002`           |
| `PIPELINE_ERROR_001-099`      | flext-meltano pipeline errors | `PIPELINE_ERROR_001`      |

### Custom Error Code Pattern

```python
class DomainError(FlextExceptions.BaseError):
    """Domain error with custom error code."""

    ERROR_CODE_PREFIX = "MYDOMAIN"  # Customize per domain

    def __init__(self, message: str, error_number: int, **kwargs: object) -> None:
        error_code = f"{self.ERROR_CODE_PREFIX}_ERROR_{error_number:03d}"
        super().__init__(
            message,
            code=error_code,
            **kwargs
        )

# Usage
raise DomainError("Something failed", error_number=42)
# Creates error_code="MYDOMAIN_ERROR_042"
```

---

## Custom Context Fields

### Adding Domain-Specific Context

Each domain library can add custom context fields to provide debugging information:

```python
class DatabaseError(FlextExceptions.OperationError):
    """Database error with query context."""

    def __init__(
        self,
        message: str,
        *,
        query: str | None = None,
        table: str | None = None,
        rows_affected: int | None = None,
        **kwargs: object,
    ) -> None:
        # Store as instance attributes
        self.query = query
        self.table = table
        self.rows_affected = rows_affected

        # Add to context dict for logging/tracing
        context = kwargs.get("context", {})
        if isinstance(context, dict):
            context.update({
                "query": query[:200] if query else None,  # Truncate long queries
                "table": table,
                "rows_affected": rows_affected,
            })

        super().__init__(message, context=context, **kwargs)

# Usage
raise DatabaseError(
    "Query execution failed",
    query="SELECT * FROM users WHERE ...",
    table="users",
    rows_affected=0,
)
```

### Context Best Practices

1. **Truncate large values** - Don't include entire request bodies or responses
2. **Sanitize sensitive data** - Never log passwords, tokens, or credentials
3. **Include debugging hints** - Add fields that help locate the issue
4. **Keep it structured** - Use consistent field names across domain
5. **Document fields** - List expected context fields in docstrings

---

## Best Practices

### 1. Inherit from Appropriate Base

Choose the closest FlextExceptions base class:

```python
# ✅ GOOD - Inherit from specific base
class ApiValidationError(FlextExceptions.ValidationError):
    """API validation error."""
    pass

# ❌ BAD - Inherit from too generic base
class ApiValidationError(FlextExceptions.BaseError):
    """API validation error."""
    pass
```

### 2. Always Call super().**init**()

```python
# ✅ GOOD - Call parent constructor
class DomainError(FlextExceptions.BaseError):
    def __init__(self, message: str, **kwargs: object) -> None:
        super().__init__(message, **kwargs)

# ❌ BAD - Skip parent constructor
class DomainError(FlextExceptions.BaseError):
    def __init__(self, message: str, **kwargs: object) -> None:
        self.message = message  # Lost error_code, correlation_id, etc.
```

### 3. Provide Meaningful Error Messages

```python
# ✅ GOOD - Specific, actionable error message
raise FlextApiExceptions.BadRequestError(
    "Missing required field 'email' in request body",
    field="email",
    endpoint="/api/users",
)

# ❌ BAD - Generic, unhelpful error message
raise FlextApiExceptions.BadRequestError("Bad request")
```

### 4. Use Context for Debugging

```python
# ✅ GOOD - Rich context for debugging
raise FlextLdapExceptions.EntryNotFoundError(
    f"User entry not found: {username}",
    dn=f"uid={username},ou=users,dc=example,dc=com",
    ldap_filter=f"(uid={username})",
)

# ❌ BAD - No debugging context
raise FlextLdapExceptions.EntryNotFoundError("User not found")
```

### 5. Export from Domain Library Root

```python
# src/flext_<domain>/__init__.py
from flext_<domain>.exceptions import FlextDomainExceptions

__all__ = [
    "FlextDomainExceptions",
    # ... other exports
]
```

---

## Testing Your Exceptions

### Test Exception Instantiation

```python
def test_domain_error_creation():
    """Test domain exception can be instantiated with context."""
    error = FlextDomainExceptions.DomainError(
        "Test error",
        domain_field="test_field",
        operation="test_operation",
    )

    assert error.domain_field == "test_field"
    assert error.operation == "test_operation"
    assert error.error_code is not None
    assert error.correlation_id is not None
```

### Test Exception Catching

```python
def test_domain_error_catching():
    """Test domain exception can be caught as base Python exception."""
    with pytest.raises(FlextExceptions.BaseError):
        raise FlextDomainExceptions.DomainError("Test error")

    # Also catchable as domain-specific type
    with pytest.raises(FlextDomainExceptions.DomainError):
        raise FlextDomainExceptions.DomainError("Test error")
```

### Test Context Propagation

```python
def test_domain_error_context():
    """Test context fields are accessible."""
    error = FlextDomainExceptions.DomainError(
        "Test error",
        domain_field="test_field",
    )

    assert "domain_field" in error.context
    assert error.context["domain_field"] == "test_field"
```

### Test Error Code Assignment

```python
def test_domain_error_code():
    """Test error codes are unique and consistent."""
    error1 = FlextDomainExceptions.DomainError("Test 1")
    error2 = FlextDomainExceptions.DomainError("Test 2")

    assert error1.error_code == error2.error_code  # Same type = same code
    assert error1.correlation_id != error2.correlation_id  # Different instance
```

---

## Summary

**Key Takeaways**:

1. ✅ **Extend FlextExceptions** for domain-specific error handling
2. ✅ **Add custom context fields** for debugging information
3. ✅ **Use error code ranges** to avoid conflicts
4. ✅ **Call super().**init**()** to preserve base functionality
5. ✅ **Export from domain root** for ecosystem consistency
6. ✅ **Test exception behavior** to ensure correctness

**Benefits**:

- Consistent error handling across FLEXT ecosystem
- Structured logging with error codes and correlation IDs
- Type-safe exception catching with domain-specific types
- Rich debugging context with domain-specific fields

**Next Steps**:

1. Review [EXCEPTION_HANDLING_GUIDE.md](EXCEPTION_HANDLING_GUIDE.md) for usage patterns
2. Review [ERROR_CODES.md](ERROR_CODES.md) for reserved error codes
3. Implement domain-specific exceptions in your library
4. Test exception handling in your domain code
5. Document custom error codes in your domain library README

---

**Questions or Issues?** Consult the FlextExceptions team or open an issue in flext-core.
