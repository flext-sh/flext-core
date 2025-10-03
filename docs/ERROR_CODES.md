# FlextExceptions Error Code Reference

**Version**: 1.0.0
**Last Updated**: 2025-10-02
**Status**: Production Ready

---

## Overview

This document provides a comprehensive reference for FlextExceptions error codes used throughout the FLEXT ecosystem. Error codes enable programmatic error handling, monitoring, and analytics.

## Error Code Format

Error codes follow the pattern: `{CATEGORY}_{SUBCATEGORY}_{NUMBER}`

Example: `VALIDATION_ERROR_001`, `CONFIG_MISSING_KEY_001`

---

## Error Code Categories

### 1. VALIDATION_ERROR (001-099)

**Base Exception**: `FlextExceptions.ValidationError`
**Inherits From**: `ValueError`
**HTTP Status**: 400 Bad Request

| Code                   | Description             | Context Fields                       | Example                                   |
| ---------------------- | ----------------------- | ------------------------------------ | ----------------------------------------- |
| `VALIDATION_ERROR_001` | Required field missing  | field, value                         | Email is required                         |
| `VALIDATION_ERROR_002` | Invalid format          | field, value, validation_details     | Invalid email format                      |
| `VALIDATION_ERROR_003` | Value out of range      | field, value, min, max               | Age must be between 18-100                |
| `VALIDATION_ERROR_004` | Invalid type            | field, expected_type, actual_type    | Expected string, got integer              |
| `VALIDATION_ERROR_005` | Duplicate value         | field, value, validation_details     | Username already exists                   |
| `VALIDATION_ERROR_006` | Invalid length          | field, value, min_length, max_length | Password must be 8-64 characters          |
| `VALIDATION_ERROR_007` | Pattern mismatch        | field, value, pattern                | Phone must match +1-XXX-XXX-XXXX          |
| `VALIDATION_ERROR_008` | Invalid enum value      | field, value, allowed_values         | Status must be: active, inactive, pending |
| `VALIDATION_ERROR_009` | Constraint violation    | field, value, constraint             | Must be unique                            |
| `VALIDATION_ERROR_010` | Business rule violation | field, value, rule                   | Cannot delete active user                 |

**Usage Example**:

```python
raise FlextExceptions.ValidationError(
    "Email is required",
    field="email",
    value=None,
)
# Error code auto-assigned: VALIDATION_ERROR_001
```

---

### 2. CONFIGURATION_ERROR (100-199)

**Base Exception**: `FlextExceptions.ConfigurationError`
**Inherits From**: `KeyError`
**HTTP Status**: 500 Internal Server Error

| Code                        | Description                   | Context Fields                            | Example                           |
| --------------------------- | ----------------------------- | ----------------------------------------- | --------------------------------- |
| `CONFIG_MISSING_KEY_001`    | Required config key missing   | config_key, config_file                   | database_url not in settings.YAML |
| `CONFIG_INVALID_VALUE_002`  | Invalid config value          | config_key, value, expected               | Invalid port number               |
| `CONFIG_FILE_NOT_FOUND_003` | Configuration file missing    | config_file, config_path                  | settings.YAML not found           |
| `CONFIG_PARSE_ERROR_004`    | Configuration parse failure   | config_file, parse_error                  | Invalid YAML syntax               |
| `CONFIG_TYPE_MISMATCH_005`  | Wrong config value type       | config_key, expected_type, actual_type    | timeout must be integer           |
| `CONFIG_DEPRECATED_006`     | Deprecated configuration used | config_key, deprecated_since, alternative | Use new_config instead            |
| `CONFIG_CONFLICT_007`       | Conflicting configurations    | config_keys, conflict_reason              | Cannot use both X and Y           |
| `CONFIG_INCOMPLETE_008`     | Incomplete configuration      | config_section, missing_keys              | Database config incomplete        |
| `CONFIG_INVALID_FORMAT_009` | Invalid format                | config_key, format, actual                | Invalid URL format                |
| `CONFIG_PERMISSION_010`     | Permission denied             | config_file, permission_required          | Cannot read /etc/app/config.YAML  |

**Usage Example**:

```python
raise FlextExceptions.ConfigurationError(
    "Database URL not configured",
    config_key="database_url",
    config_file="settings.yaml",
)
# Error code auto-assigned: CONFIG_MISSING_KEY_001
```

---

### 3. NOT_FOUND_ERROR (200-299)

**Base Exception**: `FlextExceptions.NotFoundError`
**Inherits From**: `FileNotFoundError`
**HTTP Status**: 404 Not Found

| Code                      | Description            | Context Fields               | Example                          |
| ------------------------- | ---------------------- | ---------------------------- | -------------------------------- |
| `NOT_FOUND_FILE_001`      | File not found         | resource_path, resource_type | /data/users.csv not found        |
| `NOT_FOUND_RESOURCE_002`  | Resource not found     | resource_type, resource_id   | User ID user-123 not found       |
| `NOT_FOUND_DIRECTORY_003` | Directory not found    | resource_path                | /data/exports/ not found         |
| `NOT_FOUND_ENDPOINT_004`  | API endpoint not found | endpoint, method             | GET /api/v2/users not found      |
| `NOT_FOUND_SERVICE_005`   | Service not found      | service_name, service_type   | LDAP service not available       |
| `NOT_FOUND_DATABASE_006`  | Database not found     | database_name                | Database 'analytics' not found   |
| `NOT_FOUND_TABLE_007`     | Table not found        | table_name, database         | Table 'users' not found in DB    |
| `NOT_FOUND_RECORD_008`    | Record not found       | table, record_id             | No record with ID 42 in users    |
| `NOT_FOUND_MODULE_009`    | Module not found       | module_name                  | Module 'custom_plugin' not found |
| `NOT_FOUND_ATTRIBUTE_010` | Attribute not found    | object_type, attribute_name  | User has no 'phone' attribute    |

**Usage Example**:

```python
raise FlextExceptions.NotFoundError(
    "User not found",
    resource_type="User",
    resource_id="user-123",
)
# Error code auto-assigned: NOT_FOUND_RESOURCE_002
```

---

### 4. TYPE_ERROR (300-399)

**Base Exception**: `FlextExceptions.TypeError`
**Inherits From**: `TypeError`
**HTTP Status**: 400 Bad Request

| Code                           | Description              | Context Fields                         | Example                             |
| ------------------------------ | ------------------------ | -------------------------------------- | ----------------------------------- |
| `TYPE_MISMATCH_001`            | Type mismatch            | expected_type, actual_type, field      | Expected str, got int               |
| `TYPE_CONVERSION_002`          | Type conversion failed   | source_type, target_type, value        | Cannot convert '2025-13-01' to date |
| `TYPE_INVALID_CAST_003`        | Invalid type cast        | from_type, to_type                     | Cannot cast User to dict            |
| `TYPE_NULL_NOT_ALLOWED_004`    | Null/None not allowed    | field, expected_type                   | User.email cannot be None           |
| `TYPE_COLLECTION_MISMATCH_005` | Collection type wrong    | expected_collection, actual_collection | Expected list, got tuple            |
| `TYPE_CALLABLE_REQUIRED_006`   | Callable required        | field, actual_type                     | Handler must be callable            |
| `TYPE_PROTOCOL_VIOLATION_007`  | Protocol not implemented | protocol, missing_methods              | Missing 'handle' method             |
| `TYPE_GENERIC_MISMATCH_008`    | Generic type mismatch    | expected_generic, actual_generic       | Expected List[str], got List[int]   |
| `TYPE_ANNOTATION_MISSING_009`  | Type annotation missing  | parameter, function                    | Parameter 'data' has no type hint   |
| `TYPE_INCOMPATIBLE_010`        | Incompatible types       | type_a, type_b, operation              | Cannot add str + int                |

**Usage Example**:

```python
raise FlextExceptions.TypeError(
    "Expected string, got integer",
    expected_type="str",
    actual_type="int",
    field="username",
)
# Error code auto-assigned: TYPE_MISMATCH_001
```

---

### 5. OPERATION_ERROR (400-499)

**Base Exception**: `FlextExceptions.OperationError`
**Inherits From**: `RuntimeError`
**HTTP Status**: 500 Internal Server Error

| Code                           | Description                 | Context Fields                            | Example                              |
| ------------------------------ | --------------------------- | ----------------------------------------- | ------------------------------------ |
| `OPERATION_FAILED_001`         | Generic operation failure   | operation, details                        | Database query failed                |
| `OPERATION_STATE_INVALID_002`  | Invalid state for operation | current_state, required_state, operation  | Cannot delete active user            |
| `OPERATION_PRECONDITION_003`   | Precondition not met        | operation, precondition, actual_state     | User must be logged in               |
| `OPERATION_POSTCONDITION_004`  | Postcondition failed        | operation, expected_result, actual_result | Transaction not committed            |
| `OPERATION_CONCURRENT_005`     | Concurrent modification     | resource, operation                       | Resource modified by another process |
| `OPERATION_ROLLBACK_006`       | Operation rolled back       | operation, rollback_reason                | Transaction rolled back due to error |
| `OPERATION_PARTIAL_007`        | Partial operation failure   | operation, successful_count, failed_count | 5 of 10 users created                |
| `OPERATION_CIRCUIT_OPEN_008`   | Circuit breaker open        | service, failure_threshold                | Service unavailable (circuit open)   |
| `OPERATION_RATE_LIMITED_009`   | Rate limit exceeded         | operation, limit, window                  | Max 100 requests per minute          |
| `OPERATION_QUOTA_EXCEEDED_010` | Quota exceeded              | resource, quota_limit, current_usage      | Storage quota exceeded               |

**Usage Example**:

```python
raise FlextExceptions.OperationError(
    "Cannot delete active user",
    operation="delete_user",
    current_state="active",
    required_state="inactive",
)
# Error code auto-assigned: OPERATION_STATE_INVALID_002
```

---

### 6. TIMEOUT_ERROR (500-599)

**Base Exception**: `FlextExceptions.TimeoutError`
**Inherits From**: `TimeoutError`
**HTTP Status**: 504 Gateway Timeout

| Code                          | Description              | Context Fields                               | Example                             |
| ----------------------------- | ------------------------ | -------------------------------------------- | ----------------------------------- |
| `TIMEOUT_OPERATION_001`       | Operation timeout        | operation, timeout_seconds, elapsed_time     | Database query timeout (30s)        |
| `TIMEOUT_CONNECTION_002`      | Connection timeout       | host, port, timeout_seconds                  | Connection to 10.0.0.1:5432 timeout |
| `TIMEOUT_REQUEST_003`         | Request timeout          | url, method, timeout_seconds                 | GET /api/users timeout after 60s    |
| `TIMEOUT_LOCK_004`            | Lock acquisition timeout | lock_name, timeout_seconds                   | Failed to acquire lock after 10s    |
| `TIMEOUT_RESPONSE_005`        | Response timeout         | service, operation, timeout_seconds          | Service response timeout            |
| `TIMEOUT_TRANSACTION_006`     | Transaction timeout      | transaction_id, timeout_seconds              | Transaction timeout after 30s       |
| `TIMEOUT_BATCH_007`           | Batch operation timeout  | batch_size, processed_count, timeout_seconds | Processed 500/1000 before timeout   |
| `TIMEOUT_DEADLINE_008`        | Deadline exceeded        | operation, deadline, current_time            | Deadline 2025-10-02T10:00 exceeded  |
| `TIMEOUT_RETRY_EXHAUSTED_009` | Retry attempts exhausted | operation, max_retries, total_time           | Failed after 3 retries (90s)        |
| `TIMEOUT_KEEPALIVE_010`       | Keepalive timeout        | connection, idle_time                        | Connection idle for 300s            |

**Usage Example**:

```python
raise FlextExceptions.TimeoutError(
    "Database query timeout",
    operation="fetch_users",
    timeout_seconds=30,
    elapsed_time=35.2,
)
# Error code auto-assigned: TIMEOUT_OPERATION_001
```

---

## Error Code Usage

### Programmatic Error Handling

```python
from flext_core import FlextExceptions

try:
    operation()
except FlextExceptions.ValidationError as e:
    if e.error_code == "VALIDATION_ERROR_001":
        # Handle required field missing
        return {"error": "Please provide email"}
    elif e.error_code == "VALIDATION_ERROR_002":
        # Handle invalid format
        return {"error": "Email format is invalid"}
```

### Monitoring and Metrics

```python
from prometheus_client import Counter

error_counter = Counter(
    'flext_errors_total',
    'Total errors by code',
    ['error_code', 'service']
)

try:
    operation()
except FlextExceptions.ValidationError as e:
    error_counter.labels(
        error_code=e.error_code,
        service="user-service"
    ).inc()
```

### Logging

```python
import structlog

logger = structlog.get_logger()

try:
    operation()
except FlextExceptions.ValidationError as e:
    logger.error(
        "Validation failed",
        error_code=e.error_code,
        correlation_id=str(e.correlation_id),
        field=e.field,
        value=e.value,
    )
```

### Error Analytics

```python
# Track error patterns
error_analytics = {
    "VALIDATION_ERROR_001": 152,  # Email missing - improve UX
    "VALIDATION_ERROR_002": 43,   # Email format - better validation
    "CONFIG_MISSING_KEY_001": 5,  # Config issue - documentation
    "TIMEOUT_OPERATION_001": 12,  # Performance issue - optimize
}
```

---

## HTTP Status Code Mapping

| Error Category     | HTTP Status | Description           |
| ------------------ | ----------- | --------------------- |
| ValidationError    | 400         | Bad Request           |
| ConfigurationError | 500         | Internal Server Error |
| NotFoundError      | 404         | Not Found             |
| TypeError          | 400         | Bad Request           |
| OperationError     | 500         | Internal Server Error |
| TimeoutError       | 504         | Gateway Timeout       |

---

## Custom Error Codes

### Adding New Error Codes

When creating custom error codes, follow the established pattern:

```python
from flext_core import FlextExceptions

class CustomValidationError(FlextExceptions.ValidationError):
    """Custom validation error with specific code."""

    def __init__(self, message: str, **context):
        super().__init__(message, **context)
        self.error_code = "CUSTOM_VALIDATION_001"
```

### Error Code Conventions

1. **Use uppercase**: `VALIDATION_ERROR_001`
2. **Use underscores**: Not dashes or camelCase
3. **Include category**: First word identifies exception type
4. **Include subcategory**: Second word identifies specific scenario
5. **Include number**: Three digits for uniqueness (001-999)
6. **Keep sequential**: Increment numbers within subcategory

---

## Error Code Evolution

### Version Compatibility

- Error codes are **stable** once released
- Never change meaning of existing error codes
- Deprecated codes must be documented
- New codes get next available number in category

### Deprecation Process

1. Mark error code as deprecated in documentation
2. Add deprecation warning in code comments
3. Maintain backward compatibility for 2 minor versions
4. Remove after proper deprecation cycle

Example:

```python
# DEPRECATED: Use VALIDATION_ERROR_011 instead
# Will be removed in v2.0.0
VALIDATION_ERROR_001_LEGACY = "VALIDATION_ERROR_001"
```

---

## Summary

- **6 error categories** covering all common scenarios
- **60+ predefined error codes** ready to use
- **Structured context fields** for rich debugging
- **HTTP status mapping** for web applications
- **Programmatic handling** via error codes
- **Monitoring ready** for observability platforms

**Start using today**: Import FlextExceptions and get automatic error codes!

```python
from flext_core import FlextExceptions

# Automatic error code assignment
raise FlextExceptions.ValidationError(
    "Email is required",
    field="email",
)
# Error code: VALIDATION_ERROR_001 ✅
```

---

**Document Version**: 1.0.0
**Foundation**: flext-core v0.9.9
**Authority**: FLEXT Foundation Library
**Status**: Production Ready
