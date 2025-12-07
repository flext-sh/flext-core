# Error Handling Guide

**Status**: Production Ready | **Version**: 0.10.0 | **Pattern**: Railway-Oriented Programming

Comprehensive guide to error handling strategies in FLEXT-Core using the railway-oriented programming pattern with FlextResult[T].

## Overview

FLEXT-Core provides **functional error handling** through `FlextResult[T]` instead of exceptions. This enables:

- Composable error handling through monadic operations
- Type-safe error propagation
- Predictable error flows
- Recoverable failure scenarios

**Philosophy:** Errors are values that flow through your program like any other data.

## Core Concepts

### FlextResult[T] - The Foundation

`FlextResult[T]` is a monad that represents either:

- **Success**: Contains a value of type `T`
- **Failure**: Contains an error message

```python
from flext_core import FlextResult

# Success result
success = FlextResult[int].ok(42)

# Failure result
failure = FlextResult[int].fail("Something went wrong")
```

### Checking Result State

```python
from flext_core import FlextResult

result = FlextResult[str].ok("value")

# Check success state
if result.is_success:
    value = result.unwrap()
    print(f"Success: {value}")

# Check failure state
if result.is_failure:
    error = result.error
    print(f"Error: {error}")

# Alternative check
if not result.is_success:
    print(f"Failed with: {result.error}")
```

### Extracting Values Safely

```python
from flext_core import FlextResult

result = FlextResult[int].ok(42)

# Get value (raises on failure)
value = result.unwrap()  # 42

# Get value with default
value = result.unwrap_or(0)  # 42

# Get value with computation
value = result.unwrap_or_else(lambda e: 0)  # 42

# Both .data and .value work (backward compatibility)
data = result.data  # 42
value = result.value  # 42
```

## Railway-Oriented Programming

### The Railway Pattern

Think of your program as a railway with two tracks:

- **Success track**: Happy path
- **Failure track**: Error path

Once on the failure track, you stay on it until explicitly recovered.

```python
from flext_core import FlextResult

def validate_email(email: str) -> FlextResult[str]:
    """Validate email format."""
    if "@" not in email:
        return FlextResult[str].fail("Invalid email format")
    return FlextResult[str].ok(email)

def check_email_available(email: str) -> FlextResult[str]:
    """Check if email is available."""
    reserved = ["REDACTED_LDAP_BIND_PASSWORD@example.com", "test@example.com"]
    if email in reserved:
        return FlextResult[str].fail("Email already taken")
    return FlextResult[str].ok(email)

def send_confirmation(email: str) -> FlextResult[str]:
    """Send confirmation email."""
    # Pretend this works
    return FlextResult[str].ok(f"Confirmation sent to {email}")

# Railway: one failure anywhere stops the flow
email_result = (
    validate_email("user@example.com")
    .flat_map(check_email_available)
    .flat_map(send_confirmation)
)

if email_result.is_success:
    print(f"✅ {email_result.unwrap()}")
else:
    print(f"❌ {email_result.error}")
```

### Monadic Operations

#### map() - Transform Success Values

```python
from flext_core import FlextResult

result = FlextResult[int].ok(5)

# Transform the success value
new_result = result.map(lambda x: x * 2)  # FlextResult[int].ok(10)

# Chain multiple transformations
result = (
    FlextResult[int].ok(5)
    .map(lambda x: x * 2)      # 10
    .map(lambda x: x + 3)      # 13
    .map(lambda x: str(x))     # "13"
)
```

#### flat_map() - Chain Operations

```python
from flext_core import FlextResult

def divide(a: float, b: float) -> FlextResult[float]:
    if b == 0:
        return FlextResult[float].fail("Division by zero")
    return FlextResult[float].ok(a / b)

# Chain operations, one failure stops the flow
result = (
    divide(10, 2)              # FlextResult[float].ok(5.0)
    .flat_map(lambda x: divide(x, 0))  # FlextResult[float].fail("Division by zero")
    .flat_map(lambda x: divide(x, 2))  # Never executes (stayed on failure track)
)

print(result.error)  # "Division by zero"
```

#### filter() - Add Success Conditions

```python
from flext_core import FlextResult

result = FlextResult[int].ok(15)

# Add condition to success path
result = result.filter(
    lambda x: x > 10,
    "Value must be greater than 10"
)  # FlextResult[int].ok(15)

# Failing condition
result = result.filter(
    lambda x: x < 10,
    "Value must be less than 10"
)  # FlextResult[int].fail("Value must be less than 10")
```

#### map_error() - Transform Error Messages

```python
from flext_core import FlextResult

def risky_operation() -> FlextResult[str]:
    return FlextResult[str].fail("Database connection failed")

result = (
    risky_operation()
    .map_error(lambda e: f"Operation failed: {e}")
    .map_error(lambda e: f"[ERROR] {e}")
)

print(result.error)  # "[ERROR] Operation failed: Database connection failed"
```

## Practical Error Handling Patterns

### Pattern 1: Validation Pipeline

```python
from flext_core import FlextResult

def validate_username(username: str) -> FlextResult[str]:
    """Validate username."""
    if not username:
        return FlextResult[str].fail("Username cannot be empty")
    if len(username) < 3:
        return FlextResult[str].fail("Username must be at least 3 characters")
    if not username.isalnum():
        return FlextResult[str].fail("Username must be alphanumeric")
    return FlextResult[str].ok(username)

def validate_email(email: str) -> FlextResult[str]:
    """Validate email."""
    if "@" not in email or "." not in email:
        return FlextResult[str].fail("Invalid email format")
    return FlextResult[str].ok(email)

def validate_password(password: str) -> FlextResult[str]:
    """Validate password."""
    if len(password) < 8:
        return FlextResult[str].fail("Password must be at least 8 characters")
    if not any(c.isupper() for c in password):
        return FlextResult[str].fail("Password must contain uppercase letter")
    if not any(c.isdigit() for c in password):
        return FlextResult[str].fail("Password must contain digit")
    return FlextResult[str].ok(password)

# Use validation pipeline
def register_user(username: str, email: str, password: str) -> FlextResult[str]:
    """Register user with full validation."""
    return (
        validate_username(username)
        .flat_map(lambda u: validate_email(email).map(lambda e: (u, e)))
        .flat_map(lambda ue: validate_password(password).map(lambda p: (*ue, p)))
        .map(lambda upe: f"User {upe[0]} ({upe[1]}) registered")
    )

# Test
result = register_user("alice", "alice@example.com", "SecurePass123")
if result.is_success:
    print(f"✅ {result.unwrap()}")
else:
    print(f"❌ {result.error}")
```

### Pattern 2: Database Operations

```python
from flext_core import FlextResult, FlextLogger

logger = FlextLogger(__name__)

class UserRepository:
    """Repository for user data operations."""

    def get_user_by_id(self, user_id: str) -> FlextResult[dict]:
        """Get user by ID."""
        try:
            # Simulate database lookup
            users = {"1": {"id": "1", "name": "Alice"}}
            if user_id not in users:
                return FlextResult[dict].fail(f"User {user_id} not found")
            return FlextResult[dict].ok(users[user_id])
        except Exception as e:
            logger.error(f"Database error: {e}")
            return FlextResult[dict].fail("Database operation failed")

    def save_user(self, user: dict) -> FlextResult[dict]:
        """Save user to database."""
        try:
            # Validate before saving
            if not user.get("name"):
                return FlextResult[dict].fail("User name is required")
            # Simulate save
            logger.info(f"User {user['id']} saved")
            return FlextResult[dict].ok(user)
        except Exception as e:
            logger.error(f"Save failed: {e}")
            return FlextResult[dict].fail("Failed to save user")

# Usage
repo = UserRepository()

result = (
    repo.get_user_by_id("1")
    .map(lambda user: {**user, "name": user["name"].upper()})
    .flat_map(repo.save_user)
)

if result.is_success:
    print(f"✅ {result.unwrap()}")
else:
    print(f"❌ {result.error}")
```

### Pattern 3: External Service Calls

```python
from flext_core import FlextResult
import requests
from typing import Any

class ExternalService:
    """Integration with external service."""

    def call_api(self, endpoint: str) -> FlextResult[dict]:
        """Call external API."""
        try:
            response = requests.get(f"https://api.example.com/{endpoint}", timeout=5)
            if response.status_code != 200:
                return FlextResult[dict].fail(
                    f"API error {response.status_code}: {response.text}"
                )
            return FlextResult[dict].ok(response.json())
        except requests.Timeout:
            return FlextResult[dict].fail("API request timeout")
        except requests.ConnectionError:
            return FlextResult[dict].fail("API connection failed")
        except Exception as e:
            return FlextResult[dict].fail(f"API error: {str(e)}")

    def get_user_data(self, user_id: str) -> FlextResult[dict]:
        """Get user data from external service."""
        return (
            self.call_api(f"users/{user_id}")
            .map_error(lambda e: f"Failed to fetch user: {e}")
        )

# Usage
service = ExternalService()

result = service.get_user_data("123")
if result.is_success:
    print(f"User: {result.unwrap()}")
else:
    print(f"Error: {result.error}")
```

### Pattern 4: Error Recovery

```python
from flext_core import FlextResult

def risky_operation() -> FlextResult[str]:
    return FlextResult[str].fail("Primary operation failed")

def fallback_operation() -> FlextResult[str]:
    return FlextResult[str].ok("Fallback operation succeeded")

# Try primary, fall back on failure
result = risky_operation()

if result.is_failure:
    result = fallback_operation()

print(f"Result: {result.unwrap()}")  # "Fallback operation succeeded"

# Or use lash pattern (error handling with FlextResult)
def handle_failure(error: str) -> FlextResult[str]:
    """Handle failure with fallback."""
    print(f"Failed with: {error}, trying fallback...")
    return fallback_operation()

result = risky_operation().lash(handle_failure)
```

### Pattern 5: Batch Operations

```python
from flext_core import FlextResult
from typing import List

def process_items(items: List[str]) -> FlextResult[List[str]]:
    """Process multiple items, collecting all results."""
    results = []
    errors = []

    for item in items:
        result = process_item(item)
        if result.is_success:
            results.append(result.unwrap())
        else:
            errors.append(result.error)

    # Return success if all processed, failure if any error
    if errors:
        return FlextResult[List[str]].fail(
            f"Failed to process {len(errors)} items: {', '.join(errors)}"
        )

    return FlextResult[List[str]].ok(results)

def process_item(item: str) -> FlextResult[str]:
    """Process single item."""
    if not item:
        return FlextResult[str].fail("Empty item")
    return FlextResult[str].ok(f"Processed: {item}")

# Usage
result = process_items(["item1", "item2", "item3"])
if result.is_success:
    print(f"✅ Processed: {result.unwrap()}")
else:
    print(f"❌ {result.error}")
```

## Error Types and Categorization

### Domain Errors (Business Logic)

```python
from flext_core import FlextResult

def withdraw_from_account(amount: float, balance: float) -> FlextResult[float]:
    """Withdraw from account - domain error."""
    if amount > balance:
        # Domain error - user can understand this
        return FlextResult[float].fail(
            f"Insufficient funds. Available: {balance}, Requested: {amount}"
        )
    return FlextResult[float].ok(balance - amount)

# Usage
result = withdraw_from_account(100, 50)
if result.is_failure:
    print(f"Transaction failed: {result.error}")  # Clear, user-facing
```

### System Errors (Infrastructure)

```python
from flext_core import FlextResult
import os

def read_config_file(path: str) -> FlextResult[str]:
    """Read configuration - system error."""
    try:
        with open(path, 'r') as f:
            return FlextResult[str].ok(f.read())
    except FileNotFoundError:
        return FlextResult[str].fail(f"Configuration file not found: {path}")
    except PermissionError:
        return FlextResult[str].fail(f"Permission denied reading: {path}")
    except Exception as e:
        return FlextResult[str].fail(f"System error reading config: {str(e)}")
```

## Best Practices

### 1. Always Return FlextResult from Operations That Can Fail

```python
# ✅ CORRECT
def find_user(user_id: str) -> FlextResult[dict]:
    if user_id not in users_db:
        return FlextResult[dict].fail(f"User {user_id} not found")
    return FlextResult[dict].ok(users_db[user_id])

# ❌ WRONG - Using exceptions
def find_user(user_id: str) -> dict:
    if user_id not in users_db:
        raise ValueError(f"User {user_id} not found")  # Don't do this
    return users_db[user_id]
```

### 2. Chain Operations with flat_map

```python
# ✅ CORRECT - Operations chain properly
result = (
    get_user_id()
    .flat_map(validate_user)
    .flat_map(load_user_data)
    .map(format_user_response)
)

# ❌ WRONG - Messy error checking
user_id = get_user_id()
if user_id.is_failure:
    return user_id
validation = validate_user(user_id.unwrap())
if validation.is_failure:
    return validation
# ... etc
```

### 3. Use Meaningful Error Messages

```python
# ✅ CORRECT - Clear, actionable
return FlextResult[str].fail(
    "Invalid email format. Expected format: name@domain.com"
)

# ❌ WRONG - Vague
return FlextResult[str].fail("Invalid input")
```

### 4. Handle Errors at Application Boundaries

```python
from flext_core import FlextResult, FlextLogger

logger = FlextLogger(__name__)

def api_handler(request) -> dict:
    """Handle API request."""
    result = process_request(request)

    if result.is_success:
        return {"status": "success", "data": result.unwrap()}
    else:
        logger.error(f"Request failed: {result.error}")
        return {"status": "error", "message": result.error}

def process_request(request) -> FlextResult[dict]:
    """Business logic - returns FlextResult."""
    # ... implementation
    pass
```

### 5. Log Errors Appropriately

```python
from flext_core import FlextLogger, FlextResult

logger = FlextLogger(__name__)

def risky_operation() -> FlextResult[str]:
    """Operation that might fail."""
    try:
        result = do_something()
        return FlextResult[str].ok(result)
    except Exception as e:
        # Log with context
        logger.error("Operation failed", extra={
            "error": str(e),
            "error_type": type(e).__name__
        })
        return FlextResult[str].fail(f"Operation failed: {str(e)}")
```

## Summary

Error handling in FLEXT-Core:

- ✅ Use `FlextResult[T]` for all fallible operations
- ✅ Chain operations with `flat_map()` and `map()`
- ✅ Handle errors at application boundaries
- ✅ Use meaningful error messages
- ✅ Log errors with context
- ✅ Never use exceptions for normal control flow

This approach makes error handling explicit, composable, and maintainable.

## Next Steps

1. **Railway Patterns**: Deep dive into [Railway-Oriented Programming](./railway-oriented-programming.md) for advanced composition
2. **Decorators**: Learn about [Error Handling with Decorators](./railway-oriented-programming.md#decorator-composition-with-railway-pattern) for automatic error recovery
3. **Testing**: See [Testing Guide](./testing.md) for testing error scenarios without mocks
4. **Services**: Check [Service Patterns](./service-patterns.md) for service-level error handling
5. **API Reference**: Review [FlextResult API](../api-reference/foundation.md#flextresult) for complete method reference

## See Also

- [Railway-Oriented Programming](./railway-oriented-programming.md) - Complete ROP patterns and composition
- [Testing Guide](./testing.md) - Testing error scenarios with real implementations
- [Service Patterns](./service-patterns.md) - Error handling in domain services
- [Dependency Injection Advanced](./dependency-injection-advanced.md) - Error handling with DI
- [API Reference: FlextResult](../api-reference/foundation.md#flextresult) - Complete API documentation
- **FLEXT CLAUDE.md**: Architecture principles and development workflow

---

**Example from FLEXT Ecosystem**: See `src/flext_tests/test_result.py` for comprehensive test cases demonstrating all error handling patterns.
