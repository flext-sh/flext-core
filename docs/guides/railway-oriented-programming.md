# Railway-Oriented Programming with FlextResult[T]

**Status**: Production Ready | **Version**: 0.9.9 | **Coverage**: 100% type-safe

Railway-Oriented Programming (ROP) is a functional programming pattern that treats error handling as a first-class citizen. Instead of raising exceptions, operations return results that encapsulate either success or failure, enabling predictable error propagation through monadic composition.

## Core Concept: The Railway Metaphor

Imagine two parallel tracks:

```
Success Track (happy path):     value -> validate -> transform -> output
                                   ↓         ↓            ↓          ↓
                            ─────────────────────────────────────────────

Failure Track (error path):    error -> error -> error -> error (short-circuit)
                                   ↓         ↓            ↓
                            ─────────────────────────────────────────────
```

When an operation fails, execution automatically switches to the failure track and skips remaining operations. No exceptions needed.

## FlextResult[T]: The Railway Implementation

`FlextResult[T]` is FLEXT's railway-oriented result type. It wraps either:
- **Success**: `FlextResult[T].ok(data: T)` - wraps successful data
- **Failure**: `FlextResult[T].fail(error: str)` - wraps error information

### Creating Results

#### Success Case

```python
from flext_core import FlextResult

# Wrap successful data
result: FlextResult[str] = FlextResult[str].ok("Hello, World!")

# Generic type inference
result = FlextResult.ok("data")

# None is valid success (for void operations)
result = FlextResult[None].ok(None)  # Represents successful completion
```

**Source**: `src/flext_core/result.py:313-337`

#### Failure Case

```python
from flext_core import FlextResult

# Simple error
result: FlextResult[str] = FlextResult[str].fail("Operation failed")

# With structured error information
result = FlextResult[float].fail(
    "Division by zero not allowed",
    error_code="MATH_ERROR",
    error_data={"dividend": 10, "divisor": 0},
)

# Error codes from FlextConstants for consistency
from flext_core import FlextConstants
result = FlextResult[dict].fail(
    "Configuration missing required field",
    error_code=FlextConstants.Errors.CONFIG_ERROR,
    error_data={"missing_field": "database_url"},
)
```

**Source**: `src/flext_core/result.py:342-392`

### Checking Result State

```python
from flext_core import FlextResult

result = FlextResult[int].ok(42)

# Check state with properties
if result.is_success:
    print(f"Success: {result.value}")

if result.is_failure:
    print(f"Error: {result.error}")

# Alternative names (same behavior)
if result.success:
    print("Succeeded")

if result.failed:
    print("Failed")

# Access error details
if result.is_failure:
    message = result.error          # str
    code = result.error_code        # str | None
    data = result.error_data        # dict[str, object]
```

### Accessing Success Values

Three ways to extract the success value (all raise on failure):

```python
from flext_core import FlextResult

result = FlextResult[str].ok("value")

# 1. Property access (recommended)
value = result.value  # Returns str, raises on failure

# 2. Monadic unwrap (alternative name)
value = result.unwrap()  # Returns str, raises on failure

# 3. Via properties (safe)
if result.is_success:
    value = result.value  # Type-safe - we know it's success

# All three throw ValidationError on failure:
failed_result = FlextResult[str].fail("Error")
value = failed_result.value  # Raises ValidationError: "Attempted to access value on failed result"
```

**Source**: `src/flext_core/result.py:283-292`

## Monadic Operations: Composing Railways

The power of ROP is **composability**. Chain operations without exception handling:

### flat_map: Operating on Success

`flat_map` is the monadic bind operator. It takes a function that returns a `FlextResult` and chains operations:

```python
from flext_core import FlextResult

def validate_email(email: str) -> FlextResult[str]:
    """Validate email format."""
    if "@" not in email:
        return FlextResult[str].fail("Invalid email: missing @")
    return FlextResult[str].ok(email)

def check_domain(email: str) -> FlextResult[str]:
    """Check if domain exists."""
    domain = email.split("@")[1]
    if domain == "invalid.com":
        return FlextResult[str].fail("Domain not allowed", error_code="DOMAIN_ERROR")
    return FlextResult[str].ok(email)

def send_verification(email: str) -> FlextResult[str]:
    """Send verification email."""
    # Pretend to send email
    return FlextResult[str].ok(f"Verification sent to {email}")

# Railway composition - stops at first failure
result = (
    FlextResult[str].ok("user@example.com")
    .flat_map(validate_email)      # If success, validate
    .flat_map(check_domain)         # If success, check domain
    .flat_map(send_verification)    # If success, send email
)

if result.is_success:
    print(f"✅ {result.value}")
else:
    print(f"❌ {result.error}")

# Example outputs:
# ✅ Verification sent to user@example.com
# ❌ Invalid email: missing @
# ❌ Domain not allowed
```

**Key insight**: If ANY operation fails, execution stops and propagates the failure. No exception handling needed.

### map: Transforming Success Values

`map` transforms the success value while keeping the railway structure:

```python
from flext_core import FlextResult

result: FlextResult[str] = FlextResult[str].ok("hello")

# Transform the success value
transformed: FlextResult[int] = result.map(len)  # FlextResult[int]
assert transformed.value == 5

# Chain multiple transformations
final = (
    FlextResult[str].ok("  spaces  ")
    .map(str.strip)              # Remove whitespace
    .map(str.upper)              # Convert to uppercase
    .map(len)                    # Get length
)
assert final.value == 6

# map ignores failures (they pass through unchanged)
failed = FlextResult[str].fail("Error")
result = failed.map(str.upper)  # Still fails, map is skipped
assert result.is_failure
```

### filter: Conditional Success

`filter` applies a predicate. Failure if predicate is false:

```python
from flext_core import FlextResult

def is_adult(age: int) -> bool:
    return age >= 18

# Success case
result = (
    FlextResult[int].ok(25)
    .filter(is_adult)  # Predicate passes
)
assert result.is_success and result.value == 25

# Failure case
result = (
    FlextResult[int].ok(16)
    .filter(is_adult)  # Predicate fails
)
assert result.is_failure

# With custom error message
from flext_core import FlextResult

result = (
    FlextResult[int].ok(16)
    .filter(
        is_adult,
        failure_message="User must be 18 or older"
    )
)
assert result.is_failure
assert "18 or older" in result.error
```

### recover: Error Recovery with Fallback

`recover` applies a function to transform errors back into successes with fallback values:

```python
from flext_core import FlextResult

def fetch_user(user_id: str) -> FlextResult[dict]:
    """Simulate database fetch that might fail."""
    if user_id == "missing":
        return FlextResult[dict].fail("User not found", error_code="NOT_FOUND")
    return FlextResult[dict].ok({"id": user_id, "name": "Alice"})

# Recover with fallback data
result = (
    fetch_user("missing")
    .recover(lambda error: {"id": "guest", "name": "Guest User"})
)
assert result.is_success
assert result.value["name"] == "Guest User"  # Fallback used

# Recover with conditional logic
def get_default_user_for_error(error: str) -> dict:
    if "NOT_FOUND" in error:
        return {"id": "guest", "name": "Guest"}
    else:
        return {"id": "error", "name": "Error"}

result = fetch_user("missing").recover(get_default_user_for_error)
assert result.value["id"] == "guest"
```

**Source**: `src/flext_core/result.py`

### map_error: Transform Errors

`map_error` transforms error messages while keeping the failure state:

```python
from flext_core import FlextResult

def parse_config(config: str) -> FlextResult[dict]:
    """Parse configuration that might fail."""
    if not config.startswith("{"):
        return FlextResult[dict].fail("Invalid JSON format")
    return FlextResult[dict].ok({"parsed": True})

# Transform error message
result = (
    parse_config("invalid")
    .map_error(lambda e: f"Configuration error: {e}")
)
assert result.is_failure
assert "Configuration error" in result.error

# Chain map_error
result = (
    FlextResult[str].fail("Original error")
    .map_error(lambda e: e.upper())
    .map_error(lambda e: f"[ERROR] {e}")
)
assert result.error == "[ERROR] ORIGINAL ERROR"
```

**Source**: `src/flext_core/result.py`

### safe_call: Exception-Safe Function Wrapping

`safe_call` wraps functions that might raise exceptions, converting them to FlextResult:

```python
from flext_core import FlextResult
import json

def parse_json_unsafe(text: str) -> dict:
    """Parse JSON - might raise JSONDecodeError."""
    return json.loads(text)

# Safe wrapper
result = FlextResult.safe_call(parse_json_unsafe, '{"valid": "json"}')
assert result.is_success
assert result.value["valid"] == "json"

# With invalid input
result = FlextResult.safe_call(parse_json_unsafe, 'invalid json')
assert result.is_failure
assert "Expecting" in result.error  # JSONDecodeError message

# With arguments
def divide(a: int, b: int) -> float:
    return a / b

result = FlextResult.safe_call(divide, 10, 0)
assert result.is_failure  # Catches ZeroDivisionError

result = FlextResult.safe_call(divide, 10, 2)
assert result.is_success
assert result.value == 5.0
```

**Source**: `src/flext_core/result.py`

### from_exception: Exception Conversion

`from_exception` explicitly converts exceptions to FlextResult failures:

```python
from flext_core import FlextResult

def risky_operation() -> FlextResult[str]:
    """Perform risky operation, converting exceptions to failures."""
    try:
        # Some operation that might raise
        value = int("not a number")
        return FlextResult[str].ok(str(value))
    except ValueError as e:
        # Convert exception to failure
        return FlextResult[str].from_exception(e, error_code="PARSE_ERROR")

result = risky_operation()
assert result.is_failure
assert "invalid literal" in result.error
assert result.error_code == "PARSE_ERROR"

# Or use it directly
try:
    data = json.loads("{invalid json}")
except json.JSONDecodeError as e:
    result = FlextResult.from_exception(e, error_code="JSON_ERROR")
    assert result.is_failure
```

**Source**: `src/flext_core/result.py`

### or_else: Alternative Results

`or_else` provides an alternative FlextResult if the first one fails:

```python
from flext_core import FlextResult

def fetch_primary() -> FlextResult[dict]:
    return FlextResult[dict].fail("Primary unavailable")

def fetch_backup() -> FlextResult[dict]:
    return FlextResult[dict].ok({"source": "backup", "data": "value"})

# Try primary, fallback to backup
result = fetch_primary().or_else(lambda _: fetch_backup())
assert result.is_success
assert result.value["source"] == "backup"

# Success case (or_else is skipped)
def fetch_primary_success() -> FlextResult[dict]:
    return FlextResult[dict].ok({"source": "primary"})

result = fetch_primary_success().or_else(lambda _: fetch_backup())
assert result.value["source"] == "primary"  # Primary used

# Chain multiple fallbacks
result = (
    fetch_primary()
    .or_else(lambda _: fetch_primary())  # Still fails
    .or_else(lambda _: fetch_backup())   # This succeeds
)
assert result.is_success
```

**Source**: `src/flext_core/result.py`

## Real-World Patterns

### Pattern 1: Form Validation

```python
from flext_core import FlextResult
from pydantic import BaseModel, Field

class UserRegistration(BaseModel):
    email: str = Field(pattern=r'^[^@]+@[^@]+\.[^@]+$')
    password: str = Field(min_length=8)
    username: str = Field(min_length=3, max_length=20)

def validate_registration(data: dict) -> FlextResult[UserRegistration]:
    """Validate user registration with FlextResult."""
    try:
        user = UserRegistration(**data)
        return FlextResult[UserRegistration].ok(user)
    except ValueError as e:
        return FlextResult[UserRegistration].fail(
            f"Validation failed: {e}",
            error_code="VALIDATION_ERROR",
            error_data={"provided": data},
        )

def check_username_available(user: UserRegistration) -> FlextResult[UserRegistration]:
    """Check if username is available."""
    # Pretend to check database
    if user.username.lower() == "REDACTED_LDAP_BIND_PASSWORD":
        return FlextResult[UserRegistration].fail(
            "Username already taken",
            error_code="USERNAME_TAKEN",
        )
    return FlextResult[UserRegistration].ok(user)

def create_user(user: UserRegistration) -> FlextResult[dict]:
    """Create user in database."""
    return FlextResult[dict].ok({
        "id": "user_123",
        "username": user.username,
        "email": user.email,
    })

# Use the railway
result = (
    FlextResult.ok({"email": "user@example.com", "password": "secure123", "username": "john"})
    .flat_map(validate_registration)
    .flat_map(check_username_available)
    .flat_map(create_user)
)

if result.is_success:
    print(f"✅ User created: {result.value}")
else:
    print(f"❌ Registration failed: {result.error}")
    if result.error_code:
        print(f"   Error code: {result.error_code}")
```

### Pattern 2: API Call with Fallbacks

```python
from flext_core import FlextResult
from typing import Any

def fetch_primary_data(key: str) -> FlextResult[dict]:
    """Try fetching from primary API."""
    # Simulate API call that might fail
    if key == "missing":
        return FlextResult[dict].fail(
            "Primary API: key not found",
            error_code="NOT_FOUND",
        )
    return FlextResult[dict].ok({"source": "primary", "value": key})

def fetch_backup_data(key: str) -> FlextResult[dict]:
    """Fallback to backup API."""
    return FlextResult[dict].ok({
        "source": "backup",
        "value": f"{key}_backup",
    })

def fetch_data_with_fallback(key: str) -> FlextResult[dict]:
    """Fetch data with fallback to backup API."""
    primary = fetch_primary_data(key)

    # If primary succeeds, use it
    if primary.is_success:
        return primary

    # If primary fails, try backup
    return fetch_backup_data(key)

# Usage
result = fetch_data_with_fallback("config")
print(result.value)  # {"source": "primary", "value": "config"}

result = fetch_data_with_fallback("missing")
print(result.value)  # {"source": "backup", "value": "missing_backup"}
```

### Pattern 3: Database Transaction

```python
from flext_core import FlextResult
from decimal import Decimal

def validate_transaction(amount: Decimal, account_id: str) -> FlextResult[dict]:
    """Validate transaction parameters."""
    if amount <= 0:
        return FlextResult[dict].fail(
            "Amount must be positive",
            error_code="INVALID_AMOUNT",
        )
    return FlextResult[dict].ok({"amount": amount, "account_id": account_id})

def check_balance(tx_data: dict) -> FlextResult[dict]:
    """Check if account has sufficient balance."""
    # Simulate database check
    balance = Decimal("1000")
    amount = tx_data["amount"]

    if amount > balance:
        return FlextResult[dict].fail(
            f"Insufficient balance: {amount} > {balance}",
            error_code="INSUFFICIENT_FUNDS",
            error_data={"balance": float(balance), "requested": float(amount)},
        )

    tx_data["balance_before"] = balance
    return FlextResult[dict].ok(tx_data)

def execute_transaction(tx_data: dict) -> FlextResult[dict]:
    """Execute the transaction."""
    # Simulate database update
    return FlextResult[dict].ok({
        **tx_data,
        "transaction_id": "txn_abc123",
        "status": "completed",
    })

# Railway pattern ensures validation before execution
result = (
    FlextResult.ok({"amount": Decimal("500"), "account_id": "acc_001"})
    .flat_map(validate_transaction)
    .flat_map(check_balance)
    .flat_map(execute_transaction)
)

if result.is_success:
    tx = result.value
    print(f"✅ Transaction {tx['transaction_id']} completed")
else:
    print(f"❌ Transaction failed: {result.error}")
```

### Pattern 4: Configuration Loading

```python
from flext_core import FlextResult
import os
import json

def load_config_file(path: str) -> FlextResult[dict]:
    """Load and parse JSON config file."""
    try:
        if not os.path.exists(path):
            return FlextResult[dict].fail(
                f"Config file not found: {path}",
                error_code="CONFIG_NOT_FOUND",
            )

        with open(path) as f:
            config = json.load(f)

        return FlextResult[dict].ok(config)
    except json.JSONDecodeError as e:
        return FlextResult[dict].fail(
            f"Invalid JSON in config: {e}",
            error_code="CONFIG_PARSE_ERROR",
        )
    except Exception as e:
        return FlextResult[dict].fail(
            f"Unexpected error loading config: {e}",
            error_code="CONFIG_ERROR",
        )

def validate_config_schema(config: dict) -> FlextResult[dict]:
    """Validate required config fields."""
    required = ["database_url", "api_key", "log_level"]
    missing = [field for field in required if field not in config]

    if missing:
        return FlextResult[dict].fail(
            f"Config missing required fields: {', '.join(missing)}",
            error_code="CONFIG_VALIDATION_ERROR",
            error_data={"missing_fields": missing},
        )

    return FlextResult[dict].ok(config)

def apply_defaults(config: dict) -> FlextResult[dict]:
    """Apply default values for optional fields."""
    defaults = {
        "debug": False,
        "timeout": 30,
        "retry_count": 3,
    }

    for key, value in defaults.items():
        config.setdefault(key, value)

    return FlextResult[dict].ok(config)

# Load and validate configuration
result = (
    FlextResult.ok("config.json")
    .flat_map(load_config_file)
    .flat_map(validate_config_schema)
    .flat_map(apply_defaults)
)

if result.is_success:
    print("✅ Configuration loaded successfully")
    config = result.value
else:
    print(f"❌ Configuration failed: {result.error}")
    print(f"   Error code: {result.error_code}")
    print(f"   Details: {result.error_data}")
```

## Advanced Techniques

### Combining Multiple Results

```python
from flext_core import FlextResult
from typing import Sequence

# Scenario: Process multiple items, fail on first error
def validate_all_items(items: list[str]) -> FlextResult[list[str]]:
    """Validate each item, short-circuit on first failure."""
    return FlextResult.traverse(
        items,
        lambda item: (
            FlextResult[str].ok(item)
            if len(item) > 3
            else FlextResult[str].fail(f"Item too short: {item}")
        ),
    )

result = validate_all_items(["valid", "item", "too"])  # Short "too" causes failure
assert result.is_failure

result = validate_all_items(["valid", "item", "check"])  # All valid
assert result.is_success
assert result.value == ["valid", "item", "check"]
```

### Error Recovery

```python
from flext_core import FlextResult

def parse_int(value: str) -> FlextResult[int]:
    """Parse string to int."""
    try:
        return FlextResult[int].ok(int(value))
    except ValueError:
        return FlextResult[int].fail(f"Cannot parse as int: {value}")

# Recover from error with alternative
result = (
    FlextResult.ok("not-a-number")
    .flat_map(parse_int)
    .lash(lambda error: FlextResult[int].ok(0))  # Use 0 as default
)

assert result.is_success
assert result.value == 0  # Recovered with default
```

### Resource Management

```python
from flext_core import FlextResult

def create_connection():
    """Factory function creating resource."""
    return {"type": "connection", "id": 123}

def close_connection(conn):
    """Cleanup function."""
    print(f"Closing connection {conn['id']}")

def use_connection(conn) -> FlextResult[dict]:
    """Use the connection."""
    if conn["id"] == 123:
        return FlextResult[dict].ok({"result": "success", "id": conn["id"]})
    return FlextResult[dict].fail("Connection failed")

# Automatic resource cleanup
result = FlextResult.with_resource(
    create_connection,
    use_connection,
    close_connection,
)

if result.is_success:
    print(f"✅ {result.value}")
```

## Best Practices

### 1. Always Return FlextResult from Operations

```python
# ❌ WRONG - Using exceptions
def process_user(data: dict):
    if "email" not in data:
        raise ValueError("Email required")  # Exception-based
    return {"user": data}

# ✅ CORRECT - Railway pattern
def process_user(data: dict) -> FlextResult[dict]:
    if "email" not in data:
        return FlextResult[dict].fail("Email required")  # Railway pattern
    return FlextResult[dict].ok({"user": data})
```

### 2. Include Error Context

```python
# ❌ SPARSE - Missing context
return FlextResult[dict].fail("Failed")

# ✅ RICH - Full context for debugging
return FlextResult[dict].fail(
    "Failed to fetch user data",
    error_code="USER_NOT_FOUND",
    error_data={"user_id": user_id, "database": "primary"},
)
```

### 3. Use Monadic Composition

```python
# ❌ IMPERATIVE - Manual error checking
def old_way(data):
    step1 = validate(data)
    if not step1.is_success:
        return step1

    step2 = process(step1.value)
    if not step2.is_success:
        return step2

    step3 = finalize(step2.value)
    return step3

# ✅ DECLARATIVE - Railway composition
def new_way(data):
    return (
        FlextResult.ok(data)
        .flat_map(validate)
        .flat_map(process)
        .flat_map(finalize)
    )
```

### 4. Preserve Success Values

```python
# ❌ WRONG - Forgetting to return value
result = (
    FlextResult.ok(data)
    .flat_map(transform)  # If returns FlextResult.ok(...), value is preserved
)

# ✅ CORRECT - Functions return FlextResult with wrapped value
def transform(data) -> FlextResult[dict]:
    result = process(data)
    return FlextResult[dict].ok(result)  # Wrap in FlextResult
```

## Backward Compatibility APIs

`FlextResult` maintains two API access methods for backward compatibility:

```python
from flext_core import FlextResult

result = FlextResult[str].ok("value")

# Both .value and .data access the same wrapped value
assert result.value == "value"
assert result.data == "value"  # Backward compatible alias

# Both raise ValidationError on failure
failed = FlextResult[str].fail("Error")
try:
    value = failed.value  # Raises
except Exception:
    pass

try:
    value = failed.data   # Also raises
except Exception:
    pass
```

## Key Takeaways

1. **Railway Pattern**: Errors automatically short-circuit the pipeline
2. **Type-Safe**: Generic `FlextResult[T]` ensures type safety through composition
3. **Composable**: Chain operations with `flat_map`, transform with `map`
4. **Error Context**: Include error codes and metadata for observability
5. **No Exceptions**: Business logic errors use FlextResult, not exceptions
6. **Foundation Pattern**: Used throughout 32+ FLEXT ecosystem projects

## See Also

- [Dependency Injection with FlextContainer](./dependency-injection.md)
- [Error Handling Best Practices](./error-handling.md)
- [API Reference: FlextResult](../api-reference/foundation.md#flextresult)
- **FLEXT CLAUDE.md**: Architecture principles and development workflow

---

**Example from FLEXT Ecosystem**: See `src/flext_tests/test_result.py` for 250+ test cases demonstrating all FlextResult patterns and edge cases.

