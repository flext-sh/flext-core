# FLEXT Exceptions API Guide

## ✅ CORRECT: FlextExceptions Usage Patterns

### API Overview

FlextExceptions provides a unified exception system with modern and legacy APIs:

1. **Modern API** (without Flext prefix) - PRIMARY, use this for all new code
2. **Legacy API** (with Flext prefix) - Backward compatibility only

### 🟢 CORRECT Usage Patterns

#### Modern API (PRIMARY - Use This)
```python
# ✅ CORRECT - Modern API for both raise and except
raise FlextExceptions.ValidationError("Invalid input")
raise FlextExceptions.ConfigurationError("Missing key")
raise FlextExceptions.TimeoutError("Operation timeout")

try:
    risky_operation()
except FlextExceptions.ValidationError as e:
    return FlextResult.fail(str(e))
except FlextExceptions.ConfigurationError as e:
    return FlextResult.fail(str(e))
```

#### For Dynamic Except Clauses - Use _get_exception_class()
```python
# ✅ CORRECT - Dynamic class access for except clauses
try:
    risky_operation()
except FlextExceptions._get_exception_class("FlextValidationError") as e:
    return FlextResult.fail(str(e))
```

### 🔴 DEPRECATED Patterns

#### Legacy API (Still works but not recommended)
```python
# ⚠️ DEPRECATED - Use modern API instead
raise FlextExceptions.ValidationError("Invalid input")  # Works but use modern
except FlextExceptions.ValidationError as e:  # Works but use modern
```

### Technical Explanation

- `FlextExceptions.ValidationError` → **Exception class** (modern API, use this)
- `FlextExceptions.ValidationError` → **Exception class** (legacy API, same functionality)

### Migration Guidelines

When refactoring existing code:

1. **Update raises**: `FlextExceptions.XxxError()` → `FlextExceptions.XxxError()`
2. **Update except**: `FlextExceptions.XxxError` → `FlextExceptions.XxxError`
3. **For new code**: Always use modern API (without Flext prefix)

### Available Exception Types

| Modern API (PRIMARY) | Legacy API (Backward Compat) | Purpose |
|-------------------|-------------------|---------|
| `ValidationError` | `FlextValidationError` | Data validation failures |
| `ConfigurationError` | `FlextConfigurationError` | Configuration issues |
| `ConnectionError` | `FlextConnectionError` | Network/connection failures |
| `AuthenticationError` | `FlextAuthenticationError` | Auth failures |
| `PermissionError` | `FlextPermissionError` | Authorization failures |
| `OperationError` | `FlextOperationError` | General operation failures |
| `ProcessingError` | `FlextProcessingError` | Data processing failures |
| `TimeoutError` | `FlextTimeoutError` | Timeout failures |
| `NotFoundError` | `FlextNotFoundError` | Resource not found |
| `AlreadyExistsError` | `FlextAlreadyExistsError` | Resource conflicts |
| `CriticalError` | `FlextCriticalError` | Critical system failures |
| `TypeError` | `FlextTypeError` | Type validation failures |

### Complete Example

```python
from flext_core import FlextExceptions, FlextResult

class UserService:
    def create_user(self, data: dict) -> FlextResult[User]:
        try:
            # Validate data
            if not data.get("email"):
                # ✅ CORRECT: Use modern API
                raise FlextExceptions.ValidationError(
                    "Email required", 
                    field="email"
                )
            
            # Create user logic here...
            return FlextResult.ok(user)
            
        # ✅ CORRECT: Use modern API in except too    
        except FlextExceptions.ValidationError as e:
            return FlextResult.fail(str(e))
        except FlextExceptions.OperationError as e:
            return FlextResult.fail(str(e))
```

### Quality Gates

- **Pyright**: 0 errors when using correct patterns
- **Runtime**: Perfect execution with both APIs
- **ABI**: 100% backward compatibility maintained

---

**Remember**: Modern API (without Flext prefix) for **everything**!
