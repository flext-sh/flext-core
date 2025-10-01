# Migration Guide: FLEXT-Core 0.x → 1.0

**Version**: 1.0.0 | **Date**: 2025-10-01 | **Status**: DRAFT

---

## Overview

This guide helps you migrate from FLEXT-Core 0.x (particularly 0.9.9) to the stable 1.0.0 release.

**Good News**: The 1.0.0 release is **100% backward compatible** with 0.9.9. No breaking changes!

---

## Quick Start

### For Most Projects: Zero Changes Required

If you're using FLEXT-Core 0.9.9, upgrading to 1.0.0 requires **NO code changes**:

```bash
# Update dependency in pyproject.toml or requirements.txt
pip install --upgrade flext-core>=1.0.0,<2.0.0

# Or with Poetry
poetry add "flext-core>=1.0.0,<2.0.0"

# Run your tests to verify
pytest tests/
```

### What's Changed in 1.0.0?

**ZERO Breaking Changes** - All 0.9.9 APIs work identically in 1.0.0

**What's New**:
- ✅ **Stability Guarantees**: API stability documented and guaranteed for 1.x series
- ✅ **Dependency Locks**: Version bounds prevent breaking changes from dependencies
- ✅ **Semantic Versioning**: Formal SemVer 2.0.0 commitment with deprecation policy
- ✅ **HTTP Primitives**: New HTTP constants and models (backward compatible addition)

---

## Compatibility Matrix

| Component | 0.9.9 API | 1.0.0 API | Migration Required |
|-----------|-----------|-----------|-------------------|
| FlextResult[T] | `.value`, `.data` | `.value`, `.data` (both guaranteed) | ❌ None |
| FlextContainer | `.get_global()` | `.get_global()` | ❌ None |
| FlextModels | Entity/Value/Aggregate | Entity/Value/Aggregate | ❌ None |
| FlextService | Base class | Base class | ❌ None |
| FlextLogger | `__name__` constructor | `__name__` constructor | ❌ None |
| FlextConfig | `.get_global_instance()` | `.get_global_instance()` | ❌ None |
| FlextBus | Event bus API | Event bus API | ❌ None |
| FlextCqrs | CQRS patterns | CQRS patterns | ❌ None |
| HTTP Constants | ❌ Not available | ✅ FlextConstants.Http | ✅ Optional enhancement |
| HTTP Models | ❌ Not available | ✅ HttpRequest/HttpResponse | ✅ Optional enhancement |

---

## Detailed Migration Steps

### Step 1: Update Dependencies

#### Poetry Projects

```toml
[tool.poetry.dependencies]
python = ">=3.13,<3.14"
flext-core = ">=1.0.0,<2.0.0"  # Lock to 1.x series
```

```bash
poetry update flext-core
poetry install
```

#### pip Projects

```txt
# requirements.txt
flext-core>=1.0.0,<2.0.0
```

```bash
pip install --upgrade -r requirements.txt
```

### Step 2: Verify Your Tests

```bash
# Run your full test suite
pytest tests/

# Run with coverage to identify affected code
pytest --cov=src tests/

# Run static type checking
mypy src/
pyright src/
```

### Step 3: Update Dependency Version Bounds (Recommended)

To benefit from flext-core's ABI stability guarantees, update your project's dependencies:

```toml
# Before (0.9.9 style)
dependencies = [
    "flext-core>=0.9.9",
    "pydantic>=2.11.7",
]

# After (1.0.0 style - recommended)
dependencies = [
    "flext-core>=1.0.0,<2.0.0",  # Lock to 1.x series
    "pydantic>=2.11.7,<3.0.0",   # Match flext-core bounds
]
```

---

## New Features You Can Adopt (Optional)

### 1. HTTP Primitives (New in 0.9.9, Stable in 1.0.0)

**Before (custom HTTP constants)**:

```python
# Custom constants in your code
HTTP_OK = 200
HTTP_CREATED = 201
HTTP_BAD_REQUEST = 400
HTTP_NOT_FOUND = 404
```

**After (use FlextConstants.Http)**:

```python
from flext_core import FlextConstants

# Use standardized HTTP constants
FlextConstants.Http.HTTP_OK                      # 200
FlextConstants.Http.HTTP_CREATED                 # 201
FlextConstants.Http.HTTP_BAD_REQUEST             # 400
FlextConstants.Http.HTTP_NOT_FOUND               # 404
FlextConstants.Http.HTTP_INTERNAL_SERVER_ERROR   # 500

# HTTP methods
FlextConstants.Http.Method.GET
FlextConstants.Http.Method.POST
FlextConstants.Http.Method.PUT
FlextConstants.Http.Method.DELETE

# Content types
FlextConstants.Http.ContentType.JSON
FlextConstants.Http.ContentType.XML
FlextConstants.Http.ContentType.FORM
```

### 2. HTTP Request/Response Models

**Before (custom models)**:

```python
from pydantic import BaseModel

class MyHttpRequest(BaseModel):
    url: str
    method: str
    headers: dict[str, str] = {}
    body: str | None = None
```

**After (use FlextModels.HttpRequest)**:

```python
from flext_core import FlextModels

class MyHttpRequest(FlextModels.HttpRequest):
    """Extend HTTP request base with additional fields."""
    # Inherited: url, method, headers, body, timeout
    # Computed: is_secure, has_body

    custom_field: str = ""  # Add your fields

class MyHttpResponse(FlextModels.HttpResponse):
    """Extend HTTP response base."""
    # Inherited: status_code, headers, body, elapsed_time
    # Computed: is_success, is_client_error, is_server_error

    custom_data: dict = {}  # Add your fields
```

---

## Deprecation Notices

**NONE for 1.0.0 Release** - All 0.9.9 APIs remain fully supported.

Future deprecations will follow this policy:

1. **Version N**: Feature deprecated with `DeprecationWarning`
2. **Version N+1**: Feature still works, warning continues
3. **Version N+2**: Feature may be removed (with alternative provided)

Minimum deprecation cycle: **2 minor versions** (e.g., 1.0 → 1.1 → 1.2)

---

## Breaking Changes (None in 1.0.0)

**Guaranteed for 1.x Series**: ZERO breaking changes

All APIs from 0.9.9 remain stable and supported throughout the entire 1.x release series.

---

## Testing Your Migration

### Comprehensive Test Script

```bash
#!/bin/bash
echo "=== FLEXT-Core 1.0.0 Migration Validation ==="

# 1. Install 1.0.0
echo "Installing flext-core 1.0.0..."
pip install --upgrade "flext-core>=1.0.0,<2.0.0"

# 2. Verify version
python -c "
from flext_core import __version__
print(f'✅ FLEXT-Core {__version__} installed')
assert __version__.startswith('1.'), f'Expected 1.x, got {__version__}'
"

# 3. Test core APIs
python -c "
from flext_core import (
    FlextResult, FlextContainer, FlextModels,
    FlextService, FlextLogger, FlextConfig,
    FlextBus, FlextCqrs, FlextConstants
)

# Test FlextResult - railway pattern
result = FlextResult[str].ok('test')
assert result.is_success
assert result.value == 'test'
assert result.data == 'test'  # Dual access guaranteed

# Test FlextContainer - dependency injection
container = FlextContainer.get_global()
assert container is not None

# Test FlextModels - DDD patterns
class TestEntity(FlextModels.Entity):
    name: str

entity = TestEntity(id='test', name='Test')
assert entity.id == 'test'

# Test FlextLogger
logger = FlextLogger(__name__)
logger.info('Migration test successful')

# Test HTTP primitives (new in 1.0.0)
assert FlextConstants.Http.HTTP_OK == 200
assert FlextConstants.Http.Method.GET == 'GET'

print('✅ All core APIs working correctly')
"

# 4. Run your test suite
echo "Running test suite..."
pytest tests/ -v

# 5. Run type checking
echo "Running type checking..."
mypy src/ --strict
pyright src/

# 6. Run linting
echo "Running linting..."
ruff check src/

echo "✅ Migration validation complete!"
```

### Quick Smoke Test

```python
#!/usr/bin/env python3
"""Quick smoke test for FLEXT-Core 1.0.0 migration."""

from flext_core import (
    FlextResult,
    FlextContainer,
    FlextModels,
    FlextService,
    FlextLogger,
    FlextConstants,
)


def test_railway_pattern() -> None:
    """Test FlextResult railway pattern."""
    result = FlextResult[str].ok("success")
    assert result.is_success
    assert result.value == "success"
    assert result.data == "success"  # Dual access
    print("✅ Railway pattern working")


def test_dependency_injection() -> None:
    """Test FlextContainer DI."""
    container = FlextContainer.get_global()
    container.register("test_service", "test_value")
    service = container.get("test_service")
    assert service.is_success
    assert service.unwrap() == "test_value"
    print("✅ Dependency injection working")


def test_domain_models() -> None:
    """Test FlextModels DDD patterns."""

    class User(FlextModels.Entity):
        name: str
        email: str

    user = User(id="user_123", name="Alice", email="alice@example.com")
    assert user.id == "user_123"
    assert user.name == "Alice"
    print("✅ Domain models working")


def test_http_primitives() -> None:
    """Test HTTP primitives (new in 1.0.0)."""
    assert FlextConstants.Http.HTTP_OK == 200
    assert FlextConstants.Http.HTTP_CREATED == 201
    assert FlextConstants.Http.Method.GET == "GET"
    assert FlextConstants.Http.ContentType.JSON == "application/json"
    print("✅ HTTP primitives working")


def test_logging() -> None:
    """Test FlextLogger."""
    logger = FlextLogger(__name__)
    logger.info("Test log message", extra={"test": "data"})
    print("✅ Logging working")


if __name__ == "__main__":
    test_railway_pattern()
    test_dependency_injection()
    test_domain_models()
    test_http_primitives()
    test_logging()
    print("\n✅ All smoke tests passed! Migration successful.")
```

---

## Common Migration Scenarios

### Scenario 1: Existing Application Using FlextResult

**Your 0.9.9 Code** (still works in 1.0.0):

```python
from flext_core import FlextResult

def process_user(user_id: str) -> FlextResult[dict]:
    """Process user with railway pattern."""
    if not user_id:
        return FlextResult[dict].fail("User ID required")

    user_data = {"id": user_id, "name": "Alice"}
    return FlextResult[dict].ok(user_data)

# Using the result
result = process_user("user_123")
if result.is_success:
    # Both .value and .data work (guaranteed in 1.x)
    user = result.value  # or result.data
    print(f"User: {user}")
else:
    print(f"Error: {result.error}")
```

**Migration Required**: ❌ NONE - Works identically in 1.0.0

### Scenario 2: Dependency Injection with FlextContainer

**Your 0.9.9 Code** (still works in 1.0.0):

```python
from flext_core import FlextContainer, FlextLogger

# Register services
container = FlextContainer.get_global()
container.register("logger", FlextLogger(__name__))

# Resolve services
logger_result = container.get("logger")
if logger_result.is_success:
    logger = logger_result.unwrap()
    logger.info("Application started")
```

**Migration Required**: ❌ NONE - Works identically in 1.0.0

### Scenario 3: Domain-Driven Design with FlextModels

**Your 0.9.9 Code** (still works in 1.0.0):

```python
from flext_core import FlextModels

class User(FlextModels.Entity):
    """User entity with DDD patterns."""
    name: str
    email: str
    age: int

class Address(FlextModels.Value):
    """Address value object (immutable)."""
    street: str
    city: str
    country: str

# Create instances
user = User(id="user_123", name="Alice", email="alice@example.com", age=30)
address = Address(street="123 Main St", city="Boston", country="USA")
```

**Migration Required**: ❌ NONE - Works identically in 1.0.0

### Scenario 4: Domain Services

**Your 0.9.9 Code** (still works in 1.0.0):

```python
from flext_core import FlextService, FlextResult

class UserService(FlextService):
    """User domain service."""

    def create_user(self, name: str, email: str) -> FlextResult[dict]:
        """Create a new user."""
        if not email or "@" not in email:
            return FlextResult[dict].fail("Invalid email")

        user = {"name": name, "email": email}
        return FlextResult[dict].ok(user)

# Use the service
service = UserService()
result = service.create_user("Bob", "bob@example.com")
```

**Migration Required**: ❌ NONE - Works identically in 1.0.0

### Scenario 5: Adopting HTTP Primitives (Optional Enhancement)

**Before (0.9.9)** - Custom HTTP handling:

```python
# Your custom HTTP constants
HTTP_OK = 200
HTTP_BAD_REQUEST = 400

def handle_request(status_code: int) -> str:
    if status_code == HTTP_OK:
        return "success"
    elif status_code == HTTP_BAD_REQUEST:
        return "error"
    return "unknown"
```

**After (1.0.0)** - Using FlextConstants.Http:

```python
from flext_core import FlextConstants

def handle_request(status_code: int) -> str:
    """Handle HTTP request using standard constants."""
    if status_code == FlextConstants.Http.HTTP_OK:
        return "success"
    elif status_code == FlextConstants.Http.HTTP_BAD_REQUEST:
        return "error"
    return "unknown"

# Or use HTTP response model
from flext_core import FlextModels

class ApiResponse(FlextModels.HttpResponse):
    """API response with computed properties."""
    pass

response = ApiResponse(
    status_code=200,
    headers={"Content-Type": "application/json"},
    body='{"status": "ok"}',
    elapsed_time=0.25
)

if response.is_success:  # Computed property (200-299)
    print("Request succeeded")
```

**Migration Required**: ✅ Optional enhancement (backward compatible)

---

## Troubleshooting

### Issue 1: Version Conflict with Dependencies

**Problem**: Dependency version conflicts after upgrading to 1.0.0

**Solution**: Update your dependencies to match flext-core's locked versions:

```toml
# pyproject.toml
dependencies = [
    "flext-core>=1.0.0,<2.0.0",
    "pydantic>=2.11.7,<3.0.0",      # Match flext-core
    "pydantic-settings>=2.10.1,<3.0.0",  # Match flext-core
]
```

### Issue 2: Type Checking Errors

**Problem**: MyPy or PyRight shows type errors after upgrade

**Solution**: Verify you're using Python 3.13+ and update type stubs:

```bash
# Update type checking tools
pip install --upgrade mypy pyright

# Install type stubs
pip install types-pyyaml types-setuptools

# Run type checking
mypy src/ --python-version 3.13
```

### Issue 3: Tests Failing After Upgrade

**Problem**: Tests pass with 0.9.9 but fail with 1.0.0

**Diagnosis**:

```bash
# Run tests with verbose output
pytest tests/ -vv

# Check for deprecation warnings
pytest tests/ -W default

# Run with coverage to identify affected code
pytest --cov=src --cov-report=term-missing tests/
```

**Solution**: This should not happen (zero breaking changes). If you encounter this:

1. Verify you're actually using 1.0.0:
   ```python
   import flext_core
   print(flext_core.__version__)  # Should be 1.0.x
   ```

2. Check for indirect dependencies that might have changed:
   ```bash
   pip list | grep flext
   ```

3. Report the issue: https://github.com/flext-sh/flext-core/issues

---

## Rollback Plan (If Needed)

If you encounter issues, you can rollback to 0.9.9:

```bash
# Uninstall 1.0.0
pip uninstall flext-core

# Install 0.9.9
pip install "flext-core==0.9.9"

# Or with Poetry
poetry add "flext-core==0.9.9"

# Verify rollback
python -c "from flext_core import __version__; print(__version__)"
```

**Note**: Rollback should not be necessary - 1.0.0 is 100% backward compatible.

---

## Getting Help

### Documentation

- **API Stability Guarantees**: [API_STABILITY.md](API_STABILITY.md)
- **Semantic Versioning**: [VERSIONING.md](VERSIONING.md)
- **Development Guide**: [CLAUDE.md](CLAUDE.md)
- **Architecture**: [docs/architecture.md](docs/architecture.md)

### Support Channels

- **Issues**: [GitHub Issues](https://github.com/flext-sh/flext-core/issues)
- **Discussions**: [GitHub Discussions](https://github.com/flext-sh/flext-core/discussions)
- **Security**: Report vulnerabilities privately to FLEXT maintainers

### Reporting Migration Issues

If you encounter migration problems:

1. **Check this guide** for common scenarios
2. **Search existing issues**: https://github.com/flext-sh/flext-core/issues
3. **Create new issue** with:
   - Your flext-core version (0.9.x and 1.0.x)
   - Python version
   - Minimal reproduction code
   - Error messages and stack traces
   - Expected vs actual behavior

---

## FAQ

### Q1: Do I need to change any code to upgrade to 1.0.0?

**A**: No. The 1.0.0 release is 100% backward compatible with 0.9.9. All your existing code will work without modifications.

### Q2: What's the benefit of upgrading to 1.0.0?

**A**:
- **API Stability**: Guaranteed no breaking changes in 1.x series
- **Dependency Locks**: Protection against breaking changes from dependencies
- **Long-term Support**: Minimum 2 minor version deprecation cycle
- **HTTP Primitives**: New standardized HTTP constants and models (optional)
- **Production Confidence**: Formal stability commitment for enterprise use

### Q3: Can I stay on 0.9.9?

**A**: Yes, 0.9.9 will continue to work. However, we recommend upgrading to benefit from stability guarantees and future enhancements in the 1.x series.

### Q4: How long will 1.0.0 be supported?

**A**: The entire 1.x series will be supported with:
- Security patches for critical issues
- Bug fixes in patch releases (1.0.1, 1.0.2, etc.)
- New features in minor releases (1.1.0, 1.2.0, etc.)
- No breaking changes until 2.0.0 (planned 2026+)

### Q5: What happens if I need a feature that requires breaking changes?

**A**: We will:
1. Add the feature in a backward-compatible way (if possible)
2. Deprecate old API with warnings (minimum 2 minor versions)
3. Provide migration tools and documentation
4. Only remove deprecated features in 2.0.0 (with 6+ months notice)

### Q6: Will my domain library (flext-api, flext-cli, etc.) work with 1.0.0?

**A**: Yes. All FLEXT domain libraries are tested with 1.0.0 before release. Update flext-core first, then test your application.

### Q7: What if I find a breaking change in 1.0.0?

**A**: This would be a critical bug. Please report it immediately:
- GitHub Issues: https://github.com/flext-sh/flext-core/issues
- Label: "stability-guarantee"
- Priority: P0 (hotfix within 48 hours)

---

## Summary

**Migration Complexity**: ⭐ Trivial (0/5 difficulty)

**Time Required**:
- Small projects: < 5 minutes
- Large projects: < 30 minutes
- Ecosystem-wide: < 2 hours

**Code Changes Required**: ❌ NONE for existing functionality

**Testing Effort**: ✅ Run existing test suite (no changes needed)

**Risk Level**: 🟢 Minimal (100% backward compatible)

**Recommended Approach**:
1. Update dependency to `flext-core>=1.0.0,<2.0.0`
2. Run your test suite
3. Deploy with confidence

**Key Takeaway**: The 1.0.0 release is a **stability milestone**, not a breaking change. All your 0.9.9 code works identically in 1.0.0 with added guarantees for the future.

---

**FLEXT-Core 1.0.0** - Stable foundation for the FLEXT ecosystem with guaranteed API stability, locked dependencies, and long-term support commitment.

**Questions?** See [GitHub Discussions](https://github.com/flext-sh/flext-core/discussions)
