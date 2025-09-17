# Troubleshooting Guide

**FLEXT-Core Foundation Library Troubleshooting**
**Date**: September 17, 2025 | **Version**: 0.9.0

---

## Overview

This guide covers common issues, solutions, and debugging strategies when working with FLEXT-Core foundation patterns. All solutions are verified against the current implementation.

---

## Import Issues

### Problem: Cannot Import FLEXT-Core Modules

**Symptoms**:
```bash
ImportError: No module named 'flext_core'
ModuleNotFoundError: No module named 'flext_core'
```

**Solutions**:

1. **Check PYTHONPATH**:
   ```bash
   # Add src directory to Python path
   export PYTHONPATH=src:$PYTHONPATH
   python -c "from flext_core import FlextResult; print('✓ Import successful')"
   ```

2. **Verify Installation**:
   ```bash
   # For development installation
   make setup

   # Verify installation
   python -c "import sys; print('\\n'.join(sys.path))"
   ```

3. **Check Directory Structure**:
   ```bash
   # Ensure you're in the correct directory
   ls src/flext_core/__init__.py  # Should exist

   # Check current working directory
   pwd  # Should be in flext-core project root
   ```

### Problem: Internal Import Errors

**Symptoms**:
```bash
ImportError: attempted relative import with no known parent package
```

**Solution**:
```python
# ✅ CORRECT - Use root module imports
from flext_core import FlextResult, FlextContainer, FlextModels

# ❌ AVOID - Internal module imports
from flext_core.result import FlextResult  # Don't do this
```

---

## FlextResult Issues

### Problem: AttributeError on FlextResult

**Symptoms**:
```python
AttributeError: 'FlextResult' object has no attribute 'data'
AttributeError: 'FlextResult' object has no attribute 'value'
```

**Diagnosis**:
```python
# Check FlextResult API availability
import sys
sys.path.insert(0, 'src')
from flext_core import FlextResult

result = FlextResult[str].ok("test")
print("Available attributes:", [attr for attr in dir(result) if not attr.startswith('_')])
print("Has .data:", hasattr(result, 'data'))
print("Has .value:", hasattr(result, 'value'))
print("Has .unwrap:", hasattr(result, 'unwrap'))
```

**Solutions**:

1. **Use Proper Import**:
   ```python
   # Ensure you're importing from the correct module
   from flext_core import FlextResult

   # Not from internal modules
   ```

2. **Check API Compatibility**:
   ```python
   result = FlextResult[str].ok("test")

   # Both APIs should work
   assert hasattr(result, 'data'), "Legacy .data API missing"
   assert hasattr(result, 'value'), "New .value API missing"
   ```

### Problem: FlextResult Type Errors

**Symptoms**:
```bash
TypeError: FlextResult.__init__() missing required arguments
```

**Solution**:
```python
# ✅ CORRECT - Use factory methods
success = FlextResult[str].ok("value")
failure = FlextResult[str].fail("error message")

# ❌ AVOID - Direct constructor
result = FlextResult(True, "value", None)  # Don't do this
```

### Problem: Railway Composition Not Working

**Symptoms**:
```python
AttributeError: 'FlextResult' object has no attribute 'flat_map'
AttributeError: 'FlextResult' object has no attribute 'map'
```

**Diagnosis and Solution**:
```python
from flext_core import FlextResult

# Test railway methods availability
result = FlextResult[int].ok(5)
methods = ['map', 'flat_map', 'map_error', 'filter', 'unwrap', 'unwrap_or']

for method in methods:
    has_method = hasattr(result, method)
    print(f"{method}: {'✓' if has_method else '✗'}")

# Example working composition
composed = (
    FlextResult[int].ok(10)
    .map(lambda x: x * 2)
    .flat_map(lambda x: FlextResult[int].ok(x + 5))
    .filter(lambda x: x > 20, "Too small")
)

print(f"Composition result: {composed.unwrap() if composed.is_success else composed.error}")
```

---

## FlextContainer Issues

### Problem: Container Service Not Found

**Symptoms**:
```python
# Container returns failure for registered service
service_result = container.get("my_service")
assert service_result.is_failure  # "Service 'my_service' not found"
```

**Diagnosis**:
```python
from flext_core import FlextContainer

container = FlextContainer.get_global()

# Check registered services
services_result = container.list_services()
if services_result.is_success:
    services = services_result.unwrap()
    print("Registered services:", list(services.keys()))
else:
    print("Failed to list services:", services_result.error)

# Check specific service
service_name = "my_service"
service_result = container.get(service_name)
print(f"Service '{service_name}' found: {service_result.is_success}")
```

**Solutions**:

1. **Verify Registration**:
   ```python
   # Ensure service was registered successfully
   service = MyService()
   registration_result = container.register("my_service", service)

   if registration_result.is_failure:
       print(f"Registration failed: {registration_result.error}")
   ```

2. **Check Service Name**:
   ```python
   # Service names are case-sensitive and exact match
   container.register("database", db_service)

   # ✅ Correct
   db_result = container.get("database")

   # ❌ Wrong
   db_result = container.get("Database")  # Case mismatch
   db_result = container.get("db")        # Name mismatch
   ```

### Problem: Container Singleton Issues

**Symptoms**:
```python
# Different container instances
container1 = FlextContainer.get_global()
container2 = FlextContainer.get_global()
assert container1 is not container2  # Singleton not working
```

**Solution**:
```python
from flext_core import FlextContainer

# Test singleton behavior
container1 = FlextContainer.get_global()
container2 = FlextContainer.get_global()

print(f"Same instance: {container1 is container2}")
print(f"Container1 ID: {id(container1)}")
print(f"Container2 ID: {id(container2)}")

# If singleton is broken, check for multiple imports or class redefinition
```

---

## Domain Model Issues

### Problem: FlextModels Not Available

**Symptoms**:
```python
AttributeError: module 'flext_core' has no attribute 'FlextModels'
```

**Diagnosis**:
```python
import sys
sys.path.insert(0, 'src')
from flext_core import *

# Check what's available
available = [name for name in dir() if name.startswith('Flext')]
print("Available Flext classes:", available)

# Specifically check FlextModels
try:
    from flext_core import FlextModels
    print("✓ FlextModels available")
    print("FlextModels attributes:", [attr for attr in dir(FlextModels) if not attr.startswith('_')])
except ImportError as e:
    print("✗ FlextModels not available:", e)
```

**Solution**:
```python
# Check if models module exists and is exported
from flext_core import FlextModels

# Use domain model classes
class User(FlextModels.Entity):
    name: str
    email: str

# If FlextModels is not available, check __init__.py exports
```

### Problem: Domain Events Not Working

**Symptoms**:
```python
AttributeError: 'User' object has no attribute 'add_domain_event'
```

**Solution**:
```python
from flext_core import FlextModels

class User(FlextModels.Entity):  # Must inherit from Entity
    name: str

    def change_name(self, new_name: str):
        self.name = new_name
        # This should work if properly inheriting from Entity
        self.add_domain_event("NameChanged", {"old_name": self.name, "new_name": new_name})

# Test domain events
user = User(name="John")
user.change_name("Jane")

# Check events were recorded
if hasattr(user, '_domain_events'):
    print(f"Domain events: {len(user._domain_events)}")
else:
    print("Domain events not available - check Entity inheritance")
```

---

## Configuration Issues

### Problem: FlextConfig Environment Variables Not Loading

**Symptoms**:
```python
# Environment variables not being loaded
class AppConfig(FlextConfig):
    database_url: str
    debug: bool = False

config = AppConfig()  # database_url not loaded from environment
```

**Diagnosis**:
```bash
# Check environment variables
env | grep -E "(DATABASE_URL|DEBUG)"

# Check .env file
cat .env

# Test environment variable loading
python -c "
import os
print('DATABASE_URL:', os.getenv('DATABASE_URL'))
print('DEBUG:', os.getenv('DEBUG'))
"
```

**Solutions**:

1. **Check Environment Variable Names**:
   ```python
   class AppConfig(FlextConfig):
       database_url: str  # Looks for DATABASE_URL
       debug: bool = False  # Looks for DEBUG

       class Config:
           env_file = ".env"
           case_sensitive = False
   ```

2. **Use Env Prefix**:
   ```python
   class AppConfig(FlextConfig):
       database_url: str  # Looks for APP_DATABASE_URL

       class Config:
           env_prefix = "APP_"
   ```

3. **Explicit Field Configuration**:
   ```python
   from pydantic import Field

   class AppConfig(FlextConfig):
       database_url: str = Field(env="DB_URL")  # Custom env var name
   ```

---

## Testing Issues

### Problem: Container State Pollution Between Tests

**Symptoms**:
```python
# Tests fail when run together but pass individually
def test_service_a():
    container = FlextContainer.get_global()
    container.register("service", ServiceA())
    # Test passes

def test_service_b():
    container = FlextContainer.get_global()
    container.register("service", ServiceB())  # Fails - already registered
    # Test fails
```

**Solution**:
```python
import pytest
from flext_core import FlextContainer

@pytest.fixture
def clean_container():
    """Provide clean container for each test."""
    container = FlextContainer.get_global()
    container.clear()  # Clear before test
    yield container
    container.clear()  # Clear after test

def test_service_with_clean_container(clean_container):
    # Use clean_container instead of getting global directly
    clean_container.register("service", MyService())
    # Test will be isolated
```

### Problem: FlextResult Testing Patterns

**Symptoms**:
```python
# Verbose test assertions
result = my_function()
assert result.is_success == True
assert result.error is None
assert result.value == expected_value
```

**Solution**:
```python
# Create test helpers
def assert_success(result, expected_value=None):
    """Assert result is successful with optional value check."""
    assert result.is_success, f"Expected success but got error: {result.error}"
    if expected_value is not None:
        assert result.unwrap() == expected_value

def assert_failure(result, expected_error=None):
    """Assert result is failure with optional error check."""
    assert result.is_failure, f"Expected failure but got success: {result.unwrap()}"
    if expected_error is not None:
        assert expected_error in result.error

# Use in tests
def test_my_function():
    success_result = my_function("valid_input")
    assert_success(success_result, "expected_output")

    failure_result = my_function("invalid_input")
    assert_failure(failure_result, "validation")
```

---

## Performance Issues

### Problem: Slow FlextResult Operations

**Symptoms**:
```python
# Long chains of operations are slow
result = (
    initial_result
    .map(transform1)
    .flat_map(transform2)
    .map(transform3)
    # ... many more operations
)
```

**Diagnosis**:
```python
import time
from flext_core import FlextResult

# Benchmark operations
def benchmark_operations():
    start = time.time()

    result = FlextResult[int].ok(1)
    for i in range(10000):
        result = result.map(lambda x: x + 1)

    end = time.time()
    print(f"10,000 map operations: {end - start:.4f} seconds")
    print(f"Final value: {result.unwrap()}")

benchmark_operations()
```

**Solutions**:

1. **Batch Operations**:
   ```python
   # Instead of many small maps
   result = (
       initial_result
       .map(lambda x: transform3(transform2(transform1(x))))  # Combine transforms
   )
   ```

2. **Early Success Checks**:
   ```python
   # Check success early in long chains
   result = initial_operation()
   if result.is_failure:
       return result

   # Continue with expensive operations only if successful
   return expensive_chain(result)
   ```

---

## Type Checking Issues

### Problem: MyPy Errors with FlextResult

**Symptoms**:
```bash
error: Cannot infer type argument 1 of "FlextResult"
error: Incompatible return value type
```

**Solutions**:

1. **Explicit Type Parameters**:
   ```python
   # ✅ Explicit types
   def process_data(data: str) -> FlextResult[dict]:
       if not data:
           return FlextResult[dict].fail("Data required")
       return FlextResult[dict].ok({"processed": data})
   ```

2. **Type Annotations for Complex Cases**:
   ```python
   from typing import Union, Dict

   def complex_operation(data: Dict[str, object]) -> FlextResult[Union[str, int]]:
       # Complex logic
       return FlextResult[Union[str, int]].ok(result)
   ```

### Problem: Generic Type Issues

**Symptoms**:
```bash
error: "FlextResult" is not subscriptable
```

**Solution**:
```python
# For Python < 3.9, use typing imports
from typing import Generic, TypeVar
from flext_core import FlextResult

T = TypeVar('T')

# For Python 3.9+, built-in generics work
def modern_function(data: str) -> FlextResult[str]:
    return FlextResult[str].ok(data.upper())
```

---

## Debugging Tools

### FlextResult Debugging

```python
def debug_result(result, operation_name="Operation"):
    """Debug helper for FlextResult objects."""
    print(f"\n=== {operation_name} Debug ===")
    print(f"Success: {result.is_success}")
    print(f"Failure: {result.is_failure}")

    if result.is_success:
        print(f"Value: {result.unwrap()}")
        print(f"Value type: {type(result.unwrap())}")
    else:
        print(f"Error: {result.error}")

    # Check API compatibility
    print(f"Has .data: {hasattr(result, 'data')}")
    print(f"Has .value: {hasattr(result, 'value')}")
    print("=" * 30)

# Usage
result = my_operation()
debug_result(result, "My Operation")
```

### Container Debugging

```python
def debug_container():
    """Debug helper for FlextContainer state."""
    from flext_core import FlextContainer

    container = FlextContainer.get_global()

    print("\n=== Container Debug ===")
    print(f"Container ID: {id(container)}")

    services_result = container.list_services()
    if services_result.is_success:
        services = services_result.unwrap()
        print(f"Registered services ({len(services)}):")
        for name, service in services.items():
            print(f"  - {name}: {type(service).__name__}")
    else:
        print(f"Failed to list services: {services_result.error}")

    print("=" * 25)

# Usage
debug_container()
```

### Environment Debugging

```bash
#!/bin/bash
# environment_debug.sh

echo "=== Environment Debug ==="
echo "Python version: $(python --version)"
echo "Current directory: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"

echo -e "\n=== FLEXT-Core Import Test ==="
python -c "
import sys
sys.path.insert(0, 'src')
try:
    from flext_core import FlextResult, FlextContainer, FlextModels
    print('✓ All imports successful')
except Exception as e:
    print(f'✗ Import failed: {e}')
"

echo -e "\n=== File Structure Check ==="
ls -la src/flext_core/__init__.py 2>/dev/null && echo "✓ __init__.py exists" || echo "✗ __init__.py missing"
ls -la src/flext_core/result.py 2>/dev/null && echo "✓ result.py exists" || echo "✗ result.py missing"

echo "=========================="
```

---

## Getting Help

### Internal Debugging

1. **Check Implementation Status**:
   ```bash
   # Run tests to see current status
   PYTHONPATH=src pytest tests/ --tb=short -v

   # Check coverage
   PYTHONPATH=src pytest tests/ --cov=src --cov-report=term
   ```

2. **Validate Environment**:
   ```bash
   # Run quality checks
   make validate

   # Check type annotations
   PYTHONPATH=src mypy src/ --strict --show-error-codes
   ```

### External Resources

- **Project Documentation**: [docs/](.) - Foundation-specific documentation
- **Workspace Documentation**: [FLEXT Workspace Docs](../../docs/) - Ecosystem-wide patterns
- **Examples**: [examples/](../examples/) - Working code examples (22 verified examples)
- **Source Code**: [src/flext_core/](../src/flext_core/) - Implementation details

### Reporting Issues

When reporting issues, include:

1. **Environment Information**:
   ```bash
   python --version
   pwd
   echo $PYTHONPATH
   ```

2. **Error Details**:
   - Full error message and stack trace
   - Minimal reproduction case
   - Expected vs actual behavior

3. **Context**:
   - FLEXT-Core version (0.9.0)
   - Operating system
   - Related configuration

---

**Troubleshooting Authority**: Verified solutions for FLEXT-Core v0.9.0
**Implementation Verified**: All solutions tested against current codebase
**Updated**: September 17, 2025
