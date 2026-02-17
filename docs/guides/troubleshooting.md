# Troubleshooting Guide

**Status**: Production Ready | **Version**: 0.10.0 | **Focus**: Common Issues and Solutions

Systematic troubleshooting guide for common issues in FLEXT-Core applications.

## Import and Setup Issues

### ImportError: cannot import name 'FlextResult'

**Symptom:**

```
ImportError: cannot import name 'FlextResult' from 'flext_core'
```

**Possible Causes:**

1. **FLEXT-Core not installed**
2. **Python version too old** (requires 3.13+)
3. **Wrong import path** (importing from internal module)

**Solutions:**

```bash
# Verify FLEXT-Core is installed
pip list | grep flext-core

# Check Python version (must be 3.13+)
python --version

# Reinstall if needed
pip install --upgrade flext-core
```

**Correct import:**

```python
# ✅ CORRECT
from flext_core import FlextResult

# ❌ WRONG
from flext_core.result import FlextResult  # Don't do this!
```

### ModuleNotFoundError: No module named 'flext_core'

**Symptom:**

```
ModuleNotFoundError: No module named 'flext_core'
```

**Solutions:**

```bash
# Install FLEXT-Core
pip install flext-core

# Or from development environment
pip install -e .

# Check installation
python -c "import flext_core; print(flext_core.__version__)"
```

### PYTHONPATH Issues

**Symptom:**

```
ModuleNotFoundError when running from project root
```

**Solutions:**

```bash
# Set PYTHONPATH before running
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python main.py

# Or run with explicit PYTHONPATH
PYTHONPATH=src python main.py

# For pytest
PYTHONPATH=src pytest tests/
```

## Type Checking Issues

### MyPy/Pyrefly: Cannot find implementation of 'FlextResult'

**Symptom:**

```
error: Cannot find implementation of 'FlextResult'
```

**Solutions:**

```bash
# Ensure PYTHONPATH is set
PYTHONPATH=src mypy src/

# Use pyrefly instead (newer, better for Python 3.13)
pyrefly check src/

# Add to config if needed
# pyproject.toml
[tool.mypy]
python_version = "3.13"
namespace_packages = true
explicit_package_bases = true
```

### Type Error: 'FlextResult[int]' has no attribute 'value'

**Symptom:**

```
TypeError: 'FlextResult[int]' has no attribute 'value'
error: 'FlextResult' object has no attribute 'value'
```

**Solutions:**

Both `.data` and `.value` work (backward compatibility):

```python
from flext_core import FlextResult

result = FlextResult[int].ok(42)

# ✅ Both work
value1 = result.value   # 42
value2 = result.data    # 42
```

### Type Error: Cannot instantiate 'FlextResult' directly

**Symptom:**

```
TypeError: Cannot instantiate generic class FlextResult without type parameters
```

**Solutions:**

```python
# ❌ WRONG - Missing type parameter
result = FlextResult.ok(42)

# ✅ CORRECT - Provide type parameter
result = FlextResult[int].ok(42)

# ✅ Also works - Type inference
result: FlextResult[int] = FlextResult.ok(42)
```

## Runtime Errors

### AttributeError: 'FlextModels' object has no attribute 'ValueObject'

**Symptom:**

```
AttributeError: type object 'FlextModels' has no attribute 'ValueObject'
```

**Solution:**

Use `.Value` instead of `.ValueObject`:

```python
# ❌ WRONG
class Address(FlextModels.ValueObject):
    pass

# ✅ CORRECT
class Address(FlextModels.Value):
    pass
```

### FlextContainer: Service not registered

**Symptom:**

```
FlextResult[NoneType].fail("Service 'logger' not registered")
```

**Solutions:**

```python
from flext_core import FlextContainer

container = FlextContainer.get_global()

# Check if service exists
if container.get("logger").is_failure:
    print("Logger not registered, registering now...")
    container.register("logger", FlextLogger(__name__))

# Use safely
logger_result = container.get("logger")
if logger_result.is_success:
    logger = logger_result.value
    logger.info("Message")
else:
    print(f"Error: {logger_result.error}")
```

### Circular Dependency in Dependency Injection

**Symptom:**

```
RuntimeError: Circular dependency detected
```

**Solutions:**

```python
# ✅ CORRECT - Use factory functions for circular deps
from flext_core import FlextContainer

container = FlextContainer.get_global()

# Register factories instead of instances
container.register_factory("logger_a", lambda: create_logger_a())
container.register_factory("logger_b", lambda: create_logger_b())

def create_logger_a():
    return FlextLogger("a")

def create_logger_b():
    return FlextLogger("b")

# ❌ WRONG - Direct circular reference
# logger_a depends on logger_b, logger_b depends on logger_a
```

## Pydantic Validation Errors

### ValidationError: field required

**Symptom:**

```
pydantic.ValidationError: 1 validation error for User
name
  Field required [type=missing, input_value={...}, input_type=dict, ...]
```

**Solutions:**

```python
from pydantic import BaseModel, Field
from flext_core import FlextModels

# ✅ CORRECT - Provide all required fields
class User(FlextModels.Entity):
    name: str
    email: str

user = User(id="1", name="Alice", email="alice@example.com")

# ❌ WRONG - Missing required fields
try:
    user = User(id="1")  # Missing name and email
except ValueError as e:
    print(f"Validation error: {e}")
```

### ValidationError: Value error, Invalid email format

**Symptom:**

```
ValidationError: 1 validation error for User
email
  Value error, Invalid email format
```

**Solutions:**

```python
from pydantic import BaseModel, EmailStr, field_validator

class User(BaseModel):
    email: str

    @field_validator('email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v

# Use try-except to catch validation errors
try:
    user = User(email="invalid")
except ValueError as e:
    print(f"Validation failed: {e}")
```

## Configuration Issues

### Configuration not loading from file

**Symptom:**

```
Config value not found or returns None
```

**Solutions:**

```python
from flext_core import FlextSettings
import os

# ✅ CORRECT - Verify file exists
config_file = 'config.toml'
if not os.path.exists(config_file):
    print(f"Config file not found: {config_file}")
    # Use defaults or raise error
    config = FlextSettings()
else:
    config = FlextSettings(config_files=[config_file])

# Verify values loaded
database_url = config.get('database.url')
print(f"Database URL: {database_url}")

# Get required values safely
api_key = config.get('api.key', required=True)
```

### Environment variables not expanding

**Symptom:**

```
Config value is literal string like "${API_KEY}" instead of actual value
```

**Solutions:**

```bash
# Ensure environment variable is set
export API_KEY="my_api_key"

# Verify it's set
echo $API_KEY

# Then run your application
python main.py
```

**Python code:**

```python
from flext_core import FlextSettings
import os

# Set environment variable if not set
if 'API_KEY' not in os.environ:
    os.environ['API_KEY'] = 'default_key'

config = FlextSettings(config_files=['config.toml'])
api_key = config.get('api.key')  # Will expand from environment
```

## Database and External Service Issues

### Database connection timeout

**Symptom:**

```
TimeoutError: Connection timeout
```

**Solutions:**

```python
from flext_core import FlextResult
import asyncio

def connect_to_database(url: str, timeout: int = 5) -> FlextResult[Connection]:
    """Connect with timeout handling."""
    try:
        # Try with timeout
        connection = db.connect(url, timeout=timeout)
        return FlextResult[Connection].ok(connection)
    except TimeoutError as e:
        return FlextResult[Connection].fail(
            f"Database connection timeout after {timeout}s: {str(e)}"
        )
    except Exception as e:
        return FlextResult[Connection].fail(f"Connection failed: {str(e)}")

# Usage
result = connect_to_database("postgresql://localhost/myapp", timeout=10)
if result.is_success:
    connection = result.value
else:
    print(f"Error: {result.error}")
```

### External API call fails

**Symptom:**

```
requests.ConnectionError: Failed to establish connection
```

**Solutions:**

```python
from flext_core import FlextResult, FlextLogger
import requests

logger = FlextLogger(__name__)

def call_external_api(url: str, retries: int = 3) -> FlextResult[dict]:
    """Call API with retry logic."""
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return FlextResult[dict].ok(response.json())
            else:
                return FlextResult[dict].fail(
                    f"API error {response.status_code}"
                )
        except requests.Timeout:
            logger.warning(f"Request timeout (attempt {attempt + 1}/{retries})")
            if attempt == retries - 1:
                return FlextResult[dict].fail("API request timeout")
        except requests.ConnectionError:
            logger.warning(f"Connection failed (attempt {attempt + 1}/{retries})")
            if attempt == retries - 1:
                return FlextResult[dict].fail("API connection failed")

    return FlextResult[dict].fail("API call failed after retries")

# Usage
result = call_external_api("https://api.example.com/data")
```

## Logging Issues

### Logs not appearing

**Symptom:**

```
Logging messages not showing up
```

**Solutions:**

```python
from flext_core import FlextLogger
import logging

# ✅ CORRECT - Set log level
logger = FlextLogger(__name__)

# Logger works with both levels
logger.debug("Debug message")    # Only if DEBUG level
logger.info("Info message")      # Always shown
logger.warning("Warning")        # Always shown
logger.error("Error")            # Always shown

# Set environment variable to control level
import os
os.environ['LOG_LEVEL'] = 'DEBUG'
```

### Logging causes performance issues

**Symptom:**

```
Application runs slowly with logging enabled
```

**Solutions:**

```python
# ✅ CORRECT - Use structured logging efficiently
from flext_core import FlextLogger

logger = FlextLogger(__name__)

# Good - log only what's needed
logger.info("User logged in", extra={"user_id": user_id})

# ❌ WRONG - Don't log everything
for i in range(1000000):
    logger.debug(f"Loop iteration {i}")  # This will be slow!

# ✅ BETTER - Log strategically
logger.info("Processing started")
for i in range(1000000):
    # ... processing ...
    if i % 10000 == 0:
        logger.debug(f"Processed {i} items")
logger.info("Processing completed")
```

## Testing Issues

### Pytest: Module not found in tests

**Symptom:**

```
ModuleNotFoundError: No module named 'myapp'
```

**Solutions:**

```bash
# Run pytest with PYTHONPATH
PYTHONPATH=src pytest tests/

# Or add to pytest config
# pyproject.toml
[tool.pytest.ini_options]
pythonpath = ["src"]
```

### Fixture scope issues

**Symptom:**

```
Fixture 'database' not found
```

**Solutions:**

```python
import pytest

# ✅ CORRECT - Define fixtures in conftest.py
# tests/conftest.py
@pytest.fixture
def database():
    db = setup_database()
    yield db
    cleanup_database(db)

# Then use in tests
def test_something(database):
    result = database.query("SELECT * FROM users")
    assert len(result) > 0

# ❌ WRONG - Defining fixture in individual test file doesn't share
# tests/test_module.py
@pytest.fixture
def database():  # Only available in this file!
    pass
```

### Test isolation issues

**Symptom:**

```
Tests fail when run together but pass individually
```

**Solutions:**

```python
import pytest

# ✅ CORRECT - Use fixtures to isolate state
@pytest.fixture
def clean_container():
    """Clean container for each test."""
    from flext_core import FlextContainer
    container = FlextContainer.get_global()
    container.clear()
    yield container
    container.clear()

def test_service_registration(clean_container):
    clean_container.register("service", MyService())
    # Test doesn't affect other tests

# ❌ WRONG - Shared global state
service = None

def test_setup():
    global service
    service = MyService()

def test_use():
    assert service is not None  # Depends on test order!
```

## Performance Issues

### Application runs slowly

**Symptom:**

```
Requests take longer than expected
```

**Diagnostic Steps:**

```python
import time
from flext_core import FlextLogger, FlextResult

logger = FlextLogger(__name__)

def slow_operation() -> FlextResult[str]:
    """Operation with timing."""
    start_time = time.time()

    try:
        # ... perform operation ...
        result = do_something()

        elapsed = time.time() - start_time
        logger.info(
            "Operation completed",
            extra={
                "operation": "do_something",
                "duration_ms": elapsed * 1000,
                "slow": elapsed > 1.0
            }
        )

        return FlextResult[str].ok(result)
    except Exception as e:
        logger.error(f"Operation failed: {str(e)}")
        return FlextResult[str].fail("Operation failed")

# Profile specific functions
import cProfile
import pstats

def profile_operation():
    profiler = cProfile.Profile()
    profiler.enable()

    result = slow_operation()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 functions
```

### Memory usage increasing

**Symptom:**

```
Application memory usage keeps growing
```

**Solutions:**

```python
from flext_core import FlextContainer

# ✅ CORRECT - Register singletons carefully
container = FlextContainer.get_global()

# Don't accumulate services
# ❌ WRONG - Register same service many times
for i in range(1000):
    container.register(f"service_{i}", MyService())

# ✅ CORRECT - Reuse or manage lifecycle
container.register("service", MyService(), singleton=True)

# Clean up when done
container.clear()  # Frees resources
```

## Getting Help

### When to check logs

```bash
# View application logs
tail -f logs/app.log

# Filter by error level
grep ERROR logs/app.log

# Search for specific operation
grep -A 5 "operation_name" logs/app.log
```

### When to check configuration

```bash
# Print loaded configuration (hide secrets!)
python -c "
from flext_core import FlextSettings
config = FlextSettings(config_files=['config.toml'])
# Print non-sensitive values
for key in ['app', 'database', 'api']:
    section = config.get_section(key)
    if section:
        print(f'{key}: {section}')
"
```

### When to enable debug logging

```python
import os
import logging

# Enable debug logging
os.environ['LOG_LEVEL'] = 'DEBUG'

from flext_core import FlextLogger
logger = FlextLogger(__name__)
logger.debug("Debug logging enabled")
```

## Common Solutions Summary

| Issue                                 | Solution                                   |
| ------------------------------------- | ------------------------------------------ |
| ImportError: FlextResult not found    | Use `from flext_core import FlextResult`   |
| Module not found in tests             | Set `PYTHONPATH=src` before running        |
| Type error: missing type parameter    | Use `FlextResult[T]` with type `T`         |
| AttributeError: ValueObject not found | Use `FlextModels.Value` instead                      |
| Service not registered                | Call `container.register()` first          |
| Config value None                     | Check file exists and value is in config   |
| Tests fail together                   | Use fixtures to isolate state              |
| Slow performance                      | Profile with `cProfile` to find bottleneck |

## Next Steps

1. **Error Handling**: See [Error Handling Guide](./error-handling.md) for comprehensive error patterns
2. **Testing**: Check [Testing Guide](./testing.md) for debugging test failures
3. **Configuration**: Review [Configuration Guide](./configuration.md) for configuration issues
4. **API Reference**: Consult [API Reference](../api-reference/) for detailed API documentation

## See Also

- [Error Handling Guide](./error-handling.md) - Comprehensive error handling patterns
- [Testing Guide](./testing.md) - Debugging and troubleshooting tests
- [Configuration Guide](./configuration.md) - Configuration troubleshooting
- [Getting Started](./getting-started.md) - Setup and installation
- [API Reference](../api-reference/) - Complete API documentation
- **FLEXT CLAUDE.md**: Architecture principles and development workflow

## Additional Resources

- Documentation: [docs/](../)
- GitHub Issues: [Report bugs](https://github.com/flext-sh/flext-core/issues)
- API Reference: [API Docs](../api-reference/)
- Getting Started: [Quick Start](../quick-start.md)
