<!-- Generated from docs/guides/troubleshooting.md for flext-core. -->
<!-- Source of truth: workspace docs/guides/. -->

# flext-core - FLEXT Troubleshooting Guide

> Project profile: `flext-core`



<!-- TOC START -->
- Quick Diagnosis
  - Health Check Commands
  - System Status
- Common Issues
  - 1. Import Errors
  - r
  - 2. Type Checking Errors
  - 3. Test Failures
  - 4. Configuration Issues
  - 5. LDIF Processing Issues
  - 6. Migration Issues
  - 7. Performance Issues
- Debugging Techniques
  - 1. Logging Configuration
  - 2. Exception Handling
  - 3. Debug Mode
  - 4. Step-by-Step Debugging
- Error Codes Reference
  - FLEXT Core Errors
  - LDIF Processing Errors
  - API Errors
- Performance Troubleshooting
  - Memory Issues
  - CPU Issues
- Getting Help
  - Self-Service Resources
  - Community Support
  - Reporting Issues
  - Your minimal example here
- Prevention
  - Best Practices
- Resources
<!-- TOC END -->

This guide covers common issues, their solutions, and debugging techniques for FLEXT applications and libraries.

## Quick Diagnosis

### Health Check Commands

```bash
# Check overall system health
make validate

# Check specific components
make lint          # Code quality
make type-check    # Type safety
make test          # Functionality
make security      # Security issues

# Check individual projects
cd flext-core && make validate
cd flext-ldif && make validate
cd flext-api && make validate
```

### System Status

```bash
# Check Python version
python --version  # Should be 3.13+

# Check Poetry environment
poetry env info

# Check dependencies
poetry show --tree

# Check git status
git status
```

## Common Issues

### 1. Import Errors

#### Problem: ModuleNotFoundError

```python
# Error
ModuleNotFoundError: No module named 'flext_core'
```

#### Solutions

**Check PYTHONPATH:**

```bash
export PYTHONPATH=src
python -c "import flext_core; print(flext_core.__file__)"
```

**Reinstall dependencies:**

```bash
make clean
make setup
```

**Check Poetry environment:**

```bash
poetry env info
poetry install
```

### r

```python
# Debug import issues
import sys
print("Python path:")
for path in sys.path:
    print(f"  {path}")

print("\nTrying to import flext_core...")
try:
    import flext_core
    print(f"Success: {flext_core.__file__}")
except ImportError as e:
    print(f"Failed: {e}")
```

### 2. Type Checking Errors

#### Problem: MyPy errors

```python
# Error
error: Argument 1 to "process" has incompatible type "str"; expected "dict[str, object]"
```

#### Solutions

**Fix type annotations:**

```python
# ❌ WRONG
def process(data):
    return data

# ✅ CORRECT
def process(data: dict[str, object]) -> FlextResult[ProcessedData]:
    return FlextResult.ok(ProcessedData(**data))
```

**Run MyPy with details:**

```bash
mypy src/module.py --show-error-codes --show-traceback
```

**Check specific error:**

```bash
mypy src/ --show-error-codes | grep "error-code"
```

### 3. Test Failures

#### Problem: Tests failing

```python
# Error
AssertionError: Expected success but got failure
```

#### Solutions

**Run with verbose output:**

```bash
pytest tests/unit/test_module.py -vv --tb=long
```

**Debug specific test:**

```bash
pytest tests/unit/test_module.py::TestClass::test_method -v --pdb
```

**Check test data:**

```python
def test_with_debug():
    result = my_function()
    print(f"Result: {result}")
    print(f"Success: {result.is_success}")
    if result.is_failure:
        print(f"Error: {result.failure()}")
    assert result.is_success
```

### 4. Configuration Issues

#### Problem: Configuration not loading

```python
# Error
ValidationError: field required
```

#### Solutions

**Check environment variables:**

```bash
env | grep FLEXT_
```

**Validate configuration:**

```python
from flext_core import FlextBus
from flext_core import FlextSettings
from flext_core import FlextConstants
from flext_core import FlextContainer
from flext_core import FlextContext
from flext_core import FlextDecorators
from flext_core import FlextDispatcher
from flext_core import FlextExceptions
from flext_core import h
from flext_core import FlextLogger
from flext_core import x
from flext_core import FlextModels
from flext_core import FlextProcessors
from flext_core import p
from flext_core import FlextRegistry
from flext_core import FlextResult
from flext_core import FlextRuntime
from flext_core import FlextService
from flext_core import t
from flext_core import u

try:
    config = FlextSettings()
    print("Configuration valid")
except ValidationError as e:
    print(f"Configuration error: {e}")
```

**Debug configuration loading:**

```python
import os
from flext_core import FlextBus
from flext_core import FlextSettings
from flext_core import FlextConstants
from flext_core import FlextContainer
from flext_core import FlextContext
from flext_core import FlextDecorators
from flext_core import FlextDispatcher
from flext_core import FlextExceptions
from flext_core import h
from flext_core import FlextLogger
from flext_core import x
from flext_core import FlextModels
from flext_core import FlextProcessors
from flext_core import p
from flext_core import FlextRegistry
from flext_core import FlextResult
from flext_core import FlextRuntime
from flext_core import FlextService
from flext_core import t
from flext_core import u

# Print all FLEXT environment variables
for key, value in os.environ.items():
    if key.startswith('FLEXT_'):
        print(f"{key}={value}")

# Load and print configuration
config = FlextSettings()
print(f"Config: {config.dict()}")
```

### 5. LDIF Processing Issues

#### Problem: LDIF parsing fails

```python
# Error
LdifParsingException: Invalid LDIF format
```

#### Solutions

**Check LDIF content:**

```python
from flext_ldif import FlextLdif

ldif = FlextLdif()
content = """dn: cn=test,dc=example,dc=com
cn: test
objectClass: inetOrgPerson"""

result = ldif.parse(content)
if result.is_failure:
    print(f"Parse error: {result.failure()}")
    print(f"Content: {repr(content)}")
```

**Enable debug logging:**

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your LDIF processing code
```

**Validate LDIF format:**

```python
# Check for common LDIF issues
def validate_ldif_content(content: str) -> t.StringList:
    issues = []

    if not content.strip():
        issues.append("Empty content")

    if not content.startswith("dn:"):
        issues.append("Missing DN line")

    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line and not line.startswith(('dn:', ' ', '\t')) and ':' not in line:
            issues.append(f"Invalid line {i+1}: {line}")

    return issues
```

### 6. Migration Issues

#### Problem: Migration fails

```python
# Error
LdifMigrationException: Server compatibility error
```

#### Solutions

**Check server configuration:**

```python
from flext_ldif import FlextLdifSettings

config = FlextLdifSettings(
    source_server="oid",
    target_server="oud",
    preserve_oid_modifiers=True,
    handle_schema_extensions=True
)

print(f"Config: {config.dict()}")
```

**Enable server quirks:**

```python
config = FlextLdifSettings(
    servers_enabled=True,
    source_server="oid",
    target_server="oud"
)
```

**Test with sample data:**

```python
# Test migration with small sample
sample_ldif = """dn: cn=test,dc=example,dc=com
cn: test
objectClass: inetOrgPerson"""

result = ldif.parse(sample_ldif)
if result.is_success:
    print("Sample parsing successful")
else:
    print(f"Sample parsing failed: {result.failure()}")
```

### 7. Performance Issues

#### Problem: Slow processing

```python
# Symptoms
# - High memory usage
# - Slow response times
# - Timeout errors
```

#### Solutions

**Profile memory usage:**

```python
import psutil
import os

def profile_memory():
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss

    # Your processing code here

    final_memory = process.memory_info().rss
    memory_used = final_memory - initial_memory

    print(f"Memory used: {memory_used / 1024 / 1024:.2f} MB")

profile_memory()
```

**Optimize batch size:**

```python
from flext_ldif import FlextLdifSettings

# Reduce batch size for memory-constrained environments
config = FlextLdifSettings(
    batch_size=100,  # Instead of default 1000
    parallel_processing=False  # Disable for memory issues
)
```

**Enable parallel processing:**

```python
config = FlextLdifSettings(
    parallel_processing=True,
    max_workers=4  # Adjust based on CPU cores
)
```

## Debugging Techniques

### 1. Logging Configuration

```python
import logging
from flext_core import FlextBus
from flext_core import FlextSettings
from flext_core import FlextConstants
from flext_core import FlextContainer
from flext_core import FlextContext
from flext_core import FlextDecorators
from flext_core import FlextDispatcher
from flext_core import FlextExceptions
from flext_core import h
from flext_core import FlextLogger
from flext_core import x
from flext_core import FlextModels
from flext_core import FlextProcessors
from flext_core import p
from flext_core import FlextRegistry
from flext_core import FlextResult
from flext_core import FlextRuntime
from flext_core import FlextService
from flext_core import t
from flext_core import u

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Use FLEXT logger
logger = FlextLogger.get_logger(__name__)
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

### 2. Exception Handling

```python
from flext_core import FlextBus
from flext_core import FlextSettings
from flext_core import FlextConstants
from flext_core import FlextContainer
from flext_core import FlextContext
from flext_core import FlextDecorators
from flext_core import FlextDispatcher
from flext_core import FlextExceptions
from flext_core import h
from flext_core import FlextLogger
from flext_core import x
from flext_core import FlextModels
from flext_core import FlextProcessors
from flext_core import p
from flext_core import FlextRegistry
from flext_core import FlextResult
from flext_core import FlextRuntime
from flext_core import FlextService
from flext_core import t
from flext_core import u

def safe_operation(data: dict) -> FlextResult[dict]:
    try:
        # Your operation here
        result = process_data(data)
        return FlextResult.ok(result)
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        return FlextResult.fail(f"Validation failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return FlextResult.fail(f"Operation failed: {e}")
```

### 3. Debug Mode

```python
from flext_core import FlextBus
from flext_core import FlextSettings
from flext_core import FlextConstants
from flext_core import FlextContainer
from flext_core import FlextContext
from flext_core import FlextDecorators
from flext_core import FlextDispatcher
from flext_core import FlextExceptions
from flext_core import h
from flext_core import FlextLogger
from flext_core import x
from flext_core import FlextModels
from flext_core import FlextProcessors
from flext_core import p
from flext_core import FlextRegistry
from flext_core import FlextResult
from flext_core import FlextRuntime
from flext_core import FlextService
from flext_core import t
from flext_core import u

# Enable debug mode
config = FlextSettings(debug=True)

# Debug information will be printed
print(f"Debug mode: {config.debug}")
print(f"Log level: {config.log_level}")
```

### 4. Step-by-Step Debugging

```python
def debug_ldif_processing(content: str):
    """Debug LDIF processing step by step."""
    print(f"Input content length: {len(content)}")
    print(f"First 100 chars: {repr(content[:100])}")

    # Step 1: Basic validation
    if not content.strip():
        print("ERROR: Empty content")
        return

    # Step 2: Check DN format
    lines = content.split('\n')
    dn_line = lines[0] if lines else ""
    print(f"DN line: {repr(dn_line)}")

    if not dn_line.startswith("dn:"):
        print("ERROR: Missing or invalid DN line")
        return

    # Step 3: Try parsing
    from flext_ldif import FlextLdif
    ldif = FlextLdif()

    result = ldif.parse(content)
    if result.is_success:
        entries = result.unwrap()
        print(f"SUCCESS: Parsed {len(entries)} entries")
    else:
        print(f"ERROR: Parse failed: {result.failure()}")
```

## Error Codes Reference

### FLEXT Core Errors

| Error Code  | Description                     | Solution                                     |
| ----------- | ------------------------------- | -------------------------------------------- |
| `FLEXT_001` | Configuration validation failed | Check environment variables and config files |
| `FLEXT_002` | Dependency injection failed     | Verify service registration in container     |
| `FLEXT_003` | Type validation failed          | Fix type annotations and data types          |

### LDIF Processing Errors

| Error Code | Description                | Solution                                  |
| ---------- | -------------------------- | ----------------------------------------- |
| `LDIF_001` | Invalid LDIF format        | Check LDIF syntax and structure           |
| `LDIF_002` | Server compatibility error | Enable server quirks or check server type |
| `LDIF_003` | Schema validation failed   | Verify schema definitions and attributes  |

### API Errors

| Error Code | Description           | Solution                           |
| ---------- | --------------------- | ---------------------------------- |
| `API_001`  | HTTP request failed   | Check network connectivity and URL |
| `API_002`  | Authentication failed | Verify API keys and credentials    |
| `API_003`  | Rate limit exceeded   | Implement retry logic with backoff |

## Performance Troubleshooting

### Memory Issues

```python
# Monitor memory usage
import psutil
import os

def monitor_memory():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    print(f"RSS: {memory_info.rss / 1024 / 1024:.2f} MB")
    print(f"VMS: {memory_info.vms / 1024 / 1024:.2f} MB")

    # Check for memory leaks
    if memory_info.rss > 500 * 1024 * 1024:  # 500MB
        print("WARNING: High memory usage detected")

monitor_memory()
```

### CPU Issues

```python
# Monitor CPU usage
import psutil
import time

def monitor_cpu():
    process = psutil.Process(os.getpid())

    # Get CPU usage over time
    for i in range(10):
        cpu_percent = process.cpu_percent()
        print(f"CPU usage: {cpu_percent}%")
        time.sleep(1)

monitor_cpu()
```

## Getting Help

### Self-Service Resources

1. **Check Documentation**
   - API Reference
   - Configuration Guide
   - Development Guide

2. **Run Diagnostics**

   ```bash
   # System health check
   make validate

   # Project-specific check
   cd flext-core && make validate
   ```

3. **Check Logs**

   ```bash
   # Enable debug logging
   export FLEXT_LOG_LEVEL=DEBUG
   python your_script.py
   ```

### Community Support

1. **GitHub Issues**
   - [Create Issue](https://github.com/flext-sh/flext/issues)
   - Search existing issues
   - Check closed issues for solutions

2. **GitHub Discussions**
   - [Ask Question](https://github.com/flext-sh/flext/discussions)
   - Share solutions
   - Discuss best practices

3. **Email Support**
   - <dev@flext.com> for technical issues
   - <support@flext.com> for general questions

### Reporting Issues

When reporting issues, include:

1. **Environment Information**

   ```bash
   python --version
   poetry env info
   make info
   ```

2. **Error Details**

   ```python
   # Full error traceback
   import traceback
   try:
       # Your code here
   except Exception as e:
       traceback.print_exc()
   ```

3. **Minimal Reproduction**

   ```python
   # Minimal code that reproduces the issue
   from flext_core import FlextBus
   ```

from flext_core import FlextSettings
from flext_core import FlextConstants
from flext_core import FlextContainer
from flext_core import FlextContext
from flext_core import FlextDecorators
from flext_core import FlextDispatcher
from flext_core import FlextExceptions
from flext_core import h
from flext_core import FlextLogger
from flext_core import x
from flext_core import FlextModels
from flext_core import FlextProcessors
from flext_core import p
from flext_core import FlextRegistry
from flext_core import FlextResult
from flext_core import FlextRuntime
from flext_core import FlextService
from flext_core import t
from flext_core import u

### Your minimal example here

4. **Expected vs Actual Behavior**
- What you expected to happen
- What actually happened
- Steps to reproduce

## Prevention

### Best Practices

1. **Always Use FlextResult**

```python
# ✅ GOOD
def process(data: dict) -> FlextResult[ProcessedData]:
    return FlextResult.ok(ProcessedData(**data))

# ❌ BAD
def process(data: dict) -> ProcessedData:
    return ProcessedData(**data)
```

2. **Validate Input Early**

   ```python
   def process_data(data: dict) -> FlextResult[dict]:
       if not data:
           return FlextResult.fail("Data required")

       # Process data
       return FlextResult.ok(processed_data)
   ```

3. **Use Type Hints**

   ```python
   # ✅ GOOD
   def process(items: list[Item]) -> FlextResult[list[ProcessedItem]]:
       pass

   # ❌ BAD
   def process(items):
       pass
   ```

4. **Test Thoroughly**

   ```python
   def test_process_data():
       # Test success case
       result = process_data({"key": "value"})
       assert result.is_success

       # Test failure case
       result = process_data(None)
       assert result.is_failure
   ```

## Resources

- FLEXT Core Documentation
- Configuration Guide
- Development Guide
- Testing Guide
- [GitHub Issues](https://github.com/flext-sh/flext/issues)
- [GitHub Discussions](https://github.com/flext-sh/flext/discussions)
