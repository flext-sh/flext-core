# Advanced Troubleshooting Guide

**Troubleshooting based on actual FLEXT Core implementation**

## 🔍 Basic Diagnostics

### FlextResult Error Analysis

Debug FlextResult issues using actual API:

```python
from flext_core import FlextResult
import logging

def debug_flext_result(result: FlextResult[object], operation_name: str) -> None:
    """Debug FlextResult with actual API."""
    if result.is_failure:
        logging.error(f"Operation '{operation_name}' failed: {result.error}")

        # Log the operation type
        logging.debug(f"FlextResult failure in: {operation_name}")
        logging.debug(f"Error details: {result.error}")

def safe_divide(a: float, b: float) -> FlextResult[float]:
    """Example operation with error handling."""
    if b == 0:
        return FlextResult[None].fail("Division by zero not allowed")

    return FlextResult[None].ok(a / b)

# Usage with debugging
result = safe_divide(10, 0)
debug_flext_result(result, "safe_divide")

if result.success:
    print(f"Result: {result.data}")
else:
    print(f"Error: {result.error}")
```

### FlextContainer Debugging

Debug dependency injection using actual FlextContainer API:

```python
from flext_core import FlextContainer

def diagnose_container_basic():
    """Basic container diagnostics with actual API."""
    container = FlextContainer()

    # Register a test service
    test_service = "test_value"
    reg_result = container.register("test_service", test_service)

    print(f"Registration result: {reg_result.success}")
    if reg_result.is_failure:
        print(f"Registration error: {reg_result.error}")

    # Try to retrieve service
    get_result = container.get("test_service")
    print(f"Retrieval result: {get_result.success}")

    if get_result.success:
        print(f"Retrieved service: {get_result.data}")
    else:
        print(f"Retrieval error: {get_result.error}")

    # Try to get non-existent service
    missing_result = container.get("nonexistent")
    print(f"Missing service error: {missing_result.error}")

# Run diagnostics
diagnose_container_basic()
```

### Configuration Debugging

Debug configuration using actual FlextSettings:

```python
from flext_core import FlextSettings
import os

class DiagnosticSettings(FlextSettings):
    """Settings for debugging configuration issues."""

    # Basic settings
    app_name: str = "Debug App"
    debug: bool = False
    port: int = 8000

    # Optional settings
    database_url: str = "sqlite:///app.db"

    class Config:
        env_prefix = "DEBUG_"

def diagnose_configuration():
    """Diagnose configuration issues."""
    print("Configuration Diagnosis:")
    print("=" * 30)

    # Check environment variables
    debug_vars = {k: v for k, v in os.environ.items() if k.startswith("DEBUG_")}
    print(f"Found DEBUG_ variables: {debug_vars}")

    # Try to load settings
    try:
        settings = DiagnosticSettings()
        print(f"✅ Settings loaded successfully")
        print(f"   App name: {settings.app_name}")
        print(f"   Debug: {settings.debug}")
        print(f"   Port: {settings.port}")
        print(f"   Database: {settings.database_url}")
    except Exception as e:
        print(f"❌ Settings failed to load: {e}")

# Run configuration diagnosis
diagnose_configuration()
```

## 🚨 Common Issues and Solutions

### Issue 1: FlextResult Chain Failures

**Problem**: FlextResult chains failing at unknown steps.

**Solution**: Add debugging to each step:

```python
from flext_core import FlextResult

def debug_chain_operations():
    """Debug FlextResult chains with step-by-step logging."""

    def step1() -> FlextResult[str]:
        print("Executing step1")
        result = FlextResult[None].ok("step1_data")
        print(f"Step1 result: {result.success}")
        return result

    def step2(data: str) -> FlextResult[str]:
        print(f"Executing step2 with data: {data}")
        if len(data) > 50:  # Simulate failure condition
            result = FlextResult[None].fail("Data too long in step2")
        else:
            result = FlextResult[None].ok(f"{data}_processed")
        print(f"Step2 result: {result.success}")
        return result

    def step3(data: str) -> FlextResult[str]:
        print(f"Executing step3 with data: {data}")
        result = FlextResult[None].ok(f"{data}_final")
        print(f"Step3 result: {result.success}")
        return result

    # Execute chain with debugging
    print("Starting chain execution:")
    result = (
        step1()
        .flat_map(step2)
        .flat_map(step3)
    )

    print(f"Final result: {result.success}")
    if result.success:
        print(f"Final data: {result.data}")
    else:
        print(f"Chain failed at: {result.error}")

    return result

# Test the chain
debug_chain_operations()
```

### Issue 2: Container Service Not Found

**Problem**: Services registered but not retrievable.

**Solution**: Check registration and retrieval patterns:

```python
from flext_core import FlextContainer

class TestService:
    def __init__(self, name: str):
        self.name = name

    def get_info(self) -> str:
        return f"Service: {self.name}"

def debug_container_services():
    """Debug container service registration and retrieval."""
    container = FlextContainer()

    # Test 1: Basic registration
    print("Test 1: Basic service registration")
    service = TestService("test_service")
    reg_result = container.register("my_service", service)
    print(f"Registration success: {reg_result.success}")

    if reg_result.is_failure:
        print(f"Registration failed: {reg_result.error}")
        return

    # Test 2: Retrieval
    print("\nTest 2: Service retrieval")
    get_result = container.get("my_service")
    print(f"Retrieval success: {get_result.success}")

    if get_result.success:
        retrieved_service = get_result.data
        print(f"Retrieved service info: {retrieved_service.get_info()}")
    else:
        print(f"Retrieval failed: {get_result.error}")

    # Test 3: Wrong key
    print("\nTest 3: Wrong service key")
    wrong_result = container.get("wrong_key")
    print(f"Wrong key result: {wrong_result.success}")
    print(f"Expected error: {wrong_result.error}")

# Run container debugging
debug_container_services()
```

### Issue 3: Configuration Not Loading from Environment

**Problem**: Environment variables not being picked up.

**Solution**: Verify environment variable naming and loading:

```python
from flext_core import FlextSettings
import os

class EnvTestSettings(FlextSettings):
    """Test environment variable loading."""

    test_value: str = "default"
    test_number: int = 42
    test_bool: bool = False

    class Config:
        env_prefix = "ENVTEST_"

def test_environment_loading():
    """Test environment variable loading."""
    print("Environment Variable Loading Test:")
    print("=" * 40)

    # Set environment variables
    os.environ["ENVTEST_TEST_VALUE"] = "from_env"
    os.environ["ENVTEST_TEST_NUMBER"] = "123"
    os.environ["ENVTEST_TEST_BOOL"] = "true"

    try:
        settings = EnvTestSettings()

        print("✅ Settings loaded successfully")
        print(f"   test_value: {settings.test_value} (expected: from_env)")
        print(f"   test_number: {settings.test_number} (expected: 123)")
        print(f"   test_bool: {settings.test_bool} (expected: True)")

        # Verify values were loaded from environment
        assert settings.test_value == "from_env"
        assert settings.test_number == 123
        assert settings.test_bool is True

        print("✅ All environment variables loaded correctly")

    except Exception as e:
        print(f"❌ Environment loading failed: {e}")
    finally:
        # Cleanup
        os.environ.pop("ENVTEST_TEST_VALUE", None)
        os.environ.pop("ENVTEST_TEST_NUMBER", None)
        os.environ.pop("ENVTEST_TEST_BOOL", None)

# Run environment test
test_environment_loading()
```

## 🔧 Debugging Techniques

### Enhanced FlextResult Debugging

Create a debugging wrapper for FlextResult:

```python
from flext_core import FlextResult
from typing import TypeVar, Generic

T = TypeVar('T')

class DebugFlextResult(Generic[T]):
    """Debug wrapper for FlextResult operations."""

    def __init__(self, result: FlextResult[T], context: str = ""):
        self.result = result
        self.context = context

    def debug_info(self) -> str:
        """Get debug information."""
        info = f"Context: {self.context}\n"
        info += f"Success: {self.result.success}\n"

        if self.result.success:
            info += f"Data: {self.result.data}\n"
            info += f"Data type: {type(self.result.data).__name__}\n"
        else:
            info += f"Error: {self.result.error}\n"

        return info

    def print_debug(self) -> None:
        """Print debug information."""
        print(self.debug_info())

def debug_operation_chain():
    """Example of debugging operation chain."""

    def operation_a() -> FlextResult[str]:
        return FlextResult[None].ok("data_a")

    def operation_b(data: str) -> FlextResult[str]:
        return FlextResult[None].ok(f"{data}_b")

    def operation_c(data: str) -> FlextResult[str]:
        if len(data) > 10:
            return FlextResult[None].fail("Data too long")
        return FlextResult[None].ok(f"{data}_c")

    # Debug each step
    result_a = operation_a()
    debug_a = DebugFlextResult(result_a, "operation_a")
    debug_a.print_debug()

    if result_a.success:
        result_b = operation_b(result_a.data)
        debug_b = DebugFlextResult(result_b, "operation_b")
        debug_b.print_debug()

        if result_b.success:
            result_c = operation_c(result_b.data)
            debug_c = DebugFlextResult(result_c, "operation_c")
            debug_c.print_debug()

# Run debug chain
debug_operation_chain()
```

### Simple Performance Monitoring

Basic performance monitoring for FLEXT Core operations:

```python
import time
import functools
from flext_core import FlextResult

def time_operation(operation_name: str):
    """Simple decorator to time operations."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                success = True
                if isinstance(result, FlextResult):
                    success = result.success
            except Exception as e:
                result = e
                success = False
            finally:
                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000

                status = "SUCCESS" if success else "FAILED"
                print(f"⏱️ {operation_name}: {duration_ms:.2f}ms [{status}]")

            return result
        return wrapper
    return decorator

# Usage example
class TimedUserService:
    """Example service with timing."""

    @time_operation("create_user")
    def create_user(self, name: str, email: str) -> FlextResult[dict]:
        """Timed user creation."""
        # Simulate work
        time.sleep(0.01)  # 10ms simulation

        if not email or "@" not in email:
            return FlextResult[None].fail("Invalid email")

        user_data = {"name": name, "email": email, "created": True}
        return FlextResult[None].ok(user_data)

    @time_operation("validate_user")
    def validate_user(self, user_data: dict) -> FlextResult[bool]:
        """Timed user validation."""
        # Simulate validation work
        time.sleep(0.005)  # 5ms simulation

        if not user_data.get("name") or not user_data.get("email"):
            return FlextResult[None].fail("Missing required fields")

        return FlextResult[None].ok(data=True)

# Test timed operations
service = TimedUserService()

# This will print timing information
result1 = service.create_user("John Doe", "john@example.com")
if result1.success:
    result2 = service.validate_user(result1.data)

# Test failure case
result3 = service.create_user("Jane", "invalid-email")
```

## 📊 Health Checks

### Basic Health Check System

Simple health check for FLEXT Core components:

```python
from flext_core import FlextResult, FlextContainer, FlextSettings

class HealthChecker:
    """Basic health checker for FLEXT Core components."""

    def check_flext_result(self) -> FlextResult[str]:
        """Check FlextResult functionality."""
        try:
            # Test success case
            success_result = FlextResult[None].ok("test")
            if not success_result.success or success_result.data != "test":
                return FlextResult[None].fail("FlextResult success case failed")

            # Test failure case
            failure_result = FlextResult[None].fail("test error")
            if failure_result.success or failure_result.error != "test error":
                return FlextResult[None].fail("FlextResult failure case failed")

            return FlextResult[None].ok("FlextResult health check passed")

        except Exception as e:
            return FlextResult[None].fail(f"FlextResult health check error: {e}")

    def check_flext_container(self) -> FlextResult[str]:
        """Check FlextContainer functionality."""
        try:
            container = FlextContainer()

            # Test registration
            test_value = "health_check_value"
            reg_result = container.register("health_test", test_value)
            if reg_result.is_failure:
                return FlextResult[None].fail(f"Container registration failed: {reg_result.error}")

            # Test retrieval
            get_result = container.get("health_test")
            if get_result.is_failure:
                return FlextResult[None].fail(f"Container retrieval failed: {get_result.error}")

            if get_result.data != test_value:
                return FlextResult[None].fail("Container data mismatch")

            return FlextResult[None].ok("FlextContainer health check passed")

        except Exception as e:
            return FlextResult[None].fail(f"FlextContainer health check error: {e}")

    def check_flext_settings(self) -> FlextResult[str]:
        """Check FlextSettings functionality."""
        try:
            class HealthSettings(FlextSettings):
                test_field: str = "default_value"

                class Config:
                    env_prefix = "HEALTH_"

            settings = HealthSettings()
            if settings.test_field != "default_value":
                return FlextResult[None].fail("Settings default value failed")

            return FlextResult[None].ok("FlextSettings health check passed")

        except Exception as e:
            return FlextResult[None].fail(f"FlextSettings health check error: {e}")

    def run_all_checks(self) -> FlextResult[dict]:
        """Run all health checks."""
        results = {}
        overall_health = True

        # Run each check
        checks = [
            ("flext_result", self.check_flext_result),
            ("flext_container", self.check_flext_container),
            ("flext_settings", self.check_flext_settings),
        ]

        for check_name, check_func in checks:
            result = check_func()
            results[check_name] = {
                "success": result.success,
                "message": result.data if result.success else result.error
            }

            if result.is_failure:
                overall_health = False

        results["overall_health"] = overall_health

        if overall_health:
            return FlextResult[None].ok(results)
        else:
            return FlextResult[None].fail(f"Health check failures detected: {results}")

# Run health checks
def run_health_check():
    """Run complete health check."""
    print("FLEXT Core Health Check")
    print("=" * 30)

    checker = HealthChecker()
    health_result = checker.run_all_checks()

    if health_result.success:
        print("✅ Overall Health: HEALTHY")
        results = health_result.data

        for check_name, check_result in results.items():
            if check_name != "overall_health":
                status = "✅" if check_result["success"] else "❌"
                print(f"{status} {check_name}: {check_result['message']}")
    else:
        print("❌ Overall Health: UNHEALTHY")
        print(f"Details: {health_result.error}")

# Run the health check
if __name__ == "__main__":
    run_health_check()
```

## ⚠️ Important Notes

- This guide uses **ACTUAL** FLEXT Core APIs from src/flext_core/
- All examples are **TESTED** against the current implementation
- Methods like `container.list_services()` and `container.get_service_info()` don't exist in the current API - use basic `register()` and `get()` methods
- For advanced monitoring, implement custom solutions using the available API

---

**This troubleshooting guide is based on the real implementation in src/flext_core/**
