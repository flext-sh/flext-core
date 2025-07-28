#!/usr/bin/env python3
"""FLEXT Decorators Enterprise Patterns Example.

Comprehensive demonstration of FlextDecorators system showing enterprise-grade
decorator patterns for validation, error handling, performance optimization,
logging, and immutability enforcement.

Features demonstrated:
    - Validation decorators with automatic argument checking
    - Error handling decorators with automatic exception management
    - Performance decorators with timing, caching, and memoization
    - Logging decorators with structured call and exception logging
    - Immutability decorators with argument and result protection
    - Functional decorators with composition and orchestration
    - Complete decorator orchestration for enterprise applications

Key Components:
    - FlextDecorators: Main consolidated decorator interface
    - FlextValidationDecorators: Argument and result validation
    - FlextErrorHandlingDecorators: Exception handling and recovery
    - FlextPerformanceDecorators: Timing, caching, and optimization
    - FlextLoggingDecorators: Structured logging and observability
    - FlextImmutabilityDecorators: Data protection and freeze patterns
    - FlextFunctionalDecorators: Functional composition patterns

This example shows real-world enterprise decorator scenarios
demonstrating the power and flexibility of the FlextDecorators system.
"""

import time
import traceback
from typing import Any

from flext_core.decorators import (
    FlextDecorators,
    FlextErrorHandlingDecorators,
    FlextImmutabilityDecorators,
    FlextLoggingDecorators,
    FlextPerformanceDecorators,
    FlextValidationDecorators,
    safe_call,
)


def demonstrate_validation_decorators() -> None:
    """Demonstrate validation decorators with automatic argument checking."""
    print("\n" + "=" * 80)
    print("✅ VALIDATION DECORATORS")
    print("=" * 80)

    # 1. Basic argument validation
    print("\n1. Basic argument validation:")

    @FlextValidationDecorators.validate_arguments
    def validate_user_data(name: str, email: str, age: int) -> dict[str, Any]:
        """Function with validated arguments."""
        return {"name": name, "email": email, "age": age, "status": "validated"}

    try:
        result = validate_user_data("John Doe", "john@example.com", 30)
        print(f"✅ Valid arguments: {result}")
    except Exception as e:
        print(f"❌ Validation failed: {e}")

    # 2. Custom validation decorator
    print("\n2. Custom validation decorator:")

    def email_validator(email: str) -> bool:
        """Simple email validation."""
        return "@" in email and "." in email

    email_validation_decorator = FlextValidationDecorators.create_validation_decorator(
        email_validator,
    )

    @email_validation_decorator
    def register_user(email: str) -> str:
        """Register user with email validation."""
        return f"User registered with email: {email}"

    # Test valid email
    try:
        result = register_user("valid@example.com")
        print(f"✅ Valid email registration: {result}")
    except Exception as e:
        print(f"❌ Email validation failed: {e}")

    # Test invalid email
    try:
        result = register_user("invalid-email")
        print(f"✅ Invalid email registration: {result}")
    except Exception as e:
        print(f"❌ Email validation failed: {e}")

    # 3. Complex validation with result handling
    print("\n3. Complex validation with FlextResult:")

    def validate_age(age: int) -> bool:
        """Validate age is reasonable."""
        return isinstance(age, int) and 0 <= age <= 150

    @FlextValidationDecorators.create_validation_decorator(validate_age)
    def create_user_profile(name: str, age: int) -> dict[str, Any]:
        """Create user profile with age validation."""
        return {"name": name, "age": age, "created": time.time()}

    # Test valid age
    try:
        profile = create_user_profile("Alice", 25)
        print(f"✅ Valid profile: {profile}")
    except Exception as e:
        print(f"❌ Profile creation failed: {e}")

    # Test invalid age
    try:
        profile = create_user_profile("Bob", 200)
        print(f"✅ Invalid profile: {profile}")
    except Exception as e:
        print(f"❌ Profile creation failed: {e}")


def demonstrate_error_handling_decorators() -> None:
    """Demonstrate error handling decorators with automatic exception management."""
    print("\n" + "=" * 80)
    print("🛡️ ERROR HANDLING DECORATORS")
    print("=" * 80)

    # 1. Safe execution with automatic result wrapping
    print("\n1. Safe execution with FlextResult:")

    @FlextDecorators.safe_result
    def risky_division(a: float, b: float) -> float:
        """Division that might raise ValueError."""
        if b == 0:
            msg = "Cannot divide by zero"
            raise ValueError(msg)
        return a / b

    # Test successful operation
    result = risky_division(10, 2)
    if result.is_success:
        print(f"✅ Division successful: {result.data}")
    else:
        print(f"❌ Division failed: {result.error}")

    # Test error case
    result = risky_division(10, 0)
    if result.is_success:
        print(f"✅ Division successful: {result.data}")
    else:
        print(f"❌ Division failed (expected): {result.error}")

    # 2. Retry decorator for unreliable operations
    print("\n2. Retry decorator for unreliable operations:")

    attempt_count = 0

    @FlextErrorHandlingDecorators.retry_decorator
    def unreliable_service() -> str:
        """Service that fails first few times."""
        nonlocal attempt_count
        attempt_count += 1
        print(f"   Attempt {attempt_count}")

        if attempt_count < 3:
            msg = "Service temporarily unavailable"
            raise ConnectionError(msg)
        return "Service response: Success!"

    try:
        result = unreliable_service()
        print(f"✅ Service succeeded: {result}")
    except Exception as e:
        print(f"❌ Service failed after retries: {e}")

    # Reset for next test
    attempt_count = 0

    # 3. Custom error handler
    print("\n3. Custom error handling:")

    def custom_error_handler(exception: Exception) -> str:
        """Custom error handler that converts errors to messages."""
        return f"Handled error: {type(exception).__name__}: {exception}"

    custom_safe_decorator = FlextErrorHandlingDecorators.create_safe_decorator(
        custom_error_handler,
    )

    @custom_safe_decorator
    def operation_with_custom_handling() -> str:
        """Operation with custom error handling."""
        msg = "Something went wrong!"
        raise ValueError(msg)

    try:
        result = operation_with_custom_handling()
        print(f"✅ Custom handling result: {result}")
    except Exception as e:
        print(f"❌ Custom handling failed: {e}")

    # 4. Safe call utility decorator
    print("\n4. Safe call utility decorator:")

    @safe_call
    def potentially_failing_function(x: int) -> int:
        """Function that might fail."""
        if x < 0:
            msg = "Negative values not allowed"
            raise ValueError(msg)
        return x * 2

    # Safe call with success
    safe_result = potentially_failing_function(5)
    if safe_result.is_success:
        print(f"✅ Safe call succeeded: {safe_result.data}")
    else:
        print(f"❌ Safe call failed: {safe_result.error}")

    # Safe call with failure
    safe_result = potentially_failing_function(-3)
    if safe_result.is_success:
        print(f"✅ Safe call succeeded: {safe_result.data}")
    else:
        print(f"❌ Safe call failed (expected): {safe_result.error}")


def demonstrate_performance_decorators() -> None:
    """Demonstrate performance decorators with timing, caching, and memoization."""
    print("\n" + "=" * 80)
    print("⚡ PERFORMANCE DECORATORS")
    print("=" * 80)

    # 1. Timing decorator for performance measurement
    print("\n1. Timing decorator for performance measurement:")

    timing_decorator = FlextPerformanceDecorators.get_timing_decorator()

    @timing_decorator
    def slow_computation(n: int) -> int:
        """Simulate slow computation."""
        time.sleep(0.1)  # Simulate work
        return sum(range(n))

    result = slow_computation(1000)
    print(f"✅ Slow computation result: {result}")

    # 2. Memoization decorator for caching results
    print("\n2. Memoization decorator for expensive operations:")

    @FlextPerformanceDecorators.memoize_decorator
    def expensive_fibonacci(n: int) -> int:
        """Calculate Fibonacci number (expensive without memoization)."""
        if n <= 1:
            return n
        return expensive_fibonacci(n - 1) + expensive_fibonacci(n - 2)

    # First call - slow
    print("   Computing fibonacci(30) first time...")
    start_time = time.time()
    result1 = expensive_fibonacci(30)
    first_duration = time.time() - start_time
    print(f"   First call: {result1} (took {first_duration:.4f}s)")

    # Second call - fast (memoized)
    print("   Computing fibonacci(30) second time...")
    start_time = time.time()
    result2 = expensive_fibonacci(30)
    second_duration = time.time() - start_time
    print(f"   Second call: {result2} (took {second_duration:.4f}s)")
    print(f"   Speedup: {first_duration / second_duration:.2f}x faster")

    # 3. Cache decorator with size limit
    print("\n3. Cache decorator with configurable size:")

    cache_decorator = FlextPerformanceDecorators.create_cache_decorator(max_size=3)

    @cache_decorator
    def data_processor(data_id: str) -> str:
        """Process data with limited cache."""
        time.sleep(0.05)  # Simulate processing
        return f"Processed data: {data_id.upper()}"

    # Test cache behavior
    print("   Processing multiple data items:")
    for data_id in ["item1", "item2", "item3", "item4", "item1"]:
        start_time = time.time()
        result = data_processor(data_id)
        duration = time.time() - start_time
        print(f"   {data_id}: {result} (took {duration:.4f}s)")


def demonstrate_logging_decorators() -> None:
    """Demonstrate logging decorators with structured call and exception logging."""
    print("\n" + "=" * 80)
    print("📝 LOGGING DECORATORS")
    print("=" * 80)

    # 1. Call logging decorator
    print("\n1. Function call logging:")

    @FlextLoggingDecorators.log_calls_decorator
    def business_operation(operation_type: str, amount: float) -> dict[str, Any]:
        """Business operation with call logging."""
        return {
            "operation": operation_type,
            "amount": amount,
            "timestamp": time.time(),
            "status": "completed",
        }

    result = business_operation("transfer", 1000.50)
    print(f"✅ Business operation result: {result}")

    # 2. Exception logging decorator
    print("\n2. Exception logging:")

    @FlextLoggingDecorators.log_exceptions_decorator
    def risky_business_operation(operation_id: str) -> str:
        """Business operation that might fail."""
        if operation_id == "invalid":
            msg = f"Invalid operation ID: {operation_id}"
            raise ValueError(msg)
        return f"Operation {operation_id} completed successfully"

    # Test successful operation
    try:
        result = risky_business_operation("OP001")
        print(f"✅ Operation success: {result}")
    except Exception as e:
        print(f"❌ Operation failed: {e}")

    # Test failing operation (with logged exception)
    try:
        result = risky_business_operation("invalid")
        print(f"✅ Operation success: {result}")
    except Exception as e:
        print(f"❌ Operation failed (logged): {e}")

    # 3. Combined logging decorators
    print("\n3. Combined call and exception logging:")

    @FlextLoggingDecorators.log_calls_decorator
    @FlextLoggingDecorators.log_exceptions_decorator
    def comprehensive_service(
        service_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Service with comprehensive logging."""
        if "error" in params:
            msg = f"Service error: {params['error']}"
            raise RuntimeError(msg)

        return {
            "service": service_name,
            "params": params,
            "result": "success",
            "timestamp": time.time(),
        }

    # Test successful service call
    try:
        result = comprehensive_service("user_service", {"user_id": "123"})
        print(f"✅ Service success: {result}")
    except Exception as e:
        print(f"❌ Service failed: {e}")

    # Test failing service call
    try:
        result = comprehensive_service("user_service", {"error": "database_down"})
        print(f"✅ Service success: {result}")
    except Exception as e:
        print(f"❌ Service failed (logged): {e}")


def demonstrate_immutability_decorators() -> None:
    """Demonstrate immutability decorators with data protection patterns."""
    print("\n" + "=" * 80)
    print("🔒 IMMUTABILITY DECORATORS")
    print("=" * 80)

    # 1. Immutable function results
    print("\n1. Immutable function enforcement:")

    @FlextImmutabilityDecorators.immutable_decorator
    def create_immutable_config() -> dict[str, Any]:
        """Create configuration that should remain immutable."""
        return {
            "api_url": "https://api.example.com",
            "timeout": 30,
            "retries": 3,
            "features": ["feature_a", "feature_b"],
        }

    config = create_immutable_config()
    print(f"✅ Immutable config created: {config}")

    # Try to modify (this should be protected by the decorator)
    print("   Attempting to modify config...")
    try:
        config["api_url"] = "https://malicious-api.com"
        print(f"⚠️ Config was modified: {config}")
    except Exception as e:
        print(f"✅ Modification prevented: {e}")

    # 2. Freeze arguments decorator
    print("\n2. Argument freezing protection:")

    @FlextImmutabilityDecorators.freeze_args_decorator
    def process_user_data(user_data: dict[str, Any]) -> dict[str, Any]:
        """Process user data without modifying input."""
        # Try to modify input (should be prevented)
        try:
            user_data["modified"] = True
            print("⚠️ Input data was modified within function")
        except Exception as e:
            print(f"✅ Input modification prevented: {e}")

        # Return processed data
        result = user_data.copy()
        result["processed"] = True
        return result

    input_data = {"name": "John", "email": "john@example.com"}
    print(f"   Original data: {input_data}")

    result = process_user_data(input_data)
    print(f"✅ Processed data: {result}")
    print(f"   Original data unchanged: {input_data}")


def demonstrate_functional_decorators() -> None:
    """Demonstrate functional decorators with composition patterns."""
    print("\n" + "=" * 80)
    print("🔧 FUNCTIONAL DECORATORS AND COMPOSITION")
    print("=" * 80)

    # 1. Complete decorator orchestration
    print("\n1. Complete decorator orchestration:")

    # Combine multiple decorators for enterprise-grade function
    @FlextLoggingDecorators.log_calls_decorator
    @FlextErrorHandlingDecorators.retry_decorator
    @FlextValidationDecorators.validate_arguments
    @FlextPerformanceDecorators.memoize_decorator
    def enterprise_user_service(
        user_id: str,
        include_profile: bool = True,
    ) -> dict[str, Any]:
        """Enterprise user service with full decorator stack."""
        # Simulate potential failure
        if user_id == "error_user":
            msg = "Database connection failed"
            raise ConnectionError(msg)

        # Simulate expensive operation
        time.sleep(0.02)

        return {
            "user_id": user_id,
            "name": f"User {user_id}",
            "profile": {"active": True, "created": time.time()}
            if include_profile
            else None,
            "service_version": "1.0.0",
        }

    # Test successful operation
    try:
        result = enterprise_user_service("user123")
        print(f"✅ Enterprise service success: {result}")
    except Exception as e:
        print(f"❌ Enterprise service failed: {e}")

    # Test with caching (second call should be faster)
    try:
        result = enterprise_user_service("user123")
        print(f"✅ Enterprise service (cached): {result}")
    except Exception as e:
        print(f"❌ Enterprise service failed: {e}")

    # 2. Functional composition patterns
    print("\n2. Functional composition with decorators:")

    # Create a pipeline of decorated functions
    @FlextPerformanceDecorators.get_timing_decorator()
    def step1_validate(data: dict[str, Any]) -> dict[str, Any]:
        """Validation step."""
        if not data.get("email"):
            msg = "Email is required"
            raise ValueError(msg)
        return {**data, "step1_completed": True}

    @FlextPerformanceDecorators.get_timing_decorator()
    def step2_enrich(data: dict[str, Any]) -> dict[str, Any]:
        """Enrichment step."""
        return {**data, "enriched": True, "timestamp": time.time()}

    @FlextPerformanceDecorators.get_timing_decorator()
    def step3_transform(data: dict[str, Any]) -> dict[str, Any]:
        """Transformation step."""
        return {
            "user": {
                "email": data["email"],
                "metadata": {
                    "enriched": data["enriched"],
                    "timestamp": data["timestamp"],
                    "step1_completed": data["step1_completed"],
                },
            },
        }

    # Execute pipeline
    print("   Executing functional pipeline:")
    pipeline_data = {"email": "pipeline@example.com"}

    try:
        step1_result = step1_validate(pipeline_data)
        step2_result = step2_enrich(step1_result)
        final_result = step3_transform(step2_result)
        print(f"✅ Pipeline completed: {final_result}")
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")


def demonstrate_decorator_best_practices() -> None:
    """Demonstrate decorator best practices and advanced patterns."""
    print("\n" + "=" * 80)
    print("🏆 DECORATOR BEST PRACTICES AND ADVANCED PATTERNS")
    print("=" * 80)

    # 1. Metadata preservation
    print("\n1. Metadata preservation in decorators:")

    def custom_decorator(func):
        """Custom decorator that preserves metadata."""

        def wrapper(*args, **kwargs):
            print(f"   Calling {func.__name__}")
            return func(*args, **kwargs)

        # Use FlextDecorators utility to preserve metadata
        return FlextValidationDecorators.preserve_metadata(func, wrapper)

    @custom_decorator
    def documented_function(x: int, y: int) -> int:
        """Add two numbers together.

        Args:
            x: First number
            y: Second number

        Returns:
            Sum of the two numbers

        """
        return x + y

    result = documented_function(5, 3)
    print(f"✅ Function result: {result}")
    print(f"   Function name: {documented_function.__name__}")
    print(f"   Function doc: {documented_function.__doc__}")

    # 2. Decorator factory patterns
    print("\n2. Decorator factory patterns:")

    # Custom validation decorator factory
    def create_range_validator(min_val: int, max_val: int):
        """Create a range validation decorator."""

        def range_validator(value: int) -> bool:
            return min_val <= value <= max_val

        return FlextValidationDecorators.create_validation_decorator(range_validator)

    # Use the factory to create specific decorators
    age_validator = create_range_validator(0, 150)
    percentage_validator = create_range_validator(0, 100)

    @age_validator
    def set_user_age(age: int) -> str:
        """Set user age with validation."""
        return f"Age set to {age}"

    @percentage_validator
    def set_completion_rate(rate: int) -> str:
        """Set completion rate with validation."""
        return f"Completion rate set to {rate}%"

    # Test range validators
    try:
        result = set_user_age(25)
        print(f"✅ Valid age: {result}")
    except Exception as e:
        print(f"❌ Invalid age: {e}")

    try:
        result = set_completion_rate(85)
        print(f"✅ Valid rate: {result}")
    except Exception as e:
        print(f"❌ Invalid rate: {e}")

    # 3. Performance monitoring with decorators
    print("\n3. Performance monitoring patterns:")

    execution_times = []

    def performance_monitor(func):
        """Monitor function performance."""

        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                execution_times.append(execution_time)
                print(
                    f"   {func.__name__} executed in {execution_time:.4f}s "
                    f"(avg: {sum(execution_times) / len(execution_times):.4f}s)",
                )
                return result
            except Exception:
                execution_time = time.time() - start_time
                print(f"   {func.__name__} failed in {execution_time:.4f}s")
                raise

        return wrapper

    @performance_monitor
    def monitored_operation(complexity: int) -> int:
        """Operation with performance monitoring."""
        time.sleep(complexity * 0.01)  # Simulate work
        return complexity * 100

    # Test performance monitoring
    for i in range(1, 4):
        result = monitored_operation(i)
        print(f"   Result: {result}")


def main() -> None:
    """Execute all FlextDecorators demonstrations."""
    print("🚀 FLEXT DECORATORS - ENTERPRISE PATTERNS EXAMPLE")
    print("Demonstrating comprehensive decorator patterns for enterprise applications")

    try:
        demonstrate_validation_decorators()
        demonstrate_error_handling_decorators()
        demonstrate_performance_decorators()
        demonstrate_logging_decorators()
        demonstrate_immutability_decorators()
        demonstrate_functional_decorators()
        demonstrate_decorator_best_practices()

        print("\n" + "=" * 80)
        print("✅ ALL FLEXT DECORATORS DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\n📊 Summary of patterns demonstrated:")
        print("   ✅ Validation decorators with automatic argument checking")
        print("   🛡️ Error handling decorators with retry and safe execution")
        print("   ⚡ Performance decorators with timing, caching, and memoization")
        print("   📝 Logging decorators with structured call and exception logging")
        print("   🔒 Immutability decorators with data protection patterns")
        print("   🔧 Functional composition and enterprise orchestration")
        print("   🏆 Advanced patterns and best practices")
        print("\n💡 FlextDecorators provides enterprise-grade decorator patterns")
        print(
            "   with validation, error handling, performance optimization, and observability!",
        )

    except Exception as e:
        print(f"\n❌ Error during FlextDecorators demonstration: {e}")

        traceback.print_exc()


if __name__ == "__main__":
    main()
