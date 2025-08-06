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
    - Maximum type safety using flext_core.types

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
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from flext_core._decorators_base import _DecoratedFunction

# Import shared domain models to demonstrate decorator patterns with domain objects
from shared_domain import (
    SharedDomainFactory,
    User as SharedUser,
    log_domain_operation,
)

from flext_core import (
    FlextDecorators,
    FlextErrorHandlingDecorators,
    FlextImmutabilityDecorators,
    FlextLoggingDecorators,
    FlextPerformanceDecorators,
    FlextResult,
    FlextValidationDecorators,
    TAnyObject,
    TErrorMessage,
    TLogMessage,
    TUserData,
)
from flext_core.exceptions import FlextValidationError

# Constants to avoid magic numbers
MAX_RETRY_ATTEMPTS = 3  # Maximum number of retry attempts

# =============================================================================
# VALIDATION CONSTANTS - Decorator validation constraints
# =============================================================================

# Age validation constants
MAX_REASONABLE_AGE = 150  # Maximum reasonable age for validation

# Retry constants
MAX_RETRY_ATTEMPTS = 3  # Maximum number of retry attempts before success


def demonstrate_validation_decorators() -> None:
    """Demonstrate validation decorators with automatic argument checking.

    Using flext_core.types for type safety.
    """
    log_message: TLogMessage = "\n" + "=" * 80
    print(log_message)
    print("✅ VALIDATION DECORATORS")
    print("=" * 80)

    # 1. Basic argument validation
    log_message = "\n1. Basic argument validation:"
    print(log_message)

    # Apply validation decorator with proper casting
    def _validate_user_data_impl(name: str, email: str, age: int) -> TUserData:
        """Validate user data using flext_core.types."""
        return {"name": name, "email": email, "age": age, "status": "validated"}

    validate_user_data = FlextValidationDecorators.validate_arguments(
        cast("_DecoratedFunction", _validate_user_data_impl)
    )

    try:
        result = validate_user_data("John Doe", "john@example.com", 30)
        log_message = f"✅ Valid arguments: {result}"
        print(log_message)
    except (TypeError, ValueError) as e:
        error_message: TErrorMessage = f"Validation failed: {e}"
        print(f"❌ {error_message}")

    # 2. Custom validation decorator
    log_message = "\n2. Custom validation decorator:"
    print(log_message)

    def email_validator(email: object) -> bool:
        """Validate simple email."""
        return isinstance(email, str) and "@" in email and "." in email

    email_validation_decorator = FlextValidationDecorators.create_validation_decorator(
        email_validator,
    )

    def _register_user_impl(email: str) -> str:
        """Register user with email validation."""
        return f"User registered with email: {email}"

    register_user = email_validation_decorator(
        cast("_DecoratedFunction", _register_user_impl)
    )

    # Test valid email
    try:
        result = register_user("valid@example.com")
        log_message = f"✅ Valid email registration: {result}"
        print(log_message)
    except (RuntimeError, ValueError, TypeError, FlextValidationError) as e:
        error_message = f"Email validation failed: {e}"
        print(f"❌ {error_message}")

    # Test invalid email
    try:
        result = register_user("invalid-email")
        log_message = f"✅ Invalid email registration: {result}"
        print(log_message)
    except (RuntimeError, ValueError, TypeError, FlextValidationError) as e:
        error_message = f"Email validation failed (expected): {e}"
        print(f"❌ {error_message}")

    # 3. Type-based validation
    log_message = "\n3. Type-based validation:"
    print(log_message)

    def validate_age(age: object) -> bool:
        """Validate age using flext_core.types."""
        return isinstance(age, int) and 0 <= age <= MAX_REASONABLE_AGE

    def _create_user_profile_impl(name: str, age: int) -> TUserData:
        """Create user profile with age validation."""
        return {"name": name, "age": age, "profile_created": True}

    create_user_profile = FlextValidationDecorators.create_validation_decorator(
        validate_age
    )(cast("_DecoratedFunction", _create_user_profile_impl))

    # Test valid age
    try:
        result = create_user_profile("Alice", 25)
        log_message = f"✅ Valid age profile: {result}"
        print(log_message)
    except (RuntimeError, ValueError, TypeError, FlextValidationError) as e:
        error_message = f"Age validation failed: {e}"
        print(f"❌ {error_message}")

    # Test invalid age
    try:
        result = create_user_profile("Bob", 200)  # Too old
        log_message = f"✅ Invalid age profile: {result}"
        print(log_message)
    except (RuntimeError, ValueError, TypeError, FlextValidationError) as e:
        error_message = f"Age validation failed (expected): {e}"
        print(f"❌ {error_message}")


def demonstrate_error_handling_decorators() -> None:
    """Demonstrate error handling decorators with automatic exception management.

    Using flext_core.types for type safety.
    """
    log_message: TLogMessage = "\n" + "=" * 80
    print(log_message)
    print("🛡️ ERROR HANDLING DECORATORS")
    print("=" * 80)

    # 1. Safe result decorator
    log_message = "\n1. Safe result decorator:"
    print(log_message)

    @FlextDecorators.safe_result
    def risky_division(a: float, b: float) -> float:
        """Risky division operation."""
        if b == 0:
            msg = "Division by zero"
            raise ValueError(msg)
        return a / b

    # Test successful operation
    result = risky_division(10.0, 2.0)
    if hasattr(result, "success") and getattr(result, "success", False):
        log_message = f"✅ Safe division result: {getattr(result, 'data', 'N/A')}"
        print(log_message)
    else:
        error_message = f"Division failed: {getattr(result, 'error', 'Unknown error')}"
        print(f"❌ {error_message}")

    # Test failed operation
    result = risky_division(10.0, 0.0)
    if hasattr(result, "success") and getattr(result, "success", False):
        log_message = f"✅ Safe division result: {getattr(result, 'data', 'N/A')}"
        print(log_message)
    else:
        error_message = (
            f"Division failed (expected): {getattr(result, 'error', 'Unknown error')}"
        )
        print(f"❌ {error_message}")

    # 2. Retry decorator
    log_message = "\n2. Retry decorator:"
    print(log_message)

    attempt_count = 0

    def _unreliable_service_impl() -> str:
        """Unreliable service that fails initially."""
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < MAX_RETRY_ATTEMPTS:
            msg: str = f"Service failed on attempt {attempt_count}"
            raise RuntimeError(msg)
        return "Service succeeded after retries"

    unreliable_service = FlextErrorHandlingDecorators.retry_decorator(
        cast("_DecoratedFunction", _unreliable_service_impl)
    )

    try:
        service_result = cast("str", unreliable_service())
        log_message = f"✅ Retry service result: {service_result}"
        print(log_message)
        log_message = f"   Attempts made: {attempt_count}"
        print(log_message)
    except (RuntimeError, ValueError, TypeError, FlextValidationError) as e:
        error_message = f"Retry service failed: {e}"
        print(f"❌ {error_message}")

    # 3. Custom error handling
    log_message = "\n3. Custom error handling:"
    print(log_message)

    def custom_error_handler(exception: Exception) -> str:
        """Handle custom errors using flext_core.types."""
        error_message: TErrorMessage = (
            f"Custom handler caught: {type(exception).__name__}"
        )
        log_message = f"🛡️ {error_message}"
        print(log_message)
        return "Recovered from error"

    custom_safe_decorator = FlextErrorHandlingDecorators.create_safe_decorator(
        custom_error_handler,
    )

    def _operation_with_custom_handling_impl() -> str:
        """Operation with custom error handling."""
        msg = "Intentional error for testing"
        raise ValueError(msg)

    operation_with_custom_handling = custom_safe_decorator(
        cast("_DecoratedFunction", _operation_with_custom_handling_impl)
    )

    try:
        operation_result = cast("str", operation_with_custom_handling())
        log_message = f"✅ Custom handling result: {operation_result}"
        print(log_message)
    except (RuntimeError, ValueError, TypeError, FlextValidationError) as e:
        error_message = f"Custom handling failed: {e}"
        print(f"❌ {error_message}")


def demonstrate_performance_decorators() -> None:
    """Demonstrate performance decorators with timing and caching.

    Using flext_core.types for type safety.
    """
    log_message: TLogMessage = "\n" + "=" * 80
    print(log_message)
    print("⚡ PERFORMANCE DECORATORS")
    print("=" * 80)

    # 1. Timing decorator
    log_message = "\n1. Timing decorator:"
    print(log_message)

    timing_decorator = FlextPerformanceDecorators.get_timing_decorator()

    def _slow_computation_impl(n: int) -> int:
        """Slow computation for timing demonstration."""
        time.sleep(0.01)  # Simulate slow operation
        return sum(i for i in range(n))

    slow_computation = timing_decorator(
        cast("_DecoratedFunction", _slow_computation_impl)
    )

    result = slow_computation(1000)
    log_message = f"✅ Computation result: {result}"
    print(log_message)

    # 2. Memoization decorator
    log_message = "\n2. Memoization decorator:"
    print(log_message)

    def _expensive_fibonacci_impl(n: int) -> int:
        """Expensive Fibonacci calculation with memoization."""
        if n <= 1:
            return n
        return cast("int", expensive_fibonacci(n - 1)) + cast(
            "int", expensive_fibonacci(n - 2)
        )

    expensive_fibonacci = FlextPerformanceDecorators.memoize_decorator(
        cast("_DecoratedFunction", _expensive_fibonacci_impl)
    )

    # First call (expensive)
    start_time = time.time()
    result1 = expensive_fibonacci(10)
    first_call_time = time.time() - start_time

    # Second call (cached)
    start_time = time.time()
    result2 = expensive_fibonacci(10)
    second_call_time = time.time() - start_time

    log_message = f"✅ Fibonacci(10) = {result1}"
    print(log_message)
    log_message = f"   First call: {first_call_time:.4f}s"
    print(log_message)
    log_message = f"   Second call: {second_call_time:.4f}s"
    print(log_message)
    log_message = f"   Speedup: {first_call_time / second_call_time:.1f}x"
    print(log_message)

    # 3. Cache decorator
    log_message = "\n3. Cache decorator:"
    print(log_message)

    cache_decorator = FlextPerformanceDecorators.create_cache_decorator(max_size=10)

    def _data_processor_impl(data_id: str) -> str:
        """Process data with caching."""
        time.sleep(0.01)  # Simulate processing
        return f"Processed data: {data_id}"

    data_processor = cache_decorator(cast("_DecoratedFunction", _data_processor_impl))

    # First call
    result1 = data_processor("data_001")
    log_message = f"✅ First call: {result1}"
    print(log_message)

    # Second call (cached)
    result2 = data_processor("data_001")
    log_message = f"✅ Second call: {result2}"
    print(log_message)


def demonstrate_logging_decorators() -> None:
    """Demonstrate logging decorators with structured logging using flext_core.types."""
    log_message: TLogMessage = "\n" + "=" * 80
    print(log_message)
    print("📝 LOGGING DECORATORS")
    print("=" * 80)

    # 1. Call logging decorator
    log_message = "\n1. Call logging decorator:"
    print(log_message)

    def _business_operation_impl(operation_type: str, amount: float) -> TUserData:
        """Execute business operation with call logging."""
        return {
            "operation": operation_type,
            "amount": amount,
            "status": "completed",
            "timestamp": time.time(),
        }

    business_operation = FlextLoggingDecorators.log_calls_decorator(
        cast("_DecoratedFunction", _business_operation_impl)
    )

    result = business_operation("payment", 100.50)
    log_message = f"✅ Business operation result: {result}"
    print(log_message)

    # 2. Exception logging decorator
    log_message = "\n2. Exception logging decorator:"
    print(log_message)

    def _risky_business_operation_impl(operation_id: str) -> str:
        """Risky business operation with exception logging."""
        if operation_id == "fail":
            msg = "Operation failed intentionally"
            raise RuntimeError(msg)
        return f"Operation {operation_id} completed successfully"

    risky_business_operation = FlextLoggingDecorators.log_exceptions_decorator(
        cast("_DecoratedFunction", _risky_business_operation_impl)
    )

    # Test successful operation
    try:
        result = risky_business_operation("success_001")
        log_message = f"✅ Successful operation: {result}"
        print(log_message)
    except (RuntimeError, ValueError, TypeError, FlextValidationError) as e:
        error_message = f"Operation failed: {e}"
        print(f"❌ {error_message}")

    # Test failed operation
    try:
        result = risky_business_operation("fail")
        log_message = f"✅ Failed operation: {result}"
        print(log_message)
    except (RuntimeError, ValueError, TypeError, FlextValidationError) as e:
        error_message = f"Operation failed (expected): {e}"
        print(f"❌ {error_message}")

    # 3. Combined logging decorators
    log_message = "\n3. Combined logging decorators:"
    print(log_message)

    def _comprehensive_service_impl(
        service_name: str,
        params: TUserData,
    ) -> TUserData:
        """Comprehensive service with both call and exception logging."""
        if service_name == "error_service":
            msg = "Service error for testing"
            raise ValueError(msg)
        return {
            "service": service_name,
            "params": params,
            "result": "success",
        }

    # Apply decorators in sequence
    comprehensive_service_with_exceptions = (
        FlextLoggingDecorators.log_exceptions_decorator(
            cast("_DecoratedFunction", _comprehensive_service_impl)
        )
    )
    comprehensive_service = FlextLoggingDecorators.log_calls_decorator(
        comprehensive_service_with_exceptions
    )

    # Test successful service
    result = comprehensive_service("test_service", {"param1": "value1"})
    log_message = f"✅ Comprehensive service result: {result}"
    print(log_message)

    # Test failed service
    try:
        result = comprehensive_service("error_service", {"param1": "value1"})
        log_message = f"✅ Error service result: {result}"
        print(log_message)
    except (RuntimeError, ValueError, TypeError, FlextValidationError) as e:
        error_message = f"Error service failed (expected): {e}"
        print(f"❌ {error_message}")


def demonstrate_immutability_decorators() -> None:
    """Demonstrate immutability decorators with data protection.

    Using flext_core.types for type safety.
    """
    log_message: TLogMessage = "\n" + "=" * 80
    print(log_message)
    print("🔒 IMMUTABILITY DECORATORS")
    print("=" * 80)

    # 1. Immutable result decorator
    log_message = "\n1. Immutable result decorator:"
    print(log_message)

    def _create_immutable_config_impl() -> TUserData:
        """Create immutable configuration."""
        return {
            "database_url": "postgresql://localhost:5432/mydb",
            "api_key": "secret-key",
            "timeout": 30,
        }

    create_immutable_config = FlextImmutabilityDecorators.immutable_decorator(
        cast("_DecoratedFunction", _create_immutable_config_impl)
    )

    config = create_immutable_config()
    log_message = f"✅ Immutable config created: {config}"
    print(log_message)

    # Try to modify (should fail)
    try:
        if isinstance(config, dict):
            config["new_key"] = "new_value"
        log_message = f"✅ Config modified: {config}"
        print(log_message)
    except (RuntimeError, ValueError, TypeError, FlextValidationError) as e:
        error_message = f"Config modification failed (expected): {e}"
        print(f"❌ {error_message}")

    # 2. Frozen arguments decorator
    log_message = "\n2. Frozen arguments decorator:"
    print(log_message)

    def _process_user_data_impl(user_data: TUserData) -> TUserData:
        """Process user data with frozen arguments."""
        # Try to modify input (should fail)
        try:
            user_data["processed"] = True
        except (RuntimeError, ValueError, TypeError, FlextValidationError) as e:
            error_message = f"Input modification failed (expected): {e}"
            print(f"❌ {error_message}")

        return {"original": user_data, "processed": True}

    process_user_data = FlextImmutabilityDecorators.freeze_args_decorator(
        cast("_DecoratedFunction", _process_user_data_impl)
    )

    input_data: TUserData = {"name": "Alice", "age": 30}
    result = process_user_data(input_data)
    log_message = f"✅ User data processed: {result}"
    print(log_message)


def demonstrate_functional_decorators() -> None:
    """Demonstrate functional decorators with composition using flext_core.types."""
    log_message: TLogMessage = "\n" + "=" * 80
    print(log_message)
    print("🔗 FUNCTIONAL DECORATORS")
    print("=" * 80)

    # 1. Decorator composition
    log_message = "\n1. Decorator composition:"
    print(log_message)

    def _enterprise_user_service_impl(
        user_id: str,
        *,
        include_profile: bool = True,
    ) -> TUserData:
        """Enterprise user service with multiple decorators."""
        # Simulate service logic
        if user_id == "invalid":
            msg = "Invalid user ID"
            raise ValueError(msg)

        result: TUserData = {
            "user_id": user_id,
            "name": f"User {user_id}",
            "email": f"user{user_id}@example.com",
        }

        if include_profile:
            result["profile"] = {"age": 25, "role": "user"}

        return result

    # Apply decorators in sequence
    step1 = FlextPerformanceDecorators.memoize_decorator(
        cast("_DecoratedFunction", _enterprise_user_service_impl)
    )
    step2 = FlextValidationDecorators.validate_arguments(step1)
    step3 = FlextErrorHandlingDecorators.retry_decorator(step2)
    enterprise_user_service = FlextLoggingDecorators.log_calls_decorator(step3)

    # Test successful service
    result = enterprise_user_service("user_001", include_profile=True)
    log_message = f"✅ Enterprise service result: {result}"
    print(log_message)

    # Test failed service
    try:
        result = enterprise_user_service("invalid", include_profile=False)
        log_message = f"✅ Invalid service result: {result}"
        print(log_message)
    except (RuntimeError, ValueError, TypeError, FlextValidationError) as e:
        error_message = f"Invalid service failed (expected): {e}"
        print(f"❌ {error_message}")

    # 2. Pipeline decorators
    log_message = "\n2. Pipeline decorators:"
    print(log_message)

    def _step1_validate_impl(data: TUserData) -> TUserData:
        """Step 1: Validate data."""
        return {**data, "validated": True}

    step1_validate = FlextPerformanceDecorators.get_timing_decorator()(
        cast("_DecoratedFunction", _step1_validate_impl)
    )

    def _step2_enrich_impl(data: TUserData) -> TUserData:
        """Step 2: Enrich data."""
        return {**data, "enriched": True, "timestamp": time.time()}

    step2_enrich = FlextPerformanceDecorators.get_timing_decorator()(
        cast("_DecoratedFunction", _step2_enrich_impl)
    )

    def _step3_transform_impl(data: TAnyObject) -> TAnyObject:
        """Step 3: Transform data."""
        return {**data, "transformed": True, "final": True}

    step3_transform = FlextPerformanceDecorators.get_timing_decorator()(
        cast("_DecoratedFunction", _step3_transform_impl)
    )

    # Execute pipeline
    pipeline_data: TUserData = {"input": "test_data"}

    step1_result = step1_validate(pipeline_data)
    step2_result = step2_enrich(step1_result)
    step3_result = step3_transform(step2_result)

    log_message = f"✅ Pipeline result: {step3_result}"
    print(log_message)


def demonstrate_decorator_best_practices() -> None:
    """Demonstrate decorator best practices using flext_core.types."""
    log_message: TLogMessage = "\n" + "=" * 80
    print(log_message)
    print("📚 DECORATOR BEST PRACTICES")
    print("=" * 80)

    # 1. Custom decorator with proper typing
    log_message = "\n1. Custom decorator with proper typing:"
    print(log_message)

    def custom_decorator(func: object) -> object:
        """Apply custom decorator with proper typing."""

        def wrapper(*args: object, **kwargs: object) -> object:
            func_name = getattr(func, "__name__", "unknown")
            log_message = f"🔧 Custom decorator called: {func_name}"
            print(log_message)
            if callable(func):
                return func(*args, **kwargs)
            msg = "Not callable"
            raise ValueError(msg)

        return wrapper

    def _documented_function_impl(x: int, y: int) -> int:
        """Documented function with custom decorator."""
        return x + y

    documented_function = custom_decorator(_documented_function_impl)

    result = documented_function(5, 3) if callable(documented_function) else 0
    log_message = f"✅ Custom decorator result: {result}"
    print(log_message)

    # 2. Parameterized decorators
    log_message = "\n2. Parameterized decorators:"
    print(log_message)

    def create_range_validator(
        min_val: int,
        max_val: int,
    ) -> Callable[[object], bool]:
        """Create range validator using flext_core.types."""

        def range_validator(value: object) -> bool:
            return isinstance(value, int) and min_val <= value <= max_val

        return range_validator

    age_validator = FlextValidationDecorators.create_validation_decorator(
        create_range_validator(0, 150),
    )
    percentage_validator = FlextValidationDecorators.create_validation_decorator(
        create_range_validator(0, 100),
    )

    def _set_user_age_impl(age: int) -> str:
        """Set user age with validation."""
        return f"User age set to {age}"

    set_user_age = age_validator(cast("_DecoratedFunction", _set_user_age_impl))

    def _set_completion_rate_impl(rate: int) -> str:
        """Set completion rate with validation."""
        return f"Completion rate set to {rate}%"

    set_completion_rate = percentage_validator(
        cast("_DecoratedFunction", _set_completion_rate_impl)
    )

    # Test valid values
    try:
        result1 = set_user_age(25)
        log_message = f"✅ Valid age: {result1}"
        print(log_message)

        result2 = set_completion_rate(75)
        log_message = f"✅ Valid rate: {result2}"
        print(log_message)
    except (RuntimeError, ValueError, TypeError, FlextValidationError) as e:
        error_message = f"Validation failed: {e}"
        print(f"❌ {error_message}")

    # Test invalid values
    try:
        result1 = set_user_age(200)  # Invalid age
        log_message = f"✅ Invalid age: {result1}"
        print(log_message)
    except (RuntimeError, ValueError, TypeError, FlextValidationError) as e:
        error_message = f"Age validation failed (expected): {e}"
        print(f"❌ {error_message}")

    # 3. Performance monitoring decorator
    log_message = "\n3. Performance monitoring decorator:"
    print(log_message)

    def performance_monitor(func: object) -> object:
        """Monitor performance with decorator."""

        def wrapper(*args: object, **kwargs: object) -> object:
            start_time = time.time()
            if callable(func):
                result = func(*args, **kwargs)
            else:
                msg = "Not callable"
                raise TypeError(msg)
            end_time = time.time()

            execution_time = end_time - start_time
            func_name = getattr(func, "__name__", "unknown")
            log_message = f"⏱️ {func_name} executed in {execution_time:.4f}s"
            print(log_message)

            return result

        return wrapper

    def _monitored_operation_impl(complexity: int) -> int:
        """Execute monitored operation."""
        time.sleep(0.01 * complexity)  # Simulate work
        return complexity * 2

    monitored_operation = performance_monitor(_monitored_operation_impl)

    result = monitored_operation(5) if callable(monitored_operation) else 0
    log_message = f"✅ Monitored operation result: {result}"
    print(log_message)


def demonstrate_domain_model_decorators() -> None:
    """Demonstrate decorators with shared domain models integration using
    railway-oriented programming.
    """
    _print_section_header("🏢 DOMAIN MODEL DECORATORS INTEGRATION")

    # Chain all demonstration patterns using single-responsibility methods
    (
        _demonstrate_validation_decorators()
        .flat_map(lambda _: _demonstrate_performance_decorators())
        .flat_map(lambda _: _demonstrate_error_handling_decorators())
        .flat_map(lambda _: _demonstrate_domain_validation_decorators())
        .flat_map(lambda _: _complete_domain_decorators_demo())
    )


def _print_section_header(title: str) -> None:
    """Print formatted section header."""
    log_message: TLogMessage = "\n" + "=" * 80
    print(log_message)
    print(title)
    print("=" * 80)


def _demonstrate_validation_decorators() -> FlextResult[None]:
    """Demonstrate validation decorators with domain models."""
    print("\n1. Decorator with domain model validation:")

    def _create_validated_user_impl(name: str, email: str, age: int) -> SharedUser:
        """Create validated user using shared domain models with decorators."""
        result = SharedDomainFactory.create_user(name, email, age).flat_map(
            lambda user: _log_user_creation(user, "validation+logging")
        )
        if result.success and result.data is not None:
            return result.data
        raise ValueError(result.error or "User creation failed")

    # Apply decorators in sequence
    create_validated_user_with_logging = FlextLoggingDecorators.log_calls_decorator(
        cast("_DecoratedFunction", _create_validated_user_impl)
    )
    create_validated_user = FlextValidationDecorators.validate_arguments(
        create_validated_user_with_logging
    )

    return _test_valid_user_creation(create_validated_user).flat_map(
        lambda _: _test_invalid_user_creation(create_validated_user)
    )


def _log_user_creation(
    user: SharedUser, decorator_stack: str
) -> FlextResult[SharedUser]:
    """Log user creation with decorator information."""
    log_domain_operation(
        "user_created_with_decorators",
        "SharedUser",
        user.id,
        decorator_stack=decorator_stack,
    )
    return FlextResult.ok(user)


def _test_valid_user_creation(create_user_func: object) -> FlextResult[None]:
    """Test valid user creation scenario."""
    try:
        if callable(create_user_func):
            user = create_user_func("Alice Johnson", "alice@example.com", 28)
            log_message = (
                f"✅ User created with decorators: {getattr(user, 'name', 'N/A')} "
                f"(ID: {getattr(user, 'id', 'N/A')})"
            )
            print(log_message)
            return FlextResult.ok(None)
        return FlextResult.fail("Invalid create_user_func provided")
    except (RuntimeError, ValueError, TypeError, FlextValidationError) as e:
        error_message: TErrorMessage = f"User creation failed: {e}"
        print(f"❌ {error_message}")
        return FlextResult.ok(None)  # Expected for demo


def _test_invalid_user_creation(create_user_func: object) -> FlextResult[None]:
    """Test invalid user creation scenario."""
    try:
        if callable(create_user_func):
            user = create_user_func("", "invalid-email", 15)  # Invalid data
            log_message = f"✅ Invalid user created: {getattr(user, 'name', 'N/A')}"
            print(log_message)
        return FlextResult.ok(None)
    except (RuntimeError, ValueError, TypeError, FlextValidationError) as e:
        error_message = f"Invalid user creation failed (expected): {e}"
        print(f"❌ {error_message}")
        return FlextResult.ok(None)  # Expected failure for demo


def _demonstrate_performance_decorators() -> FlextResult[None]:
    """Demonstrate performance decorators with domain operations."""
    print("\n2. Performance decorators with domain operations:")

    def _find_user_by_email_impl(email: str) -> SharedUser | None:
        """Find user by email with memoization and timing."""
        # Simulate database lookup
        time.sleep(0.01)

        # Create test user for demonstration
        if email == "cached@example.com":
            user_result = SharedDomainFactory.create_user("Cached User", email, 25)
            return user_result.data if user_result.success else None

        return None

    # Apply decorators in sequence
    find_user_with_timing = FlextPerformanceDecorators.get_timing_decorator()(
        cast("_DecoratedFunction", _find_user_by_email_impl)
    )
    find_user_by_email = FlextPerformanceDecorators.memoize_decorator(
        find_user_with_timing
    )

    return _execute_performance_lookups(find_user_by_email)


def _execute_performance_lookups(lookup_func: object) -> FlextResult[None]:
    """Execute performance lookup demonstrations."""
    if not callable(lookup_func):
        return FlextResult.fail("Invalid lookup function provided")

    # First lookup (expensive)
    user1 = lookup_func("cached@example.com")
    if user1:
        log_message = f"✅ First lookup: {getattr(user1, 'name', 'N/A')} ({getattr(getattr(user1, 'email_address', None), 'email', 'N/A') if hasattr(user1, 'email_address') else 'N/A'})"
        print(log_message)

    # Second lookup (cached)
    user2 = lookup_func("cached@example.com")
    if user2:
        log_message = f"✅ Second lookup (cached): {getattr(user2, 'name', 'N/A')}"
        print(log_message)

    return FlextResult.ok(None)


def _demonstrate_error_handling_decorators() -> FlextResult[None]:
    """Demonstrate error handling decorators with domain operations."""
    print("\n3. Error handling with domain operations:")

    def _activate_user_account_impl(user: SharedUser) -> SharedUser:
        """Activate user account with error handling decorators."""
        result = (
            _validate_user_not_active(user)
            .flat_map(lambda _: _perform_user_activation(user))
            .flat_map(lambda activated: _log_user_activation(user, activated))
        )
        if result.success and result.data is not None:
            return result.data
        raise ValueError(result.error or "User activation failed")

    # Apply decorators in sequence
    activate_user_with_exceptions = FlextLoggingDecorators.log_exceptions_decorator(
        cast("_DecoratedFunction", _activate_user_account_impl)
    )
    activate_user_account = FlextDecorators.safe_result(activate_user_with_exceptions)

    return _test_user_activation_scenarios(activate_user_account)


def _validate_user_not_active(user: SharedUser) -> FlextResult[None]:
    """Validate that user is not already active."""
    if user.status.value == "active":
        return FlextResult.fail("User is already active")
    return FlextResult.ok(None)


def _perform_user_activation(user: SharedUser) -> FlextResult[SharedUser]:
    """Perform user activation operation."""
    activation_result = user.activate()
    if activation_result.is_failure:
        return FlextResult.fail(f"Activation failed: {activation_result.error}")

    activated_user = activation_result.data
    if activated_user is not None:
        return FlextResult.ok(activated_user)
    return FlextResult.fail("User activation returned None data")


def _log_user_activation(
    user: SharedUser, activated_user: SharedUser
) -> FlextResult[SharedUser]:
    """Log user activation event."""
    log_domain_operation(
        "user_activated_with_decorators",
        "SharedUser",
        activated_user.id,
        old_status=user.status.value,
        new_status=activated_user.status.value,
    )
    return FlextResult.ok(activated_user)


def _test_user_activation_scenarios(activate_func: object) -> FlextResult[None]:
    """Test user activation scenarios."""
    test_user_result = SharedDomainFactory.create_user(
        "Test User", "test@example.com", 30
    )
    if not test_user_result.success:
        return FlextResult.fail("Failed to create test user")

    test_user = test_user_result.data
    if test_user is not None:
        return _test_successful_activation(activate_func, test_user).flat_map(
            lambda activated: _test_duplicate_activation(activate_func, activated)
        )
    return FlextResult.fail("Test user creation returned None")


def _test_successful_activation(
    activate_func: object, test_user: SharedUser
) -> FlextResult[SharedUser]:
    """Test successful user activation."""
    if not callable(activate_func):
        return FlextResult.fail("Invalid activation function")

    try:
        activated_user = activate_func(test_user)
        log_message = (
            f"✅ User activated: {getattr(activated_user, 'name', 'N/A')} "
            f"(Status: {getattr(getattr(activated_user, 'status', None), 'value', 'N/A') if hasattr(activated_user, 'status') else 'N/A'})"
        )
        print(log_message)
        return FlextResult.ok(activated_user)
    except Exception as e:
        error_message = f"Activation failed: {e}"
        print(f"❌ {error_message}")
        return FlextResult.fail(error_message)


def _test_duplicate_activation(
    activate_func: object, activated_user: SharedUser
) -> FlextResult[None]:
    """Test duplicate activation scenario (should fail)."""
    if not callable(activate_func):
        return FlextResult.ok(None)

    try:
        activate_func(activated_user)
        print("❌ Second activation should have failed")
    except Exception as e:
        error_message = f"Second activation failed (expected): {e}"
        print(f"❌ {error_message}")

    return FlextResult.ok(None)


def _demonstrate_domain_validation_decorators() -> FlextResult[None]:
    """Demonstrate domain-aware validation decorators."""
    print("\n4. Domain-aware validation decorators:")

    def domain_user_validator(user_data: object) -> bool:
        """Validate user data using domain models."""
        return (
            _validate_user_data_structure(user_data)
            .flat_map(_validate_with_domain_factory)
            .map(lambda result: result.success)
            .unwrap_or(default=False)
        )

    domain_validator = FlextValidationDecorators.create_validation_decorator(
        domain_user_validator
    )

    def _register_user_with_domain_validation_impl(
        user_data: dict[str, object],
    ) -> SharedUser:
        """Register user with domain-aware validation."""
        result = (
            _extract_user_data(user_data)
            .flat_map(_create_user_from_data)
            .flat_map(_log_user_registration)
        )
        if result.success and result.data is not None:
            return result.data
        raise ValueError(result.error or "User registration failed")

    register_user_with_domain_validation = domain_validator(
        cast("_DecoratedFunction", _register_user_with_domain_validation_impl)
    )

    return _test_domain_validation_scenarios(register_user_with_domain_validation)


def _validate_user_data_structure(user_data: object) -> FlextResult[dict[str, object]]:
    """Validate user data structure and types."""
    if not isinstance(user_data, dict):
        return FlextResult.fail("User data must be a dictionary")

    name = user_data.get("name", "")
    email = user_data.get("email", "")
    age = user_data.get("age", 0)

    if (
        not isinstance(name, str)
        or not isinstance(email, str)
        or not isinstance(age, int)
    ):
        return FlextResult.fail("Invalid data types in user data")

    return FlextResult.ok({"name": name, "email": email, "age": age})


def _validate_with_domain_factory(
    user_data: dict[str, object],
) -> FlextResult[FlextResult[SharedUser]]:
    """Validate user data with domain factory."""
    name = str(user_data["name"])
    email = str(user_data["email"])
    age = int(cast("int", user_data["age"]))

    user_result = SharedDomainFactory.create_user(name, email, age)
    return FlextResult.ok(user_result)


def _extract_user_data(
    user_data: dict[str, object],
) -> FlextResult[tuple[str, str, int]]:
    """Extract and validate user data fields."""
    name = user_data.get("name", "")
    email = user_data.get("email", "")
    age = user_data.get("age", 0)

    # Type validation and casting
    if not isinstance(name, str):
        return FlextResult.fail("Name must be a string")
    if not isinstance(email, str):
        return FlextResult.fail("Email must be a string")
    if not isinstance(age, int):
        return FlextResult.fail("Age must be an integer")

    return FlextResult.ok((name, email, age))


def _create_user_from_data(data: tuple[str, str, int]) -> FlextResult[SharedUser]:
    """Create user from validated data."""
    name, email, age = data
    user_result = SharedDomainFactory.create_user(name, email, age)
    if user_result.is_failure:
        return FlextResult.fail(f"Registration failed: {user_result.error}")

    user = user_result.data
    if user is not None:
        return FlextResult.ok(user)
    return FlextResult.fail("User registration returned None data")


def _log_user_registration(user: SharedUser) -> FlextResult[SharedUser]:
    """Log user registration event."""
    log_domain_operation(
        "user_registered_with_domain_validation",
        "SharedUser",
        user.id,
        validation_method="domain_aware",
    )
    return FlextResult.ok(user)


def _test_domain_validation_scenarios(register_func: object) -> FlextResult[None]:
    """Test domain validation scenarios."""
    return _test_valid_registration(register_func).flat_map(
        lambda _: _test_invalid_registration(register_func)
    )


def _test_valid_registration(register_func: object) -> FlextResult[None]:
    """Test valid user registration."""
    if not callable(register_func):
        return FlextResult.fail("Invalid registration function")

    try:
        valid_data = {"name": "Domain User", "email": "domain@example.com", "age": 32}
        user = register_func(valid_data)
        log_message = f"✅ Domain validation passed: {getattr(user, 'name', 'N/A')}"
        print(log_message)
        return FlextResult.ok(None)
    except (RuntimeError, ValueError, TypeError, FlextValidationError) as e:
        error_message = f"Registration failed: {e}"
        print(f"❌ {error_message}")
        return FlextResult.ok(None)  # Continue demo


def _test_invalid_registration(register_func: object) -> FlextResult[None]:
    """Test invalid user registration."""
    if not callable(register_func):
        return FlextResult.ok(None)

    try:
        invalid_data = {
            "name": "",
            "email": "invalid",
            "age": 10,
        }  # Invalid per domain rules
        user = register_func(invalid_data)
        log_message = f"✅ Invalid registration: {getattr(user, 'name', 'N/A')}"
        print(log_message)
    except (RuntimeError, ValueError, TypeError, FlextValidationError) as e:
        error_message = f"Invalid registration failed (expected): {e}"
        print(f"❌ {error_message}")

    return FlextResult.ok(None)


def _complete_domain_decorators_demo() -> FlextResult[None]:
    """Complete the domain decorators demonstration."""
    log_message = "📊 Domain model decorators demonstration completed"
    print(log_message)
    return FlextResult.ok(None)


def main() -> None:
    """Run comprehensive FlextDecorators demonstration with maximum type safety."""
    print("=" * 80)
    print("🚀 FLEXT DECORATORS - ENTERPRISE PATTERNS DEMONSTRATION")
    print("=" * 80)

    # Run all demonstrations
    demonstrate_validation_decorators()
    demonstrate_error_handling_decorators()
    demonstrate_performance_decorators()
    demonstrate_logging_decorators()
    demonstrate_immutability_decorators()
    demonstrate_functional_decorators()
    demonstrate_decorator_best_practices()
    demonstrate_domain_model_decorators()

    print("\n" + "=" * 80)
    print("🎉 FLEXT DECORATORS DEMONSTRATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
