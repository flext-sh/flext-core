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
from typing import Any

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
    FlextTypes,
    FlextValidationDecorators,
    TAnyObject,
    TErrorMessage,
    TLogMessage,
    TUserData,
)

# Constants to avoid magic numbers
MAX_RETRY_ATTEMPTS = 3  # Maximum number of retry attempts

# =============================================================================
# VALIDATION CONSTANTS - Decorator validation constraints
# =============================================================================

# Age validation constants
MAX_REASONABLE_AGE = 150  # Maximum reasonable age for validation

# Retry constants
MAX_RETRY_ATTEMPTS = 3  # Maximum number of retry attempts before success


def demonstrate_validation_decorators() -> None:  # noqa: PLR0915
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

    @FlextValidationDecorators.validate_arguments
    def validate_user_data(name: str, email: str, age: int) -> TUserData:
        """Validate user data using flext_core.types."""
        return {"name": name, "email": email, "age": age, "status": "validated"}

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

    def email_validator(email: str) -> bool:
        """Validate simple email."""
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
        log_message = f"✅ Valid email registration: {result}"
        print(log_message)
    except (RuntimeError, ValueError, TypeError) as e:
        error_message = f"Email validation failed: {e}"
        print(f"❌ {error_message}")

    # Test invalid email
    try:
        result = register_user("invalid-email")
        log_message = f"✅ Invalid email registration: {result}"
        print(log_message)
    except (RuntimeError, ValueError, TypeError) as e:
        error_message = f"Email validation failed (expected): {e}"
        print(f"❌ {error_message}")

    # 3. Type-based validation
    log_message = "\n3. Type-based validation:"
    print(log_message)

    def validate_age(age: int) -> bool:
        """Validate age using flext_core.types."""
        return (
            FlextTypes.TypeGuards.is_instance_of(age, int)
            and 0 <= age <= MAX_REASONABLE_AGE
        )

    @FlextValidationDecorators.create_validation_decorator(validate_age)
    def create_user_profile(name: str, age: int) -> TUserData:
        """Create user profile with age validation."""
        return {"name": name, "age": age, "profile_created": True}

    # Test valid age
    try:
        result = create_user_profile("Alice", 25)
        log_message = f"✅ Valid age profile: {result}"
        print(log_message)
    except (RuntimeError, ValueError, TypeError) as e:
        error_message = f"Age validation failed: {e}"
        print(f"❌ {error_message}")

    # Test invalid age
    try:
        result = create_user_profile("Bob", 200)  # Too old
        log_message = f"✅ Invalid age profile: {result}"
        print(log_message)
    except (RuntimeError, ValueError, TypeError) as e:
        error_message = f"Age validation failed (expected): {e}"
        print(f"❌ {error_message}")


def demonstrate_error_handling_decorators() -> None:  # noqa: PLR0915
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
    if result.is_success:
        log_message = f"✅ Safe division result: {result.data}"
        print(log_message)
    else:
        error_message = f"Division failed: {result.error}"
        print(f"❌ {error_message}")

    # Test failed operation
    result = risky_division(10.0, 0.0)
    if result.is_success:
        log_message = f"✅ Safe division result: {result.data}"
        print(log_message)
    else:
        error_message = f"Division failed (expected): {result.error}"
        print(f"❌ {error_message}")

    # 2. Retry decorator
    log_message = "\n2. Retry decorator:"
    print(log_message)

    attempt_count = 0

    @FlextErrorHandlingDecorators.retry_decorator
    def unreliable_service() -> str:
        """Unreliable service that fails initially."""
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < MAX_RETRY_ATTEMPTS:
            msg = f"Service failed on attempt {attempt_count}"
            raise RuntimeError(msg)
        return "Service succeeded after retries"

    try:
        result = unreliable_service()
        log_message = f"✅ Retry service result: {result}"
        print(log_message)
        log_message = f"   Attempts made: {attempt_count}"
        print(log_message)
    except (RuntimeError, ValueError, TypeError) as e:
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

    @custom_safe_decorator
    def operation_with_custom_handling() -> str:
        """Operation with custom error handling."""
        msg = "Intentional error for testing"
        raise ValueError(msg)

    try:
        result = operation_with_custom_handling()
        log_message = f"✅ Custom handling result: {result}"
        print(log_message)
    except (RuntimeError, ValueError, TypeError) as e:
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

    @timing_decorator
    def slow_computation(n: int) -> int:
        """Slow computation for timing demonstration."""
        time.sleep(0.01)  # Simulate slow operation
        return sum(i for i in range(n))

    result = slow_computation(1000)
    log_message = f"✅ Computation result: {result}"
    print(log_message)

    # 2. Memoization decorator
    log_message = "\n2. Memoization decorator:"
    print(log_message)

    @FlextPerformanceDecorators.memoize_decorator
    def expensive_fibonacci(n: int) -> int:
        """Expensive Fibonacci calculation with memoization."""
        if n <= 1:
            return n
        return expensive_fibonacci(n - 1) + expensive_fibonacci(n - 2)

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

    @cache_decorator
    def data_processor(data_id: str) -> str:
        """Process data with caching."""
        time.sleep(0.01)  # Simulate processing
        return f"Processed data: {data_id}"

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

    @FlextLoggingDecorators.log_calls_decorator
    def business_operation(operation_type: str, amount: float) -> TUserData:
        """Execute business operation with call logging."""
        return {
            "operation": operation_type,
            "amount": amount,
            "status": "completed",
            "timestamp": time.time(),
        }

    result = business_operation("payment", 100.50)
    log_message = f"✅ Business operation result: {result}"
    print(log_message)

    # 2. Exception logging decorator
    log_message = "\n2. Exception logging decorator:"
    print(log_message)

    @FlextLoggingDecorators.log_exceptions_decorator
    def risky_business_operation(operation_id: str) -> str:
        """Risky business operation with exception logging."""
        if operation_id == "fail":
            msg = "Operation failed intentionally"
            raise RuntimeError(msg)
        return f"Operation {operation_id} completed successfully"

    # Test successful operation
    try:
        result = risky_business_operation("success_001")
        log_message = f"✅ Successful operation: {result}"
        print(log_message)
    except (RuntimeError, ValueError, TypeError) as e:
        error_message = f"Operation failed: {e}"
        print(f"❌ {error_message}")

    # Test failed operation
    try:
        result = risky_business_operation("fail")
        log_message = f"✅ Failed operation: {result}"
        print(log_message)
    except (RuntimeError, ValueError, TypeError) as e:
        error_message = f"Operation failed (expected): {e}"
        print(f"❌ {error_message}")

    # 3. Combined logging decorators
    log_message = "\n3. Combined logging decorators:"
    print(log_message)

    @FlextLoggingDecorators.log_calls_decorator
    @FlextLoggingDecorators.log_exceptions_decorator
    def comprehensive_service(
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

    # Test successful service
    result = comprehensive_service("test_service", {"param1": "value1"})
    log_message = f"✅ Comprehensive service result: {result}"
    print(log_message)

    # Test failed service
    try:
        result = comprehensive_service("error_service", {"param1": "value1"})
        log_message = f"✅ Error service result: {result}"
        print(log_message)
    except (RuntimeError, ValueError, TypeError) as e:
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

    @FlextImmutabilityDecorators.immutable_decorator
    def create_immutable_config() -> TUserData:
        """Create immutable configuration."""
        return {
            "database_url": "postgresql://localhost:5432/mydb",
            "api_key": "secret-key",
            "timeout": 30,
        }

    config = create_immutable_config()
    log_message = f"✅ Immutable config created: {config}"
    print(log_message)

    # Try to modify (should fail)
    try:
        config["new_key"] = "new_value"
        log_message = f"✅ Config modified: {config}"
        print(log_message)
    except (RuntimeError, ValueError, TypeError) as e:
        error_message = f"Config modification failed (expected): {e}"
        print(f"❌ {error_message}")

    # 2. Frozen arguments decorator
    log_message = "\n2. Frozen arguments decorator:"
    print(log_message)

    @FlextImmutabilityDecorators.freeze_args_decorator
    def process_user_data(user_data: TUserData) -> TUserData:
        """Process user data with frozen arguments."""
        # Try to modify input (should fail)
        try:
            user_data["processed"] = True
        except (RuntimeError, ValueError, TypeError) as e:
            error_message = f"Input modification failed (expected): {e}"
            print(f"❌ {error_message}")

        return {"original": user_data, "processed": True}

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

    @FlextLoggingDecorators.log_calls_decorator
    @FlextErrorHandlingDecorators.retry_decorator
    @FlextValidationDecorators.validate_arguments
    @FlextPerformanceDecorators.memoize_decorator
    def enterprise_user_service(
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

    # Test successful service
    result = enterprise_user_service("user_001", include_profile=True)
    log_message = f"✅ Enterprise service result: {result}"
    print(log_message)

    # Test failed service
    try:
        result = enterprise_user_service("invalid", include_profile=False)
        log_message = f"✅ Invalid service result: {result}"
        print(log_message)
    except (RuntimeError, ValueError, TypeError) as e:
        error_message = f"Invalid service failed (expected): {e}"
        print(f"❌ {error_message}")

    # 2. Pipeline decorators
    log_message = "\n2. Pipeline decorators:"
    print(log_message)

    @FlextPerformanceDecorators.get_timing_decorator()
    def step1_validate(data: TUserData) -> TUserData:
        """Step 1: Validate data."""
        return {**data, "validated": True}

    @FlextPerformanceDecorators.get_timing_decorator()
    def step2_enrich(data: TUserData) -> TUserData:
        """Step 2: Enrich data."""
        return {**data, "enriched": True, "timestamp": time.time()}

    @FlextPerformanceDecorators.get_timing_decorator()
    def step3_transform(data: TAnyObject) -> TAnyObject:
        """Step 3: Transform data."""
        return {**data, "transformed": True, "final": True}

    # Execute pipeline
    pipeline_data: TUserData = {"input": "test_data"}

    step1_result = step1_validate(pipeline_data)
    step2_result = step2_enrich(step1_result)
    step3_result = step3_transform(step2_result)

    log_message = f"✅ Pipeline result: {step3_result}"
    print(log_message)


def demonstrate_decorator_best_practices() -> None:  # noqa: PLR0915
    """Demonstrate decorator best practices using flext_core.types."""
    log_message: TLogMessage = "\n" + "=" * 80
    print(log_message)
    print("📚 DECORATOR BEST PRACTICES")
    print("=" * 80)

    # 1. Custom decorator with proper typing
    log_message = "\n1. Custom decorator with proper typing:"
    print(log_message)

    def custom_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        """Apply custom decorator with proper typing."""

        def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            log_message = f"🔧 Custom decorator called: {func.__name__}"
            print(log_message)
            return func(*args, **kwargs)

        return wrapper

    @custom_decorator
    def documented_function(x: int, y: int) -> int:
        """Documented function with custom decorator."""
        return x + y

    result = documented_function(5, 3)
    log_message = f"✅ Custom decorator result: {result}"
    print(log_message)

    # 2. Parameterized decorators
    log_message = "\n2. Parameterized decorators:"
    print(log_message)

    def create_range_validator(
        min_val: int,
        max_val: int,
    ) -> Callable[[int], bool]:
        """Create range validator using flext_core.types."""

        def range_validator(value: int) -> bool:
            return (
                FlextTypes.TypeGuards.is_instance_of(value, int)
                and min_val <= value <= max_val
            )

        return range_validator

    age_validator = FlextValidationDecorators.create_validation_decorator(
        create_range_validator(0, 150),
    )
    percentage_validator = FlextValidationDecorators.create_validation_decorator(
        create_range_validator(0, 100),
    )

    @age_validator
    def set_user_age(age: int) -> str:
        """Set user age with validation."""
        return f"User age set to {age}"

    @percentage_validator
    def set_completion_rate(rate: int) -> str:
        """Set completion rate with validation."""
        return f"Completion rate set to {rate}%"

    # Test valid values
    try:
        result1 = set_user_age(25)
        log_message = f"✅ Valid age: {result1}"
        print(log_message)

        result2 = set_completion_rate(75)
        log_message = f"✅ Valid rate: {result2}"
        print(log_message)
    except (RuntimeError, ValueError, TypeError) as e:
        error_message = f"Validation failed: {e}"
        print(f"❌ {error_message}")

    # Test invalid values
    try:
        result1 = set_user_age(200)  # Invalid age
        log_message = f"✅ Invalid age: {result1}"
        print(log_message)
    except (RuntimeError, ValueError, TypeError) as e:
        error_message = f"Age validation failed (expected): {e}"
        print(f"❌ {error_message}")

    # 3. Performance monitoring decorator
    log_message = "\n3. Performance monitoring decorator:"
    print(log_message)

    def performance_monitor(func: Callable[..., Any]) -> Callable[..., Any]:
        """Monitor performance with decorator."""

        def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()

            execution_time = end_time - start_time
            log_message = f"⏱️ {func.__name__} executed in {execution_time:.4f}s"
            print(log_message)

            return result

        return wrapper

    @performance_monitor
    def monitored_operation(complexity: int) -> int:
        """Execute monitored operation."""
        time.sleep(0.01 * complexity)  # Simulate work
        return complexity * 2

    result = monitored_operation(5)
    log_message = f"✅ Monitored operation result: {result}"
    print(log_message)


def demonstrate_domain_model_decorators() -> None:  # noqa: PLR0915
    """Demonstrate decorators with shared domain models integration."""
    log_message: TLogMessage = "\n" + "=" * 80
    print(log_message)
    print("🏢 DOMAIN MODEL DECORATORS INTEGRATION")
    print("=" * 80)

    # 1. Decorator with domain model validation
    log_message = "\n1. Decorator with domain model validation:"
    print(log_message)

    @FlextValidationDecorators.validate_arguments
    @FlextLoggingDecorators.log_calls_decorator
    def create_validated_user(name: str, email: str, age: int) -> SharedUser:
        """Create validated user using shared domain models with decorators."""
        user_result = SharedDomainFactory.create_user(name, email, age)
        if user_result.is_failure:
            error_message = f"User creation failed: {user_result.error}"
            raise ValueError(error_message)

        user = user_result.data
        log_domain_operation(
            "user_created_with_decorators",
            "SharedUser",
            user.id,
            decorator_stack="validation+logging",
        )
        return user

    # Test valid user creation
    try:
        user = create_validated_user("Alice Johnson", "alice@example.com", 28)
        log_message = f"✅ User created with decorators: {user.name} (ID: {user.id})"
        print(log_message)
    except (RuntimeError, ValueError, TypeError) as e:
        error_message: TErrorMessage = f"User creation failed: {e}"
        print(f"❌ {error_message}")

    # Test invalid user creation
    try:
        user = create_validated_user("", "invalid-email", 15)  # Invalid data
        log_message = f"✅ Invalid user created: {user.name}"
        print(log_message)
    except (RuntimeError, ValueError, TypeError) as e:
        error_message = f"Invalid user creation failed (expected): {e}"
        print(f"❌ {error_message}")

    # 2. Performance decorators with domain operations
    log_message = "\n2. Performance decorators with domain operations:"
    print(log_message)

    @FlextPerformanceDecorators.memoize_decorator
    @FlextPerformanceDecorators.get_timing_decorator()
    def find_user_by_email(email: str) -> SharedUser | None:
        """Find user by email with memoization and timing."""
        # Simulate database lookup
        time.sleep(0.01)

        # Create test user for demonstration
        if email == "cached@example.com":
            user_result = SharedDomainFactory.create_user("Cached User", email, 25)
            if user_result.is_success:
                return user_result.data

        return None

    # First lookup (expensive)
    user1 = find_user_by_email("cached@example.com")
    if user1:
        log_message = f"✅ First lookup: {user1.name} ({user1.email_address.email})"
        print(log_message)

    # Second lookup (cached)
    user2 = find_user_by_email("cached@example.com")
    if user2:
        log_message = f"✅ Second lookup (cached): {user2.name}"
        print(log_message)

    # 3. Error handling with domain operations
    log_message = "\n3. Error handling with domain operations:"
    print(log_message)

    @FlextDecorators.safe_result
    @FlextLoggingDecorators.log_exceptions_decorator
    def activate_user_account(user: SharedUser) -> SharedUser:
        """Activate user account with error handling decorators."""
        if user.status.value == "active":
            error_message = "User is already active"
            raise ValueError(error_message)

        activation_result = user.activate()
        if activation_result.is_failure:
            error_message = f"Activation failed: {activation_result.error}"
            raise RuntimeError(error_message)

        activated_user = activation_result.data
        log_domain_operation(
            "user_activated_with_decorators",
            "SharedUser",
            activated_user.id,
            old_status=user.status.value,
            new_status=activated_user.status.value,
        )
        return activated_user

    # Test user activation
    test_user_result = SharedDomainFactory.create_user(
        "Test User",
        "test@example.com",
        30,
    )
    if test_user_result.is_success:
        test_user = test_user_result.data

        # Activate user
        activation_result = activate_user_account(test_user)
        if activation_result.is_success:
            activated_user = activation_result.data
            log_message = (
                f"✅ User activated: {activated_user.name} "
                f"(Status: {activated_user.status.value})"
            )
            print(log_message)
        else:
            error_message = f"Activation failed: {activation_result.error}"
            print(f"❌ {error_message}")

        # Try to activate again (should fail)
        if activation_result.is_success:
            second_activation = activate_user_account(activation_result.data)
            if second_activation.is_success:
                log_message = f"✅ Second activation: {second_activation.data.name}"
                print(log_message)
            else:
                error_message = (
                    f"Second activation failed (expected): {second_activation.error}"
                )
                print(f"❌ {error_message}")

    # 4. Domain-aware validation decorators
    log_message = "\n4. Domain-aware validation decorators:"
    print(log_message)

    def domain_user_validator(user_data: dict[str, object]) -> bool:
        """Validate user data using domain models."""
        if not isinstance(user_data, dict):
            return False

        name = user_data.get("name", "")
        email = user_data.get("email", "")
        age = user_data.get("age", 0)

        # Use domain factory for validation
        user_result = SharedDomainFactory.create_user(name, email, age)
        return user_result.is_success

    domain_validator = FlextValidationDecorators.create_validation_decorator(
        domain_user_validator,
    )

    @domain_validator
    def register_user_with_domain_validation(
        user_data: dict[str, object],
    ) -> SharedUser:
        """Register user with domain-aware validation."""
        name = user_data.get("name", "")
        email = user_data.get("email", "")
        age = user_data.get("age", 0)

        user_result = SharedDomainFactory.create_user(name, email, age)
        if user_result.is_failure:
            error_message = f"Registration failed: {user_result.error}"
            raise ValueError(error_message)

        user = user_result.data
        log_domain_operation(
            "user_registered_with_domain_validation",
            "SharedUser",
            user.id,
            validation_method="domain_aware",
        )
        return user

    # Test valid registration
    try:
        valid_data = {"name": "Domain User", "email": "domain@example.com", "age": 32}
        user = register_user_with_domain_validation(valid_data)
        log_message = f"✅ Domain validation passed: {user.name}"
        print(log_message)
    except (RuntimeError, ValueError, TypeError) as e:
        error_message = f"Registration failed: {e}"
        print(f"❌ {error_message}")

    # Test invalid registration
    try:
        invalid_data = {
            "name": "",
            "email": "invalid",
            "age": 10,
        }  # Invalid per domain rules
        user = register_user_with_domain_validation(invalid_data)
        log_message = f"✅ Invalid registration: {user.name}"
        print(log_message)
    except (RuntimeError, ValueError, TypeError) as e:
        error_message = f"Invalid registration failed (expected): {e}"
        print(f"❌ {error_message}")

    log_message = "📊 Domain model decorators demonstration completed"
    print(log_message)


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
