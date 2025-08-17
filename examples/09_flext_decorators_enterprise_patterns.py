#!/usr/bin/env python3
"""Enterprise decorator patterns with FlextDecorators.

Demonstrates validation, error handling, performance optimization,
logging, and immutability enforcement decorators.
    - Performance decorators with timing, caching, and memoization
    - Logging decorators with structured call and exception logging
    - Immutability decorators with argument and result protection
    - Functional decorators with composition and orchestration
    - Complete decorator orchestration for enterprise applications
    - Maximum type safety using flext_core.typings

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

import contextlib
import time
from collections.abc import Callable
from typing import cast

from flext_core import (
    FlextDecorators,
    FlextErrorHandlingDecorators,
    FlextImmutabilityDecorators,
    FlextLoggingDecorators,
    FlextPerformanceDecorators,
    FlextResult,
    FlextValidationDecorators,
    FlextValidationError,
    TAnyObject,
    TUserData,
)

from .shared_domain import (
    SharedDomainFactory,
    User as SharedUser,
    log_domain_operation,
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


def _demonstrate_basic_argument_validation() -> None:
    """Demonstrate basic argument validation."""

    # Apply validation decorator with proper casting
    def _validate_user_data_impl(name: str, email: str, age: int) -> TUserData:
      """Validate user data using flext_core.typings."""
      return {"name": name, "email": email, "age": age, "status": "validated"}

    validate_user_data = FlextValidationDecorators.validate_arguments(
      _validate_user_data_impl,
    )

    with contextlib.suppress(TypeError, ValueError):
      validate_user_data("John Doe", "john@example.com", 30)


def demonstrate_validation_decorators() -> None:
    """Demonstrate validation decorators with automatic argument checking.

    Using flext_core.typings for type safety.
    """
    "\n" + "=" * 80

    _demonstrate_basic_argument_validation()

    # 2. Custom validation decorator

    def email_validator(email: object) -> bool:
      """Validate simple email."""
      return isinstance(email, str) and "@" in email and "." in email

    email_validation_decorator = FlextValidationDecorators.create_validation_decorator(
      email_validator,
    )

    def _register_user_impl(email: str) -> str:
      """Register user with email validation."""
      return f"User registered with email: {email}"

    register_user = email_validation_decorator(_register_user_impl)

    # Test valid email
    with contextlib.suppress(RuntimeError, ValueError, TypeError, FlextValidationError):
      register_user("valid@example.com")

    # Test invalid email
    with contextlib.suppress(RuntimeError, ValueError, TypeError, FlextValidationError):
      register_user("invalid-email")

    # 3. Type-based validation

    def validate_age(age: object) -> bool:
      """Validate age using flext_core.typings."""
      return isinstance(age, int) and 0 <= age <= MAX_REASONABLE_AGE

    def _create_user_profile_impl(name: str, age: int) -> TUserData:
      """Create user profile with age validation."""
      return {"name": name, "age": age, "profile_created": True}

    create_user_profile = FlextValidationDecorators.create_validation_decorator(
      validate_age,
    )(_create_user_profile_impl)

    # Test valid age
    with contextlib.suppress(RuntimeError, ValueError, TypeError, FlextValidationError):
      create_user_profile("Alice", 25)

    # Test invalid age
    with contextlib.suppress(RuntimeError, ValueError, TypeError, FlextValidationError):
      create_user_profile("Bob", 200)  # Too old


def demonstrate_error_handling_decorators() -> None:
    """Demonstrate error handling decorators with automatic exception management.

    Using flext_core.typings for type safety.
    """
    _print_error_header()
    _demo_safe_result_decorator()
    _demo_retry_decorator()
    _demo_custom_error_handler()


def _print_error_header() -> None:
    "\n" + "=" * 80


def _demo_safe_result_decorator() -> None:
    def _risky_division_impl(a: object, b: object) -> object:
      if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
          msg = "Arguments must be numbers"
          raise TypeError(msg)
      if b == 0:
          msg = "Division by zero"
          raise ValueError(msg)
      return float(a) / float(b)

    risky_division = FlextDecorators.safe_result(_risky_division_impl)

    result = risky_division(10.0, 2.0)
    if hasattr(result, "success") and getattr(result, "success", False):
      pass

    result = risky_division(10.0, 0.0)
    if hasattr(result, "success") and getattr(result, "success", False):
      pass


def _demo_retry_decorator() -> None:
    attempt_count = 0

    def _unreliable_service_impl() -> str:
      nonlocal attempt_count
      attempt_count += 1
      if attempt_count < MAX_RETRY_ATTEMPTS:
          msg = f"Service failed on attempt {attempt_count}"
          raise RuntimeError(msg)
      return "Service succeeded after retries"

    unreliable_service = FlextErrorHandlingDecorators.retry_decorator(
      _unreliable_service_impl,
    )

    with contextlib.suppress(RuntimeError, ValueError, TypeError, FlextValidationError):
      cast("str", unreliable_service())


def _demo_custom_error_handler() -> None:
    def custom_error_handler(exception: Exception) -> str:
      _ = f"Custom handler caught: {type(exception).__name__}"
      return "Recovered from error"

    custom_safe_decorator = FlextErrorHandlingDecorators.create_safe_decorator(
      custom_error_handler,
    )

    def _operation_with_custom_handling_impl() -> str:
      msg = "Intentional error for testing"
      raise ValueError(msg)

    operation_with_custom_handling = custom_safe_decorator(
      _operation_with_custom_handling_impl,
    )

    with contextlib.suppress(RuntimeError, ValueError, TypeError, FlextValidationError):
      cast("str", operation_with_custom_handling())


def demonstrate_performance_decorators() -> None:
    """Demonstrate performance decorators with timing and caching.

    Using flext_core.typings for type safety.
    """
    "\n" + "=" * 80

    # 1. Timing decorator

    timing_decorator = FlextPerformanceDecorators.get_timing_decorator()

    def _slow_computation_impl(n: int) -> int:
      """Slow computation for timing demonstration."""
      time.sleep(0.01)  # Simulate slow operation
      return sum(i for i in range(n))

    slow_computation = timing_decorator(_slow_computation_impl)

    slow_computation(1000) if callable(slow_computation) else 0

    # 2. Memoization decorator

    def _expensive_fibonacci_impl(n: int) -> int:
      """Expensive Fibonacci calculation with memoization."""
      if n <= 1:
          return n
      return cast("int", expensive_fibonacci(n - 1)) + cast(
          "int",
          expensive_fibonacci(n - 2),
      )

    expensive_fibonacci = FlextPerformanceDecorators.memoize_decorator(
      _expensive_fibonacci_impl,
    )

    # First call (expensive)
    start_time = time.time()
    expensive_fibonacci(10)
    first_call_time = time.time() - start_time

    # Second call (cached)
    start_time = time.time()
    expensive_fibonacci(10)
    second_call_time = time.time() - start_time

    f"   Speedup: {first_call_time / second_call_time:.1f}x"

    # 3. Cache decorator

    cache_decorator = FlextPerformanceDecorators.create_cache_decorator(max_size=10)

    def _data_processor_impl(data_id: str) -> str:
      """Process data with caching."""
      time.sleep(0.01)  # Simulate processing
      return f"Processed data: {data_id}"

    data_processor = cache_decorator(_data_processor_impl)

    # First call
    data_processor("data_001")

    # Second call (cached)
    data_processor("data_001")


def demonstrate_logging_decorators() -> None:
    """Demonstrate logging decorators with structured logging using flext_core.typings."""
    _print_logging_header()
    _demo_call_logging()
    _demo_exception_logging()
    _demo_combined_logging()


def _print_logging_header() -> None:
    "\n" + "=" * 80


def _demo_call_logging() -> None:
    def _business_operation_impl(operation_type: str, amount: float) -> TUserData:
      return {
          "operation": operation_type,
          "amount": amount,
          "status": "completed",
          "timestamp": time.time(),
      }

    business_operation = FlextLoggingDecorators.log_calls_decorator(
      _business_operation_impl,
    )
    business_operation("payment", 100.50)


def _demo_exception_logging() -> None:
    def _risky_business_operation_impl(operation_id: str) -> str:
      if operation_id == "fail":
          msg = "Operation failed intentionally"
          raise RuntimeError(msg)
      return f"Operation {operation_id} completed successfully"

    risky_business_operation = FlextLoggingDecorators.log_exceptions_decorator(
      _risky_business_operation_impl,
    )

    with contextlib.suppress(RuntimeError, ValueError, TypeError, FlextValidationError):
      risky_business_operation("success_001")

    with contextlib.suppress(RuntimeError, ValueError, TypeError, FlextValidationError):
      risky_business_operation("fail")


def _demo_combined_logging() -> None:
    def _comprehensive_service_impl(
      service_name: str,
      params: TUserData,
    ) -> TUserData:
      if service_name == "error_service":
          msg = "Service error for testing"
          raise ValueError(msg)
      return {"service": service_name, "params": params, "result": "success"}

    service_with_exceptions = FlextLoggingDecorators.log_exceptions_decorator(
      _comprehensive_service_impl,
    )
    comprehensive_service = FlextLoggingDecorators.log_calls_decorator(
      service_with_exceptions,
    )

    comprehensive_service("test_service", {"param1": "value1"})
    with contextlib.suppress(RuntimeError, ValueError, TypeError, FlextValidationError):
      comprehensive_service("error_service", {"param1": "value1"})


def demonstrate_immutability_decorators() -> None:
    """Demonstrate immutability decorators with data protection.

    Using flext_core.typings for type safety.
    """
    "\n" + "=" * 80

    # 1. Immutable result decorator

    def _create_immutable_config_impl() -> TUserData:
      """Create immutable configuration."""
      return {
          "database_url": "postgresql://localhost:5432/mydb",
          "api_key": "secret-key",
          "timeout": 30,
      }

    create_immutable_config = FlextImmutabilityDecorators.immutable_decorator(
      _create_immutable_config_impl,
    )

    config = create_immutable_config()

    # Try to modify (should fail)
    try:
      if isinstance(config, dict):
          config["new_key"] = "new_value"
    except (RuntimeError, ValueError, TypeError, FlextValidationError):
      pass

    # 2. Frozen arguments decorator

    def _process_user_data_impl(user_data: TUserData) -> TUserData:
      """Process user data with frozen arguments."""
      # Try to modify input (should fail)
      with contextlib.suppress(
          RuntimeError,
          ValueError,
          TypeError,
          FlextValidationError,
      ):
          user_data["processed"] = True

      return {"original": user_data, "processed": True}

    process_user_data = FlextImmutabilityDecorators.freeze_args_decorator(
      _process_user_data_impl,
    )

    input_data: TUserData = {"name": "Alice", "age": 30}
    process_user_data(input_data)


def demonstrate_functional_decorators() -> None:
    """Demonstrate functional decorators with composition using flext_core.typings."""
    "\n" + "=" * 80

    # 1. Decorator composition

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
    step1 = FlextPerformanceDecorators.memoize_decorator(_enterprise_user_service_impl)
    step2 = FlextValidationDecorators.validate_arguments(step1)
    step3 = FlextErrorHandlingDecorators.retry_decorator(step2)
    enterprise_user_service = FlextLoggingDecorators.log_calls_decorator(step3)

    # Test successful service
    enterprise_user_service("user_001", include_profile=True)

    # Test failed service
    with contextlib.suppress(RuntimeError, ValueError, TypeError, FlextValidationError):
      enterprise_user_service("invalid", include_profile=False)

    # 2. Pipeline decorators

    def _step1_validate_impl(data: TUserData) -> TUserData:
      """Step 1: Validate data."""
      return {**data, "validated": True}

    step1_validate = FlextPerformanceDecorators.get_timing_decorator()(
      _step1_validate_impl,
    )

    def _step2_enrich_impl(data: TUserData) -> TUserData:
      """Step 2: Enrich data."""
      return {**data, "enriched": True, "timestamp": time.time()}

    step2_enrich = FlextPerformanceDecorators.get_timing_decorator()(_step2_enrich_impl)

    def _step3_transform_impl(data: TAnyObject) -> TAnyObject:
      """Step 3: Transform data."""
      if isinstance(data, dict):
          result = dict(data)
          result.update({"transformed": True, "final": True})
          return cast("TAnyObject", result)
      return cast("TAnyObject", {"transformed": True, "final": True})

    step3_transform = FlextPerformanceDecorators.get_timing_decorator()(
      _step3_transform_impl,
    )

    # Execute pipeline
    pipeline_data: TUserData = {"input": "test_data"}

    step1_result = (
      step1_validate(pipeline_data) if callable(step1_validate) else pipeline_data
    )
    step2_result = (
      step2_enrich(step1_result) if callable(step2_enrich) else step1_result
    )
    (step3_transform(step2_result) if callable(step3_transform) else step2_result)


def demonstrate_decorator_best_practices() -> None:
    """Demonstrate decorator best practices using flext_core.typings."""
    _print_best_practices_header()
    _demo_custom_decorator_example()
    _demo_parameterized_decorators()
    _demo_performance_monitoring_decorator()


def _print_best_practices_header() -> None:
    "\n" + "=" * 80


def _demo_custom_decorator_example() -> None:
    def custom_decorator(func: object) -> object:
      def wrapper(*args: object, **kwargs: object) -> object:
          getattr(func, "__name__", "unknown")
          if callable(func):
              return func(*args, **kwargs)
          msg = "Not callable"
          raise ValueError(msg)

      return wrapper

    def _documented_function_impl(x: int, y: int) -> int:
      return x + y

    documented_function = custom_decorator(_documented_function_impl)
    documented_function(5, 3) if callable(documented_function) else 0


def _demo_parameterized_decorators() -> None:
    def create_range_validator(min_val: int, max_val: int) -> Callable[[object], bool]:
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
      return f"User age set to {age}"

    set_user_age = age_validator(_set_user_age_impl)

    def _set_completion_rate_impl(rate: int) -> str:
      return f"Completion rate set to {rate}%"

    set_completion_rate = percentage_validator(_set_completion_rate_impl)

    try:
      set_user_age(25)
      set_completion_rate(75)
    except (RuntimeError, ValueError, TypeError, FlextValidationError):
      pass

    with contextlib.suppress(RuntimeError, ValueError, TypeError, FlextValidationError):
      set_user_age(200)


def _demo_performance_monitoring_decorator() -> None:
    def performance_monitor(func: object) -> object:
      def wrapper(*args: object, **kwargs: object) -> object:
          start_time = time.time()
          if callable(func):
              result = func(*args, **kwargs)
          else:
              msg = "Not callable"
              raise TypeError(msg)
          end_time = time.time()
          end_time - start_time
          getattr(func, "__name__", "unknown")
          return result

      return wrapper

    def _monitored_operation_impl(complexity: int) -> int:
      time.sleep(0.01 * complexity)
      return complexity * 2

    monitored_operation = performance_monitor(_monitored_operation_impl)
    monitored_operation(5) if callable(monitored_operation) else 0


def demonstrate_domain_model_decorators() -> None:
    """Demonstrate decorators with shared domain models integration.

    Uses railway-oriented programming patterns.
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


def _print_section_header(title: str) -> None:  # noqa: ARG001
    """Print formatted section header."""
    "\n" + "=" * 80


def _demonstrate_validation_decorators() -> FlextResult[None]:
    """Demonstrate validation decorators with domain models."""

    def _create_validated_user_impl(name: str, email: str, age: int) -> SharedUser:
      """Create validated user using shared domain models with decorators."""
      result = SharedDomainFactory.create_user(name, email, age).flat_map(
          lambda user: _log_user_creation(user, "validation+logging"),
      )
      if result.success and result.data is not None:
          return result.data
      raise ValueError(result.error or "User creation failed")

    # Apply decorators in sequence
    create_validated_user_with_logging = FlextLoggingDecorators.log_calls_decorator(
      _create_validated_user_impl,
    )
    create_validated_user = FlextValidationDecorators.validate_arguments(
      create_validated_user_with_logging,
    )

    return _test_valid_user_creation(create_validated_user).flat_map(
      lambda _: _test_invalid_user_creation(create_validated_user),
    )


def _log_user_creation(
    user: SharedUser,
    decorator_stack: str,
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
          (
              f"✅ User created with decorators: {getattr(user, 'name', 'N/A')} "
              f"(ID: {getattr(user, 'id', 'N/A')})"
          )
          return FlextResult.ok(None)
      return FlextResult.fail("Invalid create_user_func provided")
    except (RuntimeError, ValueError, TypeError, FlextValidationError):
      return FlextResult.ok(None)  # Expected for demo


def _test_invalid_user_creation(create_user_func: object) -> FlextResult[None]:
    """Test invalid user creation scenario."""
    try:
      if callable(create_user_func):
          user = create_user_func("", "invalid-email", 15)  # Invalid data
          f"✅ Invalid user created: {getattr(user, 'name', 'N/A')}"
      return FlextResult.ok(None)
    except (RuntimeError, ValueError, TypeError, FlextValidationError):
      return FlextResult.ok(None)  # Expected failure for demo


def _demonstrate_performance_decorators() -> FlextResult[None]:
    """Demonstrate performance decorators with domain operations."""

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
      _find_user_by_email_impl,
    )
    find_user_by_email = FlextPerformanceDecorators.memoize_decorator(
      find_user_with_timing,  # type: ignore[arg-type]
    )

    return _execute_performance_lookups(find_user_by_email)


def _execute_performance_lookups(lookup_func: object) -> FlextResult[None]:
    """Execute performance lookup demonstrations."""
    if not callable(lookup_func):
      return FlextResult.fail("Invalid lookup function provided")

    # First lookup (expensive)
    user1 = lookup_func("cached@example.com")
    if user1:
      f"✅ First lookup: {getattr(user1, 'name', 'N/A')} ({getattr(getattr(user1, 'email_address', None), 'email', 'N/A') if hasattr(user1, 'email_address') else 'N/A'})"

    # Second lookup (cached)
    user2 = lookup_func("cached@example.com")
    if user2:
      f"✅ Second lookup (cached): {getattr(user2, 'name', 'N/A')}"

    return FlextResult.ok(None)


def _demonstrate_error_handling_decorators() -> FlextResult[None]:
    """Demonstrate error handling decorators with domain operations."""

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
      _activate_user_account_impl,
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
    user: SharedUser,
    activated_user: SharedUser,
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
      "Test User",
      "test@example.com",
      30,
    )
    if not test_user_result.success:
      return FlextResult.fail("Failed to create test user")

    test_user = test_user_result.data
    if test_user is not None:
      return _test_successful_activation(activate_func, test_user).flat_map(
          lambda activated: _test_duplicate_activation(activate_func, activated),
      )
    return FlextResult.fail("Test user creation returned None")


def _test_successful_activation(
    activate_func: object,
    test_user: SharedUser,
) -> FlextResult[SharedUser]:
    """Test successful user activation."""
    if not callable(activate_func):
      return FlextResult.fail("Invalid activation function")

    try:
      activated_user = activate_func(test_user)
      (
          f"✅ User activated: {getattr(activated_user, 'name', 'N/A')} "
          f"(Status: {getattr(getattr(activated_user, 'status', None), 'value', 'N/A') if hasattr(activated_user, 'status') else 'N/A'})"
      )
      return FlextResult.ok(activated_user)
    except Exception as e:
      error_message = f"Activation failed: {e}"
      return FlextResult.fail(error_message)


def _test_duplicate_activation(
    activate_func: object,
    activated_user: SharedUser,
) -> FlextResult[None]:
    """Test duplicate activation scenario (should fail)."""
    if not callable(activate_func):
      return FlextResult.ok(None)

    with contextlib.suppress(Exception):
      activate_func(activated_user)

    return FlextResult.ok(None)


def _demonstrate_domain_validation_decorators() -> FlextResult[None]:
    """Demonstrate domain-aware validation decorators."""

    def domain_user_validator(user_data: object) -> bool:
      """Validate user data using domain models."""
      return (
          _validate_user_data_structure(user_data)
          .flat_map(_validate_with_domain_factory)
          .map(lambda result: result.success)
          .unwrap_or(default=False)
      )

    domain_validator = FlextValidationDecorators.create_validation_decorator(
      domain_user_validator,
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
      _register_user_with_domain_validation_impl,
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
      lambda _: _test_invalid_registration(register_func),
    )


def _test_valid_registration(register_func: object) -> FlextResult[None]:
    """Test valid user registration."""
    if not callable(register_func):
      return FlextResult.fail("Invalid registration function")

    try:
      valid_data = {"name": "Domain User", "email": "domain@example.com", "age": 32}
      user = register_func(valid_data)
      f"✅ Domain validation passed: {getattr(user, 'name', 'N/A')}"
      return FlextResult.ok(None)
    except (RuntimeError, ValueError, TypeError, FlextValidationError):
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
      f"✅ Invalid registration: {getattr(user, 'name', 'N/A')}"
    except (RuntimeError, ValueError, TypeError, FlextValidationError):
      pass

    return FlextResult.ok(None)


def _complete_domain_decorators_demo() -> FlextResult[None]:
    """Complete the domain decorators demonstration."""
    return FlextResult.ok(None)


def main() -> None:
    """Run comprehensive FlextDecorators demonstration with maximum type safety."""
    # Run all demonstrations
    demonstrate_validation_decorators()
    demonstrate_error_handling_decorators()
    demonstrate_performance_decorators()
    demonstrate_logging_decorators()
    demonstrate_immutability_decorators()
    demonstrate_functional_decorators()
    demonstrate_decorator_best_practices()
    demonstrate_domain_model_decorators()


if __name__ == "__main__":
    main()
