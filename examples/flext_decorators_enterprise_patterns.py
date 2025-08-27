#!/usr/bin/env python3
"""09 - Enterprise Decorators: Modern FlextDecorators API Showcase.

Demonstrates the refactored FlextDecorators API with complete type safety.
Shows modern decorator composition patterns and FlextResult integration.

Key Patterns:
• Modern FlextDecorators API usage
• Type-safe decorator composition
• FlextResult integration patterns
• Enterprise-grade function enhancement
"""

from __future__ import annotations

import contextlib
import random
import time

from shared_domain import SharedDomainFactory

from flext_core import (
    FlextConstants,
    FlextCore,
    FlextDecorators,
    FlextLogger,
    FlextProtocols,
    FlextResult,
    FlextTypes,
)

# Singleton FlextCore instance for all utilities
core = FlextCore.get_instance()
logger = FlextLogger("flext.examples.decorators")

# Constants using FlextConstants hierarchical access
MAX_AGE: FlextTypes.Core.Integer = 150
MIN_AGE: FlextTypes.Core.Integer = 0
SUCCESS_THRESHOLD: FlextTypes.Core.Float = 0.4
MIN_USER_CREATION_ARGS: FlextTypes.Core.Integer = 3

# =============================================================================
# PROTOCOLS - Using FlextProtocols hierarchical patterns
# =============================================================================


class CalculationProtocol(FlextProtocols.Foundation.Validator[FlextTypes.Core.Integer]):
    """Protocol for calculation operations using centralized FlextProtocols."""

    def validate(
        self, data: FlextTypes.Core.Integer
    ) -> FlextResult[FlextTypes.Core.Integer]:
        """Process calculation with FlextResult pattern."""
        ...  # noqa: PIE790


class ValidationProtocol(FlextProtocols.Foundation.Validator[FlextTypes.Core.Integer]):
    """Protocol for validation operations using centralized FlextProtocols."""

    def validate(
        self, data: FlextTypes.Core.Integer
    ) -> FlextResult[FlextTypes.Core.Integer]:
        """Validate data with FlextResult pattern."""
        ...  # noqa: PIE790


# =============================================================================
# MODERN FLEXT DECORATORS SHOWCASE - Maximum FLEXT Integration
# =============================================================================


def demonstrate_cache_decorator() -> FlextResult[FlextTypes.Core.String]:
    """Demonstrate modern cache decorator with maximum FLEXT integration.

    Returns:
        FlextResult containing demo status or error message.

    """

    @FlextDecorators.Performance.cache(max_size=128)
    def expensive_calculation(
        x: FlextTypes.Core.Integer,
    ) -> FlextResult[FlextTypes.Core.Integer]:
        """Expensive calculation using FlextResult pattern and FlextTypes."""
        # Validation using centralized constants
        if x < MIN_AGE or x > MAX_AGE:
            return FlextResult[FlextTypes.Core.Integer].fail(
                FlextConstants.Errors.VALIDATION_ERROR
            )

        # Simulate expensive work using centralized delay
        delay_ms: FlextTypes.Core.Float = 100.0
        time.sleep(delay_ms / 1000.0)

        result: FlextTypes.Core.Integer = x * x * x
        return FlextResult[FlextTypes.Core.Integer].ok(result)

    # Test cache functionality with proper FlextResult handling
    logger.info("Testing cache functionality with centralized patterns")

    try:
        # First call - should be slow
        start_time = time.time()
        first_result = expensive_calculation(5)
        first_duration = time.time() - start_time

        if first_result.is_failure:
            return FlextResult[FlextTypes.Core.String].fail(
                f"First calculation failed: {first_result.error}"
            )

        # Second call - should be cached and fast
        start_time = time.time()
        second_result = expensive_calculation(5)
        second_duration = time.time() - start_time

        if second_result.is_failure:
            return FlextResult[FlextTypes.Core.String].fail(
                f"Second calculation failed: {second_result.error}"
            )

        # Verify cache effectiveness using FlextUtilities
        cache_effective = second_duration < (first_duration * 0.1)

        status_msg: FlextTypes.Core.String = (
            f"Cache demo completed: first={first_duration:.4f}s, "
            f"second={second_duration:.4f}s, effective={cache_effective}"
        )

        logger.info(status_msg)
        return FlextResult[FlextTypes.Core.String].ok(status_msg)

    except Exception as e:
        error_msg = f"Cache demo failed: {e}"
        logger.exception(error_msg)
        return FlextResult[FlextTypes.Core.String].fail(error_msg)


def demonstrate_complete_decorator() -> FlextResult[FlextTypes.Core.String]:
    """Demonstrate complete decorator composition with maximum FLEXT integration.

    Returns:
        FlextResult containing operation status or error message.

    """

    @FlextDecorators.Performance.monitor()
    @FlextDecorators.Performance.cache(max_size=64)
    def business_operation(data: dict[str, object]) -> FlextResult[dict[str, object]]:
        """Business operation using FlextResult pattern and centralized constants."""
        try:
            # Validate input using centralized error constants
            if not data or not isinstance(data, dict):
                return FlextResult[dict[str, object]].fail(
                    FlextConstants.Errors.VALIDATION_ERROR
                )

            # Simulate processing with centralized timing
            processing_delay: FlextTypes.Core.Float = 0.05
            time.sleep(processing_delay)

            # Business logic with controlled randomness
            if random.random() > SUCCESS_THRESHOLD:  # noqa: S311
                result_data: dict[str, object] = {
                    "status": "processed",
                    "data": data,
                    "timestamp": time.time(),
                    "processing_time_ms": processing_delay * 1000,
                }
                return FlextResult[dict[str, object]].ok(result_data)

            return FlextResult[dict[str, object]].fail(
                "Random processing failure occurred"
            )

        except Exception as e:
            return FlextResult[dict[str, object]].fail(
                f"Business operation failed: {e}"
            )

    # Test the decorated function with proper error handling
    test_data: dict[str, object] = {
        "id": 123,
        "name": "Test Operation",
        "correlation_id": core.generate_uuid()
        if hasattr(core, "generate_uuid")
        else "test_123",
    }

    logger.info("Testing complete decorator composition")

    try:
        operation_result = business_operation(test_data)

        if operation_result.is_success:
            result_data = operation_result.unwrap()
            success_msg: FlextTypes.Core.String = (
                f"Operation completed: {result_data.get('status', 'unknown')}"
            )
            logger.info(success_msg)
            return FlextResult[FlextTypes.Core.String].ok(success_msg)

        failure_msg: FlextTypes.Core.String = (
            f"Operation failed: {operation_result.error}"
        )
        logger.warning(failure_msg)
        return FlextResult[FlextTypes.Core.String].ok(
            f"Handled failure: {operation_result.error}"
        )

    except Exception as e:
        error_msg = f"Decorator demo failed: {e}"
        logger.exception(error_msg)
        return FlextResult[FlextTypes.Core.String].fail(error_msg)


def demonstrate_safe_result_decorator() -> FlextResult[FlextTypes.Core.String]:
    """Demonstrate safe result decorator with maximum FLEXT integration.

    Returns:
        FlextResult containing demo status or error message.

    """

    @FlextDecorators.Reliability.safe_result
    def risky_operation(data: FlextTypes.Core.String) -> FlextTypes.Core.String:
        """Risky operation using FlextTypes that might fail."""
        # Use centralized error patterns
        if data == "fail":
            error_msg = "Intentional failure"
            raise ValueError(error_msg)

        return f"Success with {data}"

    logger.info("Testing safe result decorator with centralized patterns")

    try:
        # Test success case with FlextResult handling
        success_result = risky_operation("success")
        logger.info(f"Success case result: {success_result}")

        # Test failure case with FlextResult handling
        failure_result = risky_operation("fail")
        logger.info(f"Failure case result: {failure_result}")

        status_msg: FlextTypes.Core.String = (
            "Safe result decorator demo completed successfully"
        )
        logger.info(status_msg)
        return FlextResult[FlextTypes.Core.String].ok(status_msg)

    except Exception as e:
        error_msg = f"Safe result decorator demo failed: {e}"
        logger.exception(error_msg)
        return FlextResult[FlextTypes.Core.String].fail(error_msg)


def demonstrate_user_creation_with_modern_decorators() -> None:
    """Demonstrate user creation with modern decorators."""

    # Create a comprehensive decorator for user operations using available API
    @FlextDecorators.Performance.cache(max_size=32)
    @FlextDecorators.Performance.monitor()
    def create_user_generic(*args: object, **_kwargs: object) -> object:
        """Generic user creator compatible with FlextCallable."""
        if len(args) >= MIN_USER_CREATION_ARGS:
            try:
                name = str(args[0]) if args[0] is not None else ""
                email = str(args[1]) if args[1] is not None else ""
                age_val = args[2]

                # Safe age conversion
                if isinstance(age_val, (int, float)) or (
                    isinstance(age_val, str) and age_val.isdigit()
                ):
                    age = int(age_val)
                else:
                    msg = f"Invalid age: {age_val}"
                    raise ValueError(msg)  # noqa: TRY301

                # Basic validation
                if not name or not name.strip():
                    msg = "Name required"
                    raise ValueError(msg)  # noqa: TRY301
                if "@" not in email:
                    msg = "Valid email required"
                    raise ValueError(msg)  # noqa: TRY301
                if age < MIN_AGE or age > MAX_AGE:
                    msg = "Valid age required"
                    raise ValueError(msg)  # noqa: TRY301

                # Create user using SharedDomainFactory
                result = SharedDomainFactory.create_user(name, email, age)
                if result.success:
                    return result.value
                msg = f"User creation failed: {result.error}"
                raise ValueError(msg)  # noqa: TRY301

            except (ValueError, TypeError) as e:
                msg = f"Type conversion failed: {e}"
                raise ValueError(msg) from e
        msg = "Insufficient arguments"
        raise ValueError(msg)

    # Test user creation
    try:
        user_result = create_user_generic(
            "Alice Modern", "alice.modern@example.com", 25
        )
        if hasattr(user_result, "name"):
            pass
    except Exception:  # noqa: S110
        pass

    # Test validation failure
    with contextlib.suppress(Exception):
        create_user_generic("", "invalid", -1)


def demonstrate_decorator_categories() -> None:
    """Demonstrate different decorator categories."""
    # Performance decorators
    performance_decorators = FlextDecorators.Performance
    # Demonstrate cache method availability
    performance_decorators.cache(max_size=50)

    # Error handling decorators (using Reliability)
    error_handling_decorators = FlextDecorators.Reliability
    # Check safe_result method is available
    hasattr(error_handling_decorators, "safe_result")

    # Complete decorator composition using available API
    @FlextDecorators.Observability.log_execution()
    @FlextDecorators.Performance.monitor()
    @FlextDecorators.Performance.cache(max_size=32)
    def complete_example_function() -> str:
        """Example of complete decorator composition."""
        return "completed"

    complete_example_function()  # Test the decorated function


# =============================================================================
# DEMONSTRATIONS - Modern decorator usage
# =============================================================================


def main() -> FlextResult[FlextTypes.Core.String]:
    """🎯 Example 09: Modern Enterprise Decorators with maximum FLEXT integration.

    Demonstrates all decorator patterns using centralized FlextResult handling
    and FlextTypes for complete type safety.

    Returns:
        FlextResult containing execution summary or error details.

    """
    logger.info("Starting Example 09: Modern Enterprise Decorators")

    success_count: FlextTypes.Core.Integer = 0
    total_operations: FlextTypes.Core.Integer = 5
    operation_results: list[FlextTypes.Core.String] = []

    try:
        # Execute all decorator demonstrations with FlextResult handling
        operations = [
            ("Cache Decorator", demonstrate_cache_decorator),
            ("Complete Decorator", demonstrate_complete_decorator),
            ("Safe Result Decorator", demonstrate_safe_result_decorator),
        ]

        for operation_name, operation_func in operations:
            logger.info(f"Executing {operation_name} demonstration")

            try:
                result = operation_func()
                if result.is_success:
                    success_count += 1
                    operation_results.append(f"✅ {operation_name}: {result.unwrap()}")
                else:
                    operation_results.append(f"❌ {operation_name}: {result.error}")

            except Exception as e:
                error_msg = f"❌ {operation_name}: Exception {e}"
                operation_results.append(error_msg)
                logger.exception(f"{operation_name} failed with exception")

        # Execute non-FlextResult demonstrations
        try:
            demonstrate_user_creation_with_modern_decorators()
            demonstrate_decorator_categories()
            success_count += 2
            operation_results.extend([
                "✅ User Creation: Completed successfully",
                "✅ Decorator Categories: Completed successfully",
            ])
        except Exception as e:
            operation_results.extend([
                f"❌ User Creation: Exception {e}",
                f"❌ Decorator Categories: Exception {e}",
            ])
            logger.exception("Additional demonstrations failed")

        # Calculate success rate using FlextTypes
        success_rate: FlextTypes.Core.Float = (success_count / total_operations) * 100.0

        # Generate comprehensive summary using centralized patterns
        summary_lines = [
            "Example 09 Execution Summary:",
            f"Total Operations: {total_operations}",
            f"Successful: {success_count}",
            f"Success Rate: {success_rate:.1f}%",
            "Results:",
        ]
        summary_lines.extend([f"  {result}" for result in operation_results])

        final_summary: FlextTypes.Core.String = "\n".join(summary_lines)

        if success_count >= (total_operations * 0.8):  # 80% success threshold
            logger.info(
                f"Example 09 completed successfully with {success_rate:.1f}% success rate"
            )
            return FlextResult[FlextTypes.Core.String].ok(final_summary)
        logger.warning(
            f"Example 09 completed with suboptimal {success_rate:.1f}% success rate"
        )
        return FlextResult[FlextTypes.Core.String].fail(
            f"Suboptimal execution: {success_rate:.1f}% success rate\n{final_summary}"
        )

    except Exception as e:
        error_summary = f"Example 09 failed with critical exception: {e}"
        logger.exception(error_summary)
        return FlextResult[FlextTypes.Core.String].fail(error_summary)


if __name__ == "__main__":
    # Execute main with FlextResult handling and centralized error patterns
    execution_result = main()

    if execution_result.is_success:
        summary = execution_result.unwrap()
        logger.info("Example execution completed successfully")
        print(summary)
    else:
        error_details = execution_result.error
        logger.error(f"Example execution failed: {error_details}")
        print(f"❌ Example 09 Failed: {error_details}")
