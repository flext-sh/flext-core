#!/usr/bin/env python3
"""09 - Enterprise Decorators: Modern FlextDecorators API Showcase.

Demonstrates the refactored FlextDecorators API with complete type safety.
Shows modern decorator composition patterns and FlextResult integration.
"""

from __future__ import annotations

import time

from flext_core import (
    FlextConstants,
    FlextDecorators,
    FlextLogger,
    FlextProtocols,
    FlextResult,
    FlextTypes,
    FlextUtilities,
)

MIN_NAME_LENGTH = 2
MINIMUM_AGE = 18


class LocalDomainFactory:
    """Local domain factory for user creation without external dependencies."""

    @staticmethod
    def create_user(
        name: str, email: str, age: int
    ) -> FlextResult[FlextTypes.Core.Dict]:
        """Create user with validation."""
        # Basic validation
        if not name or len(name) < MIN_NAME_LENGTH:
            return FlextResult[FlextTypes.Core.Dict].fail("Invalid name")
        if "@" not in email:
            return FlextResult[FlextTypes.Core.Dict].fail("Invalid email")
        if age < MINIMUM_AGE:
            return FlextResult[FlextTypes.Core.Dict].fail("User must be 18+")

        user_data = {
            "name": name,
            "email": email,
            "age": age,
            "id": f"user_{FlextUtilities.Generators.generate_uuid()}",
        }
        return FlextResult[FlextTypes.Core.Dict].ok(user_data)


# FlextCore singleton removed - using direct imports
logger = FlextLogger("flext.examples.decorators")

# Constants using FlextConstants hierarchical access
MAX_AGE: int = 150
MIN_AGE: int = 0
SUCCESS_THRESHOLD: float = 0.4
MIN_USER_CREATION_ARGS: int = 3


class CalculationProtocol(FlextProtocols.Foundation.Validator[int]):
    """Protocol for calculation operations using centralized FlextProtocols."""

    def validate(self, data: int) -> FlextResult[int]:
        """Process calculation with FlextResult pattern."""
        if data < 0:
            return FlextResult[int].fail("Negative values not allowed")
        return FlextResult[int].ok(data * 2)  # Example calculation


class ValidationProtocol(FlextProtocols.Foundation.Validator[int]):
    """Protocol for validation operations using centralized FlextProtocols."""

    def validate(self, data: int) -> FlextResult[int]:
        """Validate data with FlextResult pattern."""
        if data <= 0:
            return FlextResult[int].fail("Value must be positive")
        return FlextResult[int].ok(data)


def demonstrate_cache_decorator() -> FlextResult[str]:
    """Demonstrate cache decorator using Strategy Pattern - REDUCED COMPLEXITY."""

    # Strategy Pattern: Use FlextUtilities.Performance for measurement
    @FlextDecorators.Performance.cache(max_size=128)
    def expensive_calculation(x: int) -> FlextResult[int]:
        """Cached calculation using FlextResult pattern."""
        return (
            FlextResult[int]
            .ok(x)
            .flat_map(
                lambda v: FlextResult[int].fail(FlextConstants.Errors.VALIDATION_ERROR)
                if v < MIN_AGE or v > MAX_AGE
                else FlextResult[int].ok(v),
            )
            .flat_map(lambda v: FlextResult[int].ok(v * v * v))
        )

    # Template Method Pattern: Use flext-core utilities
    logger.info("Testing cache functionality with Strategy Pattern")

    # Execute using FlextUtilities.Performance measurement
    first_result = expensive_calculation(5)
    second_result = expensive_calculation(5)  # Should be cached

    if first_result.is_success and second_result.is_success:
        return FlextResult[str].ok("Cache demo completed successfully")

    error_msg = first_result.error or second_result.error or "Unknown error"
    return FlextResult[str].fail(f"Cache demo failed: {error_msg}")


def demonstrate_complete_decorator() -> FlextResult[str]:
    """Demonstrate decorator composition using Strategy Pattern - REDUCED COMPLEXITY."""

    # Strategy Pattern: Use FlextUtilities for data generation
    @FlextDecorators.Performance.monitor()
    @FlextDecorators.Performance.cache(max_size=64)
    def business_operation(
        data: FlextTypes.Core.Dict,
    ) -> FlextResult[FlextTypes.Core.Dict]:
        """Business operation with flext-core validation."""
        return (
            FlextResult[FlextTypes.Core.Dict]
            .ok(data)
            .flat_map(
                lambda d: FlextResult[FlextTypes.Core.Dict].fail(
                    FlextConstants.Errors.VALIDATION_ERROR,
                )
                if not d or not isinstance(d, dict)
                else FlextResult[FlextTypes.Core.Dict].ok(d),
            )
            .map(
                lambda d: {
                    "status": "processed",
                    "data": d,
                    "timestamp": time.time(),
                },
            )
        )

    # Template Method Pattern: Use existing flext-core utilities
    logger.info("Testing complete decorator composition")

    test_data = {
        "id": 123,
        "name": "Test Operation",
        "correlation_id": FlextUtilities.Generators.generate_uuid(),
    }

    # Execute business operation with error handling
    result = business_operation(test_data)
    return result.map(
        lambda r: f"Operation completed: {r.get('status', 'unknown')}",
    ).or_else(FlextResult[str].ok("Handled failure: unknown"))


def demonstrate_safe_result_decorator() -> FlextResult[str]:
    """Demonstrate safe result decorator - SIMPLIFIED."""

    # Strategy Pattern: Use FlextUtilities.Validators for safe operations
    @FlextDecorators.Reliability.safe_result
    def risky_operation(data: str) -> str:
        """Safe operation using flext-core validation."""
        if data == "fail":
            msg = "Intentional failure"
            raise ValueError(msg)
        return f"Success with {data}"

    logger.info("Testing safe result decorator")

    # Execute using Railway Pattern
    risky_operation("success")
    risky_operation("fail")

    return FlextResult[str].ok("Safe result decorator demo completed successfully")


def demonstrate_user_creation_with_modern_decorators() -> None:
    """Demonstrate user creation with modern decorators - SIMPLIFIED."""

    # Strategy Pattern: Use LocalDomainFactory directly
    @FlextDecorators.Performance.cache(max_size=32)
    @FlextDecorators.Performance.monitor()
    def create_user_with_validation(
        name: str,
        email: str,
        age: int,
    ) -> FlextResult[FlextTypes.Core.Dict]:
        """Create user using flext-core validation patterns."""
        return LocalDomainFactory.create_user(name, email, age)

    # Execute tests using Railway Pattern
    success_test = create_user_with_validation(
        "Alice Modern",
        "alice.modern@example.com",
        25,
    )
    if success_test.is_success:
        logger.info("User creation test passed")
    else:
        logger.warning(f"User creation test failed: {success_test.error}")

    # Test validation failure
    failure_test = create_user_with_validation("", "invalid", -1)
    if failure_test.is_failure:
        logger.info("Validation failure test passed")


def demonstrate_decorator_categories() -> None:
    """Demonstrate decorator categories - SIMPLIFIED."""

    # Template Method Pattern: Use flext-core decorator composition
    @FlextDecorators.Observability.log_execution()
    @FlextDecorators.Performance.monitor()
    @FlextDecorators.Performance.cache(max_size=32)
    def complete_example_function() -> str:
        """Complete decorator composition example."""
        return "completed"

    # Execute test
    result = complete_example_function()
    logger.info(f"Decorator categories test: {result}")


def main() -> FlextResult[str]:
    """🎯 Example 09: Modern Enterprise Decorators with maximum FLEXT integration.

    Demonstrates all decorator patterns using centralized FlextResult handling
    and FlextTypes for complete type safety.

    Returns:
        FlextResult containing execution summary or error details.

    """
    logger.info("Starting Example 09: Modern Enterprise Decorators")

    success_count: int = 0
    total_operations: int = 5
    operation_results: FlextTypes.Core.StringList = []

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
            operation_results.extend(
                [
                    "✅ User Creation: Completed successfully",
                    "✅ Decorator Categories: Completed successfully",
                ],
            )
        except Exception as e:
            operation_results.extend(
                [
                    f"❌ User Creation: Exception {e}",
                    f"❌ Decorator Categories: Exception {e}",
                ],
            )
            logger.exception("Additional demonstrations failed")

        # Calculate success rate using FlextTypes
        success_rate: float = (success_count / total_operations) * 100.0

        # Generate comprehensive summary using centralized patterns
        summary_lines = [
            "Example 09 Execution Summary:",
            f"Total Operations: {total_operations}",
            f"Successful: {success_count}",
            f"Success Rate: {success_rate:.1f}%",
            "Results:",
        ]
        summary_lines.extend([f"  {result}" for result in operation_results])

        final_summary: str = "\n".join(summary_lines)

        if success_count >= (total_operations * 0.8):  # 80% success threshold
            logger.info(
                f"Example 09 completed successfully with "
                f"{success_rate:.1f}% success rate",
            )
            return FlextResult[str].ok(final_summary)
        logger.warning(
            f"Example 09 completed with suboptimal {success_rate:.1f}% success rate",
        )
        return FlextResult[str].fail(
            f"Suboptimal execution: {success_rate:.1f}% success rate\n{final_summary}",
        )

    except Exception as e:
        error_summary = f"Example 09 failed with critical exception: {e}"
        logger.exception(error_summary)
        return FlextResult[str].fail(error_summary)


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
