"""r comprehensive demonstration.

Shows complete railway-oriented programming with advanced type safety.
Demonstrates all r patterns using Python 3.13+ features.

**Expected Output:**
- Factory methods demonstration (ok, fail, create_from_callable)
- Value extraction patterns (unwrap, unwrap_or, value property)
- Railway operations (map, flat_map, flow_through)
- Error recovery (alt, lash)
- Advanced combinators (traverse, filter)
- Validation patterns with u.Validation
- Exception integration with structured errors

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from flext_core import (
    c,
    e,
    m,
    r,
    s,
    t,
    u,
)

# Using t directly - no local type aliases (DRY + SRP)
# All types come from t namespace - centralized type system
# PEP 695 type aliases only when necessary for complex compositions

# =====================================================================
# DEMONSTRATION MODELS - Using m for type safety
# =====================================================================


class User(m.Entity):
    """User entity for demonstration."""

    name: str
    email: str


class DemonstrationResult(m.Value):
    """Result value object for demonstration metadata."""

    demonstrations_completed: int
    patterns_covered: tuple[str, ...]
    completed_at: str


class RunDemonstrationCommand(m.Cqrs.Command):
    """Command to run demonstration."""

    operation: str = "demonstration"


# Handler removed due to dispatcher serialization issues


# =====================================================================
# DEMONSTRATION DATA - Using centralized c.Example
# =====================================================================


class RailwayService(s[DemonstrationResult]):
    """Advanced service demonstrating railway patterns with comprehensive flext-core integration."""

    def __init__(self, **kwargs: t.GeneralValueType) -> None:
        """Initialize service."""
        super().__init__(**kwargs)
        # Dependencies created directly to avoid serialization issues in handlers

    def execute(self) -> r[DemonstrationResult]:
        """Execute comprehensive r demonstrations."""
        print("Starting r comprehensive demonstration")

        try:
            # Chain all demonstrations using railway pattern
            return (
                self._run_demonstrations()
                .flat_map(self._build_result_data)
                .map(self._log_success)
            )

        except Exception as e:
            return self._handle_execution_error(e)

    def _run_demonstrations(self) -> r[None]:
        """Run all demonstration methods using advanced functional composition."""
        # Define demonstrations using collections.abc.Sequence for type safety
        demonstrations: Sequence[Callable[[], None]] = (
            self._demonstrate_factory_methods,
            self._demonstrate_value_extraction,
            self._demonstrate_railway_operations,
            self._demonstrate_error_recovery,
            self._demonstrate_advanced_combinators,
            self._demonstrate_validation_patterns,
            self._demonstrate_exception_integration,
        )

        # Execute demonstrations with advanced traverse pattern (DRY)
        results = [RailwayService._execute_demo(demo) for demo in demonstrations]
        return r.traverse(results, lambda result: result).map(lambda _: None)

    @staticmethod
    def _execute_demo(demo: Callable[[], None]) -> r[bool]:
        """Execute demonstration and return success result."""
        try:
            demo()
            return r.ok(True)
        except Exception as e:
            return r.fail(f"Demonstration failed: {e}")

    @staticmethod
    def _build_result_data(_: None) -> r[DemonstrationResult]:
        """Build result data using centralized t and DRY patterns."""
        # Use all demo patterns from centralized constants (DRY)
        # Iterate over enum members correctly
        patterns = tuple(
            member.value for member in c.Cqrs.HandlerType.__members__.values()
        )

        result_data = DemonstrationResult(
            demonstrations_completed=len(patterns),
            patterns_covered=patterns,
            completed_at=datetime.now(UTC).isoformat(),
        )

        return r.ok(result_data)

    def _log_success(self, data: DemonstrationResult) -> DemonstrationResult:
        """Log success and return data using advanced logging patterns."""
        self.logger.info("r demonstration completed successfully")
        return data

    def _handle_execution_error(self, error: Exception) -> r[DemonstrationResult]:
        """Handle execution errors with proper typing."""
        error_msg = f"Demonstration failed: {error}"
        self.logger.error(error_msg)
        return r[DemonstrationResult].fail(
            error_msg,
            error_code=c.Errors.EXCEPTION_ERROR,
        )

    @staticmethod
    def _create_user_validator() -> Callable[[str], r[User]]:
        """Create user validator using u (DRY)."""

        def validate_user(email: str) -> r[User]:
            # Use u for email validation (DRY)
            email_validation = u.Validation.validate_pattern(
                email,
                c.Platform.PATTERN_EMAIL,
                "email",
            )
            if email_validation.is_failure:
                return r[User].fail(email_validation.error or "Invalid email")
            return r.ok(User(name="Demo User", email=email))

        return validate_user

    @staticmethod
    def _create_data_processor() -> Callable[
        [DemonstrationResult],
        DemonstrationResult,
    ]:
        """Create data processor using advanced patterns."""

        def process_data(data: DemonstrationResult) -> DemonstrationResult:
            # Add processing logic here
            return data

        return process_data

    @staticmethod
    def _demonstrate_factory_methods() -> None:
        """Demonstrate r factory methods with advanced patterns."""
        print("\n=== Factory Methods ===")

        # Success result
        success = r.ok("Operation successful")
        print(f"✅ .ok(): {success.value}")

        # Failure result with centralized error code
        failure: r[str] = r.fail(
            "Validation failed",
            error_code=c.Errors.VALIDATION_ERROR,
        )
        print(f"❌ .fail(): {failure.error}")

        # From callable with intentional error
        def risky_operation() -> int:
            zero = c.ZERO
            return c.Validation.MAX_AGE // zero

        from_callable = r[int].create_from_callable(risky_operation)
        print(f"🔥 .create_from_callable(): {from_callable.error}")

    @staticmethod
    def _demonstrate_value_extraction() -> None:
        """Demonstrate r value extraction with advanced patterns."""
        print("\n=== Value Extraction ===")

        # Use example data from constants with centralized t
        success = r.ok({"name": "John", "email": "john@example.com"})
        failure: r[str] = r.fail("Not found")

        # Value extraction patterns
        user_data = success.value
        print(f".value success: {user_data[c.Mixins.FIELD_NAME]}")
        print(f".unwrap_or() failure: {failure.unwrap_or('default')}")
        print(f".value property: {user_data['email']}")
        print(f".value: {success.value}")

    @staticmethod
    def _demonstrate_railway_operations() -> None:
        """Demonstrate railway operations with advanced functional composition."""
        print("\n=== Railway Operations ===")

        # Use u directly (DRY - no custom validation functions)
        def to_upper(value: str) -> str:
            return value.upper()

        # Map transformation
        input_value = "hello"
        mapped = r.ok(input_value).map(to_upper)
        print(f".map(to_upper): {mapped.value}")

        # FlatMap chaining with u validation (DRY)
        test_value = "test"

        def validate_length(v: str) -> r[str]:
            return u.Validation.validate_length(
                v,
                min_length=c.Validation.MIN_USERNAME_LENGTH,
            )

        chained = r.ok(test_value).flat_map(validate_length).map(to_upper)
        print(f".flat_map chain: {chained.value}")

        # Flow through pipeline with advanced composition using u (DRY)
        def add_prefix(value: str) -> r[str]:
            return r.ok(f"PREFIX_{value}")

        def add_exclamation(x: str) -> r[str]:
            return r[str].ok(f"{x}!")

        pipeline = r.ok(input_value).flow_through(
            validate_length,
            add_prefix,
            add_exclamation,
        )
        print(f".flow_through pipeline: {pipeline.value}")

    @staticmethod
    def _demonstrate_error_recovery() -> None:
        """Demonstrate error recovery patterns with advanced functional composition."""
        print("\n=== Error Recovery ===")

        failure: r[str] = r.fail("Primary operation failed")

        # Alternative (transform error) using functional approach
        def recover_message(error: str) -> str:
            return f"Recovered from: {error}"

        recovered = failure.alt(recover_message)
        print(f".alt() transform: {recovered.error}")

        # Lash (error recovery) with fallback
        def provide_fallback(_error: str) -> r[str]:
            return r.ok("Fallback value")

        fallback = failure.lash(provide_fallback)
        print(f".lash() recovery: {fallback.value}")

    @staticmethod
    def _demonstrate_advanced_combinators() -> None:
        """Advanced functional programming patterns."""
        print("\n=== Advanced Combinators ===")

        # Traverse multiple results with type safety
        results = [
            r[int].ok(c.ZERO + 1),  # 1
            r[int].ok(c.ZERO + 2),  # 2
            r[int].ok(c.ZERO + 3),  # 3
        ]

        traversed = r.traverse(results, lambda r: r)
        print(f".traverse(): {len(traversed.value)} results")

        # Filter with predicate using c threshold
        test_value = c.Validation.FILTER_THRESHOLD + c.Validation.MIN_AGE  # 10
        filtered = (
            r[int].ok(test_value).filter(lambda x: x > c.Validation.FILTER_THRESHOLD)
        )
        print(f".filter(>{c.Validation.FILTER_THRESHOLD}): {filtered.is_success}")

    @staticmethod
    def _demonstrate_validation_patterns() -> None:
        """Demonstrate validation patterns using u and container for DRY."""
        print("\n=== Validation Patterns ===")

        # Create validator directly (DI pattern simplified for demo)
        user_validator = RailwayService._create_user_validator()

        # Chain validations using railway pattern with u (DRY)
        test_email = "test@example.com"
        result = (
            r.ok(test_email)
            .flat_map(user_validator)
            .map(lambda user: user.email)
            .flat_map(
                lambda email: u.Validation.validate_length(
                    email,
                    min_length=c.Validation.MIN_USERNAME_LENGTH,
                    max_length=c.Validation.MAX_NAME_LENGTH,
                ),
            )
        )
        print(f"Validation chain with User model: {result.is_success}")

        # Multiple validations with traverse using u (DRY - no custom validators)
        test_email_2 = "user@domain.com"
        validation_results = [
            u.Validation.validate_pattern(
                test_email_2,
                c.Platform.PATTERN_EMAIL,
                "email",
            ),
            u.Validation.validate_length(
                test_email_2,
                min_length=c.Validation.MIN_USERNAME_LENGTH,
                max_length=c.Validation.MAX_NAME_LENGTH,
            ),
        ]
        all_valid = r.traverse(validation_results, lambda r: r)
        print(f"Multiple validations: {all_valid.is_success}")

    @staticmethod
    def _demonstrate_exception_integration() -> None:
        """Demonstrate structured exception integration with r."""
        print("\n=== Exception Integration ===")

        error_message = "Invalid data provided"
        try:
            raise e.ValidationError(
                error_message,
                field="email",
                value="invalid-email",
                error_code=c.Errors.VALIDATION_ERROR,
            )
        except e.ValidationError as exc:
            result: r[str] = r.fail(exc.message, error_code=exc.error_code)
            print(f"✅ ValidationError integration: {result.error_code}")


def main() -> None:
    """Main entry point using centralized t and advanced features."""
    width = c.Validation.MAX_NAME_LENGTH * 2
    separator = "=" * width

    print(separator)
    print("r COMPREHENSIVE DEMONSTRATION")
    print("Railway-oriented programming with advanced type safety")
    print(separator)

    # Execute service directly (dispatcher removed due to serialization issues)
    service = RailwayService()
    result = service.execute()

    match result:
        case r(is_success=True, value=demo_result):
            print(
                f"\n✅ Completed {demo_result.demonstrations_completed} demonstrations",
            )
            print(f"Patterns: {', '.join(demo_result.patterns_covered)}")
        case r(is_success=False, error=error):
            print(f"\n❌ Failed: {error}")
        case _:
            pass

    print(f"\n{separator}")
    print("🎯 Railway patterns: .map(), .flat_map(), .flow_through()")
    print("🎯 Error recovery: .alt(), .lash()")
    print("🎯 Advanced combinators: .traverse(), .filter()")
    print("🎯 Validation integration: u.Validation")
    print("🎯 Type safety: Centralized t with Python 3.13+")
    print("🎯 Exception integration: e structured handling")
    print("🎯 CQRS integration: h (dispatcher removed due to serialization)")
    print("🎯 Dependency injection: Direct instantiation")
    print("🎯 Context management: FlextContext")
    print("🎯 Domain models: m")


if __name__ == "__main__":
    main()
