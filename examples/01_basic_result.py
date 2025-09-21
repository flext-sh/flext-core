#!/usr/bin/env python3
"""01 - FlextResult Fundamentals: Complete Railway-Oriented Programming.

This example demonstrates the COMPLETE FlextResult[T] API - the foundation
for error handling across the entire FLEXT ecosystem. FlextResult provides
railway-oriented programming that eliminates exceptions in business logic.

Key Concepts Demonstrated:
- Factory methods: .ok() and .fail() with error codes
- Value extraction: .unwrap(), .unwrap_or(), .expect()
- Railway operations: .map(), .flat_map(), .filter()
- Error recovery: .recover(), .recover_with(), .or_else()
- Advanced combinators: .tap(), .zip_with(), .traverse()
- Collection operations: .sequence(), .all_success(), .any_success()
- Validation chaining: .chain_validations(), .validate_all()
- Context and logging: .with_context(), .rescue_with_logging()
- Operators: >>, <<, @, /, %, &, |, ^

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import operator
import warnings
from collections.abc import Callable
from typing import cast

from flext_core import FlextLogger, FlextResult


class ComprehensiveResultService:
    """Service demonstrating ALL FlextResult patterns and methods."""

    def __init__(self) -> None:
        """Initialize with FlextLogger for structured logging."""
        self._logger = FlextLogger(__name__)

    # ========== BASIC OPERATIONS ==========

    def demonstrate_factory_methods(self) -> None:
        """Show all ways to create FlextResult instances."""
        print("\n=== Factory Methods ===")

        # Success creation
        success = FlextResult[str].ok("value")
        print(f"✅ .ok(): {success}")

        # Failure creation with error code
        failure = FlextResult[str].fail(
            "Validation failed",
            error_code="VALIDATION_ERROR",
            error_data={"field": "email"},
        )
        print(f"❌ .fail(): {failure}")

        # Safe call for exception handling
        def risky_operation() -> int:
            return 1 // 0  # Will raise ZeroDivisionError

        from_exc: FlextResult[int] = cast(
            "FlextResult[int]", FlextResult.safe_call(risky_operation)
        )
        print(f"🔥 .safe_call() for exceptions: {from_exc}")

    def demonstrate_value_extraction(self) -> None:
        """Show all ways to extract values from FlextResult."""
        print("\n=== Value Extraction ===")

        success = FlextResult[str].ok("hello")
        failure = FlextResult[str].fail("error")

        # Safe extraction methods
        print(f".unwrap() on success: {success.unwrap()}")
        print(f".unwrap_or('default'): {failure.unwrap_or('default')}")
        print(f".value_or_none: {failure.value_or_none}")

        # API compatibility (CRITICAL for ecosystem)
        print(f".value property: {success.value}")
        # Note: .data property is deprecated but still supported in some versions
        # print(f".data property (legacy): {success.data}")

        # Expect with custom message
        print(f".expect('Must have value'): {success.expect('Must have value')}")

    # ========== RAILWAY OPERATIONS ==========

    def demonstrate_railway_operations(self) -> None:
        """Core railway-oriented programming patterns."""
        print("\n=== Railway Operations ===")

        def validate_length(s: str) -> FlextResult[str]:
            if len(s) < 3:
                return FlextResult[str].fail("Too short")
            return FlextResult[str].ok(s)

        def to_upper(s: str) -> str:
            return s.upper()

        def add_prefix(s: str) -> FlextResult[str]:
            return FlextResult[str].ok(f"PREFIX_{s}")

        # Map: transform success value
        result = FlextResult[str].ok("test").map(to_upper)
        print(f".map(to_upper): {result.unwrap()}")

        # FlatMap: chain operations that return FlextResult
        result = (
            FlextResult[str].ok("hello").flat_map(validate_length).flat_map(add_prefix)
        )
        print(f".flat_map chain: {result.unwrap()}")

        # Filter: conditional success
        filtered_result: FlextResult[int] = (
            FlextResult[int].ok(10).filter(lambda x: x > 5, "Too small")
        )
        print(f".filter(>5): {filtered_result}")

        # Using operators (syntactic sugar)
        result = FlextResult[str].ok("test") >> validate_length >> add_prefix
        print(f">> operator chain: {result}")

    # ========== ERROR RECOVERY ==========

    def demonstrate_error_recovery(self) -> None:
        """Show error recovery patterns."""
        print("\n=== Error Recovery ===")

        failure = FlextResult[str].fail("Initial error")

        # Recover: transform error to success
        recovered = failure.recover(lambda e: f"Recovered from: {e}")
        print(f".recover(): {recovered.unwrap()}")

        # RecoverWith: chain recovery operations
        def try_recovery(error: str) -> FlextResult[str]:
            return FlextResult[str].ok(f"Recovery attempted for: {error}")

        recovered = failure.recover_with(try_recovery)
        print(f".recover_with(): {recovered.unwrap()}")

        # OrElse: provide fallback value
        fallback = failure.or_else(FlextResult[str].ok("fallback"))
        print(f".or_else(): {fallback.unwrap()}")

    # ========== ADVANCED COMBINATORS ==========

    def demonstrate_advanced_combinators(self) -> None:
        """Advanced functional programming patterns."""
        print("\n=== Advanced Combinators ===")

        # Tap: side effects without changing value
        result = (
            FlextResult[int]
            .ok(42)
            .tap(lambda x: print(f"  Tapping success: {x}"))
            .map(lambda x: x * 2)
        )
        print(f".tap() result: {result.unwrap()}")

        # ZipWith: combine two results
        result1 = FlextResult[int].ok(10)
        result2 = FlextResult[int].ok(20)
        combined = result1.zip_with(result2, operator.add)
        print(f".zip_with(): {combined.unwrap()}")

        # Traverse: map and sequence
        items = [1, 2, 3]

        def process(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x * 2)

        traversed = FlextResult.traverse(items, process)
        print(f".traverse(): {traversed.unwrap()}")

    # ========== COLLECTION OPERATIONS ==========

    def demonstrate_collection_operations(self) -> None:
        """Operations on collections of FlextResults."""
        print("\n=== Collection Operations ===")

        results = [
            FlextResult[int].ok(1),
            FlextResult[int].ok(2),
            FlextResult[int].ok(3),
        ]

        # Sequence: convert list of results to result of list
        def sequence_results(
            results_list: list[FlextResult[int]],
        ) -> FlextResult[list[int]]:
            sequenced: FlextResult[list[int]] = FlextResult.sequence(results_list)
            return sequenced

        sequenced = sequence_results(results)
        print(f".sequence(): {sequenced.unwrap()}")

        # AllSuccess: check if all are successful
        all_ok = FlextResult.all_success(*results)  # Unpack list
        print(f".all_success(): {all_ok}")

        # Add a failure
        results.append(FlextResult[int].fail("error"))

        # AnySuccess: check if any is successful
        any_ok = FlextResult.any_success(*results)  # Unpack list
        print(f".any_success(): {any_ok}")

        # CollectSuccesses: get only successful values
        successes = FlextResult.collect_successes(results)
        print(f".collect_successes(): {successes}")

    # ========== VALIDATION PATTERNS ==========

    def demonstrate_validation_chaining(self) -> None:
        """Complex validation scenarios."""
        print("\n=== Validation Chaining ===")

        def validate_not_empty(s: str) -> FlextResult[str]:
            if not s:
                return FlextResult[str].fail("Empty string")
            return FlextResult[str].ok(s)

        def validate_email(s: str) -> FlextResult[str]:
            if "@" not in s:
                return FlextResult[str].fail("Invalid email")
            return FlextResult[str].ok(s)

        def validate_domain(s: str) -> FlextResult[str]:
            if not s.endswith(".com"):
                return FlextResult[str].fail("Must be .com domain")
            return FlextResult[str].ok(s)

        # Chain multiple validations using flat_map
        result = (
            FlextResult[str]
            .ok("test@example.com")
            .flat_map(validate_not_empty)
            .flat_map(validate_email)
            .flat_map(validate_domain)
        )
        print(f".chain_validations(): {result}")

        # Validate all with accumulation - create validators that return FlextResult[None]
        def validate_not_empty_none(s: str) -> FlextResult[None]:
            if not s:
                return FlextResult[None].fail("Empty string")
            return FlextResult[None].ok(None)

        def validate_email_none(s: str) -> FlextResult[None]:
            if "@" not in s:
                return FlextResult[None].fail("Invalid email")
            return FlextResult[None].ok(None)

        def validate_domain_none(s: str) -> FlextResult[None]:
            if not s.endswith(".com"):
                return FlextResult[None].fail("Must be .com domain")
            return FlextResult[None].ok(None)

        all_results = FlextResult.validate_all(
            "test@example.com",
            validate_not_empty_none,
            validate_email_none,
            validate_domain_none,
        )
        print(f".validate_all(): {all_results}")

    # ========== MONADIC OPERATORS ==========

    def demonstrate_operators(self) -> None:
        """All operator overloads for ergonomic usage."""
        print("\n=== Operator Overloads ===")

        def double(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x * 2)

        def add_ten(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x + 10)

        # >> operator (flat_map)
        result = FlextResult[int].ok(5) >> double >> add_ten
        print(f">> (flat_map): 5 >> double >> add_ten = {result.unwrap()}")

        # << operator (map)
        def multiply_by_three(x: int) -> int:
            return x * 3

        mapped_result = FlextResult[int].ok(5) << multiply_by_three
        print(f"<< (map): 5 << (*3) = {mapped_result.unwrap()}")

        # @ operator (applicative)
        func_result = FlextResult[Callable[[int], int]].ok(lambda x: x + 100)
        value_result = FlextResult[int].ok(42)
        applied = func_result @ value_result
        print(f"@ (apply): (+100) @ 42 = {applied.unwrap()}")

        # & operator (combine/and)
        r1 = FlextResult[int].ok(10)
        r2 = FlextResult[int].ok(20)
        combined = r1 & r2
        print(f"& (combine): 10 & 20 = {combined}")

        # | operator (or_else)
        failure = FlextResult[int].fail("error")
        fallback = FlextResult[int].ok(999)
        result = failure.or_else(fallback)
        print(f"| (or_else): failure | 999 = {result.unwrap()}")

    # ========== CONTEXT AND LOGGING ==========

    def demonstrate_context_operations(self) -> None:
        """Context enrichment and logging."""
        print("\n=== Context and Logging ===")

        def risky_operation() -> FlextResult[int]:
            return FlextResult[int].fail("Something went wrong")

        # Add context to errors (takes a function)
        result = risky_operation().with_context(
            lambda err: f"Error for user 12345: {err}"
        )
        print(f".with_context(): {result}")

        # Rescue with logging (logs but keeps failure state)
        def log_error(error: str) -> None:
            self._logger.error(f"Logged error: {error}")

        result = risky_operation().rescue_with_logging(log_error)
        print(f".rescue_with_logging() (logs error): {result}")

        # To recover with a fallback after logging, chain with recover
        result_with_fallback = (
            risky_operation().rescue_with_logging(log_error).recover(lambda _: 0)
        )
        print(f"Logged and recovered: {result_with_fallback}")

    # ========== DEPRECATED PATTERNS (WITH WARNINGS) ==========

    def demonstrate_deprecated_patterns(self) -> None:
        """Show deprecated patterns with proper warnings."""
        print("\n=== ⚠️ DEPRECATED PATTERNS ===")

        # OLD: Manual try/except (DEPRECATED)
        warnings.warn(
            "Manual try/except is DEPRECATED! Use FlextResult.safe_call() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("❌ OLD WAY (try/except):")
        print("try:")
        print("    result = risky_function()")
        print("except Exception as e:")
        print("    handle_error(e)")

        # NEW: FlextResult pattern
        print("\n✅ CORRECT WAY (FlextResult):")

        def risky_function() -> int:
            error_message = "Division by zero"
            raise ZeroDivisionError(error_message)  # Will raise

        result: FlextResult[int] = cast(
            "FlextResult[int]", FlextResult.safe_call(risky_function)
        )
        print(f"FlextResult.safe_call(): {result}")

        # OLD: Multiple return types (DEPRECATED)
        warnings.warn(
            "Returning Optional[T] or Union[T, None] is DEPRECATED! "
            "Always return FlextResult[T].",
            DeprecationWarning,
            stacklevel=2,
        )
        print("\n❌ OLD WAY (Optional return):")
        print("def find_user(id: int) -> Optional[User]:")
        print("    return None  # or User")

        print("\n✅ CORRECT WAY (FlextResult):")
        print("def find_user(id: int) -> FlextResult[User]:")
        print("    return FlextResult[User].fail('Not found')")

        # OLD: Boolean success flags (DEPRECATED)
        warnings.warn(
            "Returning (bool, T) tuples is DEPRECATED! Use FlextResult[T].",
            DeprecationWarning,
            stacklevel=2,
        )
        print("\n❌ OLD WAY (success flag):")
        print("def process() -> tuple[bool, str]:")
        print("    return (False, 'error message')")

        print("\n✅ CORRECT WAY (FlextResult):")
        print("def process() -> FlextResult[str]:")
        print("    return FlextResult[str].fail('error message')")


def main() -> None:
    """Main entry point demonstrating all FlextResult capabilities."""
    service = ComprehensiveResultService()

    print("=" * 60)
    print("FLEXTRESULT COMPLETE API DEMONSTRATION")
    print("Foundation for 32+ FLEXT Ecosystem Projects")
    print("=" * 60)

    # Core patterns
    service.demonstrate_factory_methods()
    service.demonstrate_value_extraction()
    service.demonstrate_railway_operations()

    # Advanced patterns
    service.demonstrate_error_recovery()
    service.demonstrate_advanced_combinators()
    service.demonstrate_collection_operations()

    # Professional patterns
    service.demonstrate_validation_chaining()
    service.demonstrate_operators()
    service.demonstrate_context_operations()

    # Deprecation warnings
    service.demonstrate_deprecated_patterns()

    print("\n" + "=" * 60)
    print("✅ ALL FlextResult methods demonstrated!")
    print("🎯 Next: See 02_dependency_injection.py for FlextContainer patterns")
    print("=" * 60)


if __name__ == "__main__":
    main()
