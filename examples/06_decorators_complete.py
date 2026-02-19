"""FlextDecorators comprehensive demonstration.

Demonstrates inject, log_operation, railway, with_context, retry, timeout,
and combined decorators using Python 3.13+ strict patterns with PEP 695 type aliases.

**Expected Output:**
- Dependency injection examples
- Structured logging demonstrations
- Railway pattern error handling
- Context management patterns
- Retry logic with exponential backoff
- Timeout enforcement
- Combined decorator composition

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from flext_core import (
    FlextContainer,
    FlextDecorators,
    FlextLogger,
    c,
    m,
    r,
    s,
    t,
)
from pydantic import BaseModel

# ═══════════════════════════════════════════════════════════════════
# SERVICE IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════


class DecoratorsService(s[m.ConfigMap]):
    """Service demonstrating FlextDecorators comprehensive features."""

    def execute(
        self,
    ) -> r[m.ConfigMap]:
        """Execute decorators demonstrations."""
        print("Starting decorators demonstration")

        try:
            self._demonstrate_inject()
            self._demonstrate_log_operation()
            self._demonstrate_railway()
            self._demonstrate_with_context()
            self._demonstrate_retry_timeout()
            self._demonstrate_combined()

            return r[m.ConfigMap].ok({
                "decorators_demonstrated": [
                    "inject",
                    "log_operation",
                    "railway",
                    "with_context",
                    "retry",
                    "timeout",
                    "combined",
                ],
                "decorator_categories": 7,
                "features": [
                    "dependency_injection",
                    "structured_logging",
                    "railway_pattern",
                    "context_management",
                    "retry_logic",
                    "timeout_enforcement",
                    "composition",
                ],
            })

        except Exception as e:
            error_msg = f"Decorators demonstration failed: {e}"
            return r[m.ConfigMap].fail(error_msg)

    @staticmethod
    def _demonstrate_inject() -> None:
        """Show dependency injection decorator."""
        print("\n=== Dependency Injection Decorator ===")

        # Setup container
        container = FlextContainer()
        logger = FlextLogger.create_module_logger(__name__)
        # Business Rule: Container accepts any object type including FlextLogger
        # Cast to container.register() compatible type for type checker

        logger_typed: (
            t.GeneralValueType | BaseModel | Callable[..., t.GeneralValueType] | object
        ) = logger
        container.register("logger", logger_typed)

        @FlextDecorators.inject(logger="logger")
        def process_with_logger(message: str) -> str:
            """Process with injected logger."""
            # Logger is injected automatically
            return f"Processed: {message}"

        result = process_with_logger("test message")
        print(f"✅ Injected dependency: {result}")

    @staticmethod
    def _demonstrate_log_operation() -> None:
        """Show log operation decorator."""
        print("\n=== Log Operation Decorator ===")

        @FlextDecorators.log_operation(operation_name="demo_operation")
        def logged_operation(value: int) -> int:
            """Operation with automatic logging."""
            return value * 2

        result = logged_operation(5)
        print(f"✅ Logged operation: 5 → {result}")

        @FlextDecorators.log_operation(track_perf=True)
        def performance_tracked(value: int) -> int:
            """Operation with performance tracking."""
            return value * 3

        result = performance_tracked(4)
        print(f"✅ Performance tracked: 4 → {result}")

    @staticmethod
    def _demonstrate_railway() -> None:
        """Show railway decorator."""
        print("\n=== Railway Decorator ===")

        @FlextDecorators.railway(error_code=c.Errors.VALIDATION_ERROR)
        def railway_operation(value: int) -> r[int]:
            """Operation with railway pattern."""
            if value < 0:
                return r[int].fail("Value must be positive")
            return r[int].ok(value * 2)

        success_result = railway_operation(5)
        if success_result.is_success:
            print(f"✅ Railway success: {success_result.value}")

        failure_result = railway_operation(-1)
        if failure_result.is_failure:
            print(f"✅ Railway failure: {failure_result.error}")

    @staticmethod
    def _demonstrate_with_context() -> None:
        """Show with context decorator."""
        print("\n=== With Context Decorator ===")

        @FlextDecorators.with_context(operation="demo", user_id="test_user")
        def context_operation(value: str) -> str:
            """Operation with context variables."""
            return f"Context operation: {value}"

        result = context_operation("test")
        print(f"✅ Context operation: {result}")

    @staticmethod
    def _demonstrate_retry_timeout() -> None:
        """Show retry and timeout decorators with composition."""
        print("\n=== Retry and Timeout Decorators ===")

        # Retry decorator with exponential backoff
        @FlextDecorators.retry(
            max_attempts=3,
            delay_seconds=0.1,
            backoff_strategy="exponential",
        )
        @FlextDecorators.railway(error_code=c.Errors.CONNECTION_ERROR)
        def unreliable_operation(attempt_count: list[int]) -> r[str]:
            """Operation that may fail initially but succeeds on retry."""
            attempt_count[0] += 1
            if attempt_count[0] < 2:
                return r[str].fail("Transient failure")
            return r[str].ok(f"Success on attempt {attempt_count[0]}")

        attempts = [0]
        result = unreliable_operation(attempts)
        if result.is_success:
            print(f"✅ Retry succeeded: {result.value}")

        # Timeout decorator
        @FlextDecorators.timeout(timeout_seconds=0.5)
        @FlextDecorators.railway(error_code=c.Errors.TIMEOUT_ERROR)
        def fast_operation() -> r[str]:
            """Fast operation that completes within timeout."""
            time.sleep(0.1)  # Quick operation
            return r[str].ok("Operation completed")

        result = fast_operation()
        if result.is_success:
            print(f"✅ Timeout protection: {result.value}")

        # Composition: Retry + Timeout + Railway
        @FlextDecorators.retry(max_attempts=2, delay_seconds=0.05)
        @FlextDecorators.timeout(timeout_seconds=1.0)
        @FlextDecorators.railway(error_code=c.Errors.CONNECTION_ERROR)
        def robust_operation(value: int) -> r[int]:
            """Operation with retry and timeout protection."""
            if value < 0:
                return r[int].fail("Value must be positive")
            return r[int].ok(value * 2)

        result = robust_operation(5)
        if result.is_success:
            print(f"✅ Retry + Timeout composition: {result.value}")

    @staticmethod
    def _demonstrate_combined() -> None:
        """Show combined decorator."""
        print("\n=== Combined Decorator ===")

        # Setup container for combined decorator
        container = FlextContainer()
        logger = FlextLogger.create_module_logger(__name__)
        # Business Rule: Container accepts any object type including FlextLogger
        # Cast to container.register() compatible type for type checker

        logger_typed: (
            t.GeneralValueType | BaseModel | Callable[..., t.GeneralValueType] | object
        ) = logger
        container.register("logger", logger_typed)

        @FlextDecorators.combined(
            inject_deps={"logger": "logger"},
            operation_name="combined_demo",
            track_perf=True,
            use_railway=True,
        )
        def combined_operation(value: int) -> r[int]:
            """Operation with all decorators combined."""
            # Logger is injected automatically
            if value < 0:
                return r[int].fail("Value must be positive")
            return r[int].ok(value * 2)

        result = combined_operation(6)
        if result.is_success:
            print(f"✅ Combined decorator: {result.value}")


def main() -> None:
    """Main entry point."""
    print("=" * 60)
    print("FLEXT DECORATORS - COMPREHENSIVE DEMONSTRATION")
    print("Inject, log_operation, railway, with_context, retry, timeout, combined")
    print("=" * 60)

    service = DecoratorsService()
    result = service.execute()

    if result.is_success:
        data = result.value
        decorators = data["decorators_demonstrated"]
        categories = data["decorator_categories"]
        if isinstance(decorators, Sequence) and isinstance(categories, int):
            decorators_list = list(decorators)
            print(f"\n✅ Demonstrated {categories} decorator categories")
            print(f"✅ Covered {len(decorators_list)} decorator types")
    else:
        print(f"\n❌ Failed: {result.error}")

    print("\n" + "=" * 60)
    print(
        "🎯 Decorator Patterns: Inject, Log, Railway, Context, Retry, Timeout, Combined",
    )
    print(
        "🎯 Cross-Cutting Concerns: DI, Logging, Error Handling, Context, Reliability",
    )
    print("🎯 Composition: Retry + Timeout + Railway working together")
    print("🎯 Python 3.13+: PEP 695 type aliases, collections.abc")
    print("=" * 60)


if __name__ == "__main__":
    main()
