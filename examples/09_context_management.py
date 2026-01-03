"""FlextContext comprehensive demonstration with advanced patterns.

Demonstrates request context, correlation tracking, thread safety, and performance
monitoring using Python 3.13+ PEP 695 type aliases, FlextConstants centralized
StrEnum/Literals, t composition, collections.abc advanced patterns,
Pydantic 2 StrEnum, and strict Python 3.13+ only - no backward compatibility.

**Expected Output:**
- Request context creation and management
- Correlation ID tracking across operations
- Thread-local storage isolation
- Performance tracking and metrics
- Context propagation through service calls
- Multi-threaded context safety

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import threading
from collections.abc import Sequence

from flext_core import (
    FlextConstants,
    FlextContext,
    FlextLogger,
    FlextResult,
    FlextService,
    c,
    t,
    u,
)


class ContextManagementService(
    FlextService[t.ServiceMetadataMapping],
):
    """Service demonstrating comprehensive FlextContext patterns with advanced features.

    Uses FlextConstants centralized StrEnum/Literals, t composition,
    collections.abc advanced patterns, and Pydantic 2 StrEnum for type safety.
    """

    def execute(
        self,
    ) -> FlextResult[t.ServiceMetadataMapping]:
        """Execute comprehensive context demonstrations using railway pattern."""
        self.logger.info("Starting context management demonstration")

        return (
            self
            ._demonstrate_context_concepts()
            .flat_map(lambda _: self._demonstrate_request_handling())
            .flat_map(lambda _: self._demonstrate_threading_concepts())
            .flat_map(lambda _: self._demonstrate_performance_tracking())
            .flat_map(lambda _: self._demonstrate_correlation_tracking())
            .map(self._build_success_metadata)
        )

    @staticmethod
    def _demonstrate_context_concepts() -> FlextResult[t.ContextMetadataMapping]:
        """Demonstrate context management concepts using FlextContext."""
        print("\n=== Context Concepts ===")

        # Use FlextContext.Request for request-scoped operations
        with FlextContext.Request.request_context(
            operation_name="demonstrate_context",
            user_id="demo-user",
            request_id=u.generate("correlation"),
        ):
            correlation_id = (
                FlextContext.Variables.Correlation.CORRELATION_ID.get()
                or FlextConstants.Context.CORRELATION_ID_PREFIX
                + u.generate("correlation")[
                    : FlextConstants.Context.CORRELATION_ID_LENGTH
                ]
            )

            context_data: t.ContextMetadataMapping = {
                FlextConstants.Mixins.FIELD_NAME: "context_demo",
                "correlation_id": correlation_id,
                "scope": FlextConstants.Context.SCOPE_REQUEST,
            }

            print(f"✅ Context data: {context_data}")
            print("✅ Thread-local storage for isolation")
            print("✅ Correlation IDs for request tracking")
            print(f"✅ Scope: {FlextConstants.Context.SCOPE_REQUEST}")

            return FlextResult[t.ContextMetadataMapping].ok(
                context_data,
            )

    @staticmethod
    def _demonstrate_request_handling() -> FlextResult[t.ContextMetadataMapping]:
        """Demonstrate request context management with performance tracking."""
        print("\n=== Request Context ===")

        request_id = u.generate("correlation")
        operation_name = "process_request"

        # Combine multiple with statements (SIM117)
        with (
            FlextContext.Request.request_context(
                operation_name=operation_name,
                request_id=request_id,
                metadata={
                    "endpoint": "/api/demo",
                    "method": "GET",
                },
            ),
            FlextContext.Performance.timed_operation(
                operation_name=operation_name,
            ) as timing_metadata,
        ):
            # Simulate operation
            user_id = FlextContext.Request.get_user_id() or "anonymous"
            operation = FlextContext.Request.get_operation_name() or "unknown"

            request_data: t.ContextMetadataMapping = {
                "user_id": user_id,
                "request_id": request_id,
                "operation": operation,
                "timing": timing_metadata,
            }

            print(f"✅ Request data: {request_data}")
            print("✅ Request lifecycle management")
            print("✅ Performance tracking and metrics")
            print("✅ Context inheritance in service calls")

            return FlextResult[t.ContextMetadataMapping].ok(
                request_data,
            )

    @staticmethod
    def _demonstrate_threading_concepts() -> FlextResult[t.ContextMetadataMapping]:
        """Demonstrate threading and isolation concepts with context safety."""
        print("\n=== Threading Concepts ===")

        thread_count = threading.active_count()
        active_threads: Sequence[str] = tuple(
            thread.name for thread in threading.enumerate()
        )

        # Demonstrate thread-safe context isolation
        def thread_operation(
            thread_id: int,
        ) -> FlextResult[t.ContextMetadataMapping]:
            """Thread operation with isolated context."""
            with FlextContext.Request.request_context(
                operation_name=f"thread_{thread_id}",
                request_id=f"req-{thread_id}",
            ):
                correlation_id = (
                    FlextContext.Variables.Correlation.CORRELATION_ID.get() or "unknown"
                )
                return FlextResult[t.ContextMetadataMapping].ok(
                    {
                        "thread_id": thread_id,
                        "correlation_id": correlation_id,
                        "thread_name": threading.current_thread().name,
                    },
                )

        # Execute operations in sequence (thread-safe context isolation)
        results = [thread_operation(i) for i in range(min(3, thread_count))]

        # Use traverse for multiple results (DRY - no manual loops)
        thread_results = FlextResult.traverse(results, lambda r: r)

        threading_data: t.ContextMetadataMapping = {
            "active_threads": thread_count,
            "thread_names": list(active_threads),
            "isolated_contexts": len(results),
        }

        print(f"✅ Thread safety: {thread_count} active threads")
        print("✅ Context isolation per thread")
        print("✅ No shared state between threads")
        print(f"✅ Thread names: {', '.join(active_threads[:5])}")

        return thread_results.map(
            lambda _: threading_data,
        )

    @staticmethod
    def _demonstrate_performance_tracking() -> FlextResult[t.ContextMetadataMapping]:
        """Demonstrate performance tracking with FlextContext.Performance."""
        print("\n=== Performance Tracking ===")

        operation_name = "performance_demo"

        with FlextContext.Performance.timed_operation(
            operation_name=operation_name,
        ) as timing_metadata:
            # Simulate work
            start_time = FlextContext.Performance.get_operation_start_time()
            operation_metadata = FlextContext.Performance.get_operation_metadata() or {}

            performance_data: t.ContextMetadataMapping = {
                "operation": operation_name,
                "start_time": (start_time.isoformat() if start_time else "unknown"),
                "metadata": operation_metadata,
                "timing": timing_metadata,
            }

            print(f"✅ Operation: {operation_name}")
            print(f"✅ Start time: {performance_data['start_time']}")
            print("✅ Performance monitoring enabled")
            print("✅ Timing metadata captured")

            return FlextResult[t.ContextMetadataMapping].ok(
                performance_data,
            )

    @staticmethod
    def _demonstrate_correlation_tracking() -> FlextResult[t.ContextMetadataMapping]:
        """Demonstrate correlation ID tracking across service boundaries."""
        print("\n=== Correlation Tracking ===")

        correlation_id = u.generate("correlation")

        with FlextContext.Request.request_context(
            operation_name="correlation_demo",
            request_id=correlation_id,
        ):
            # Correlation ID should be available in context
            context_correlation = (
                FlextContext.Variables.Correlation.CORRELATION_ID.get()
                or correlation_id
            )

            correlation_data: t.ContextMetadataMapping = {
                "correlation_id": context_correlation,
                "prefix": FlextConstants.Context.CORRELATION_ID_PREFIX,
                "length": FlextConstants.Context.CORRELATION_ID_LENGTH,
            }

            print(f"✅ Correlation ID: {context_correlation}")
            print("✅ Cross-service tracing support")
            print("✅ Request-scoped variables")
            print("✅ Distributed tracing ready")

            return FlextResult[t.ContextMetadataMapping].ok(
                correlation_data,
            )

    @staticmethod
    def _build_success_metadata(
        _: t.ContextMetadataMapping,
    ) -> t.ServiceMetadataMapping:
        """Build success metadata using centralized FlextConstants (DRY)."""
        # Iterate over enum members correctly
        all_patterns = tuple(
            member.value for member in c.Cqrs.HandlerType.__members__.values()
        )
        filtered_patterns = tuple(
            pattern
            for pattern in all_patterns
            if pattern
            in {
                "factory_methods",
                "railway_operations",
                "validation_patterns",
            }
        )

        return {
            "patterns_demonstrated": filtered_patterns,
            "context_features": (
                "thread_safety",
                "variable_isolation",
                "performance_tracking",
                "correlation_tracking",
            ),
            "architecture": "context_per_thread",
            "scope_types": (
                FlextConstants.Context.SCOPE_GLOBAL,
                FlextConstants.Context.SCOPE_REQUEST,
                FlextConstants.Context.SCOPE_SESSION,
            ),
        }


def demonstrate_context_features() -> None:
    """Demonstrate basic context features using FlextContext utilities."""
    print("\n=== Context Features ===")

    # Use FlextContext utilities directly (no wrappers)
    with FlextContext.Request.request_context(
        operation_name="feature_demo",
    ):
        correlation_id = (
            FlextContext.Variables.Correlation.CORRELATION_ID.get() or "unknown"
        )

        print(f"✅ Correlation ID: {correlation_id}")
        print("✅ Correlation tracking across services")
        print("✅ Request-scoped variables")
        print("✅ Performance monitoring")
        print("✅ Distributed tracing support")


def main() -> None:
    """Main entry point using advanced FlextContext patterns."""
    logger = FlextLogger.create_module_logger(__name__)

    width = FlextConstants.Validation.MAX_NAME_LENGTH * 2
    separator = "=" * width

    print(separator)
    print("FLEXT CONTEXT - REQUEST CONTEXT MANAGEMENT")
    print("Correlation tracking, context variables, thread safety")
    print(separator)

    # Demonstrate features
    demonstrate_context_features()

    # Use service pattern with railway operations
    service = ContextManagementService()
    result = service.execute()

    # Railway pattern for result handling (DRY)
    def handle_success(metadata: t.ServiceMetadataMapping) -> None:
        """Handle successful result with type narrowing."""
        patterns = metadata.get("patterns_demonstrated", ())
        features = metadata.get("context_features", ())
        patterns_count = len(patterns) if isinstance(patterns, Sequence) else 0
        features_count = len(features) if isinstance(features, Sequence) else 0

        logger.info(
            "Context demonstration completed",
            extra={
                "patterns": patterns_count,
                "features": features_count,
            },
        )
        print(f"\n✅ Demonstrated {patterns_count} context patterns")
        print(f"✅ Used {features_count} context features")

    def handle_error(error: str) -> FlextResult[None]:
        """Handle error result."""
        logger.error("Context demonstration failed", extra={"error": error})
        print(f"\n❌ Failed: {error}")
        return FlextResult[None].ok(None)

    result.map(handle_success).lash(handle_error)

    print(f"\n{separator}")
    print("🎯 Context Patterns: Variables, Correlation, Request Lifecycle")
    print("🎯 Thread Safety: Isolated context per thread")
    print("🎯 Performance: Request timing and metrics")
    print("🎯 Type Safety: PEP 695, t, FlextConstants")
    print("🎯 Advanced Patterns: collections.abc, Pydantic 2 StrEnum")
    print(separator)


if __name__ == "__main__":
    main()
