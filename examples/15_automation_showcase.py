"""Automation Decorators Showcase.

This example demonstrates the context enrichment capabilities introduced in the
Phase 1 architectural enhancement.

KEY FEATURES DEMONSTRATED:
- Automatic context enrichment in FlextCore.Service and FlextCore.Handlers
- _with_correlation_id: Distributed tracing support
- _with_user_context: User audit trail
- _with_operation_context: Operation tracking
- _enrich_context: Service metadata enrichment

USAGE PATTERNS:
- Context enrichment best practices
- Integration with FlextCore.Service base class
- Structured logging with automatic context

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextCore

# =============================================================================
# EXAMPLE 1: Service with Context Enrichment
# =============================================================================


class UserService(FlextCore.Service[dict[str, object]]):
    """Service demonstrating automatic context enrichment."""

    def __init__(self, **data: object) -> None:
        """Initialize with automatic context enrichment.

        FlextCore.Service.__init__ automatically calls:
        - _enrich_context() with service metadata
        """
        super().__init__(**data)
        # Context now includes: service_type, service_module

    def execute(self) -> FlextCore.Result[dict[str, object]]:
        """Required abstract method implementation."""
        return FlextCore.Result[dict[str, object]].ok({"status": "initialized"})

    def create_user(
        self, username: str, email: str
    ) -> FlextCore.Result[dict[str, object]]:
        """Create user with automatic context enrichment."""
        # Context includes service metadata from __init__
        if self.logger:
            self.logger.info("Creating user", username=username, email=email)

        # Business logic
        user_data: dict[str, object] = {
            "id": "usr_123",
            "username": username,
            "email": email,
        }

        return FlextCore.Result[dict[str, object]].ok(user_data)


# =============================================================================
# EXAMPLE 2: Context Enrichment with Correlation ID
# =============================================================================


class PaymentService(FlextCore.Service[dict[str, object]]):
    """Service demonstrating correlation ID tracking."""

    def __init__(self, **data: object) -> None:
        """Initialize with automatic context enrichment."""
        super().__init__(**data)

    def execute(self) -> FlextCore.Result[dict[str, object]]:
        """Required abstract method implementation."""
        return FlextCore.Result[dict[str, object]].ok({"status": "initialized"})

    def process_payment(
        self,
        payment_id: str,
        amount: float,
        user_id: str,
    ) -> FlextCore.Result[dict[str, object]]:
        """Process payment with correlation tracking.

        Demonstrates:
        1. Correlation ID generation for distributed tracing
        2. User context for audit trail
        3. Operation context for tracking
        """
        # Generate correlation ID for distributed tracing
        correlation_id = self._with_correlation_id()

        # Set user context for audit trail
        self._with_user_context(user_id, payment_id=payment_id)

        # Set operation context
        self._with_operation_context("process_payment", amount=amount)

        # All logs now include full context automatically
        if self.logger:
            self.logger.info(
                "Processing payment",
                payment_id=payment_id,
                amount=amount,
                correlation_id=correlation_id,
            )

        # Business logic
        payment_data: dict[str, object] = {
            "payment_id": payment_id,
            "amount": amount,
            "status": "completed",
            "correlation_id": correlation_id,
        }

        # Clean up operation context
        self._clear_operation_context()

        return FlextCore.Result[dict[str, object]].ok(payment_data)


# =============================================================================
# EXAMPLE 3: Using execute_with_context_enrichment Helper
# =============================================================================


class OrderService(FlextCore.Service[dict[str, object]]):
    """Service demonstrating context enrichment helper method."""

    def __init__(self, **data: object) -> None:
        """Initialize service."""
        super().__init__(**data)
        self._order_data: dict[str, object] = {}

    def execute(self) -> FlextCore.Result[dict[str, object]]:
        """Process order with business logic."""
        # Implement actual order processing
        self._order_data = {
            "order_id": "ord_123",
            "status": "processed",
        }
        return FlextCore.Result[dict[str, object]].ok(self._order_data)

    def process_order(
        self,
        order_id: str,
        customer_id: str,
        correlation_id: str | None = None,
    ) -> FlextCore.Result[dict[str, object]]:
        """Process order with automatic context enrichment.

        Uses execute_with_context_enrichment() helper that:
        - Sets correlation ID
        - Sets operation context
        - Sets user context
        - Tracks performance
        - Logs operation start/complete/error
        - Cleans up context after operation
        """
        # Store order data for execute() to process
        self._order_data = {
            "order_id": order_id,
            "customer_id": customer_id,
        }

        # Use helper method for automatic context enrichment
        return self.execute_with_context_enrichment(
            operation_name="process_order",
            correlation_id=correlation_id,
            user_id=customer_id,
            order_id=order_id,
        )


# =============================================================================
# DEMONSTRATION
# =============================================================================


# =============================================================================
# EXAMPLE 4: New FlextCore.Result Methods (v0.9.9+)
# =============================================================================


class AutomationService(FlextCore.Service[dict[str, object]]):
    """Service demonstrating the 5 new FlextCore.Result methods in automation context.

    Shows how the new v0.9.9+ methods work with automated workflows:
    - from_callable: Safe automation task execution
    - flow_through: Automation pipeline composition
    - lash: Fallback automation strategies
    - alt: Alternative automation paths
    - value_or_call: Lazy resource initialization
    """

    def __init__(self, **data: object) -> None:
        """Initialize automation service."""
        super().__init__(**data)

    def execute(self) -> FlextCore.Result[dict[str, object]]:
        """Required abstract method implementation."""
        return FlextCore.Result[dict[str, object]].ok({"status": "automation_ready"})

    def demonstrate_new_flextresult_methods(self) -> None:
        """Demonstrate the 5 new FlextCore.Result methods in automation context."""
        print("\n" + "=" * 60)
        print("NEW FlextCore.Result METHODS - AUTOMATION CONTEXT")
        print("Demonstrating v0.9.9+ methods with automated workflows")
        print("=" * 60)

        # 1. from_callable - Safe Automation Task Execution
        print("\n=== 1. from_callable: Safe Automation Task Execution ===")

        def risky_automation_task() -> dict[str, object]:
            """Automation task that might fail."""
            # Simulate automated data processing
            task_data: dict[str, object] = {
                "task_id": "AUTO-001",
                "task_type": "data_sync",
                "records_processed": 1000,
                "status": "success",
            }
            # Could raise exception if automation fails
            if task_data.get("records_processed", 0) == 0:
                msg = "No records to process"
                raise ValueError(msg)
            return task_data

        # Safe execution without try/except
        automation_result = FlextCore.Result.from_callable(risky_automation_task)
        if automation_result.is_success:
            data = automation_result.unwrap()
            print(f"✅ Automation successful: {data.get('task_type', 'N/A')}")
            print(f"   Records: {data.get('records_processed', 0)}")
        else:
            print(f"❌ Automation failed: {automation_result.error}")

        # 2. flow_through - Automation Pipeline Composition
        print("\n=== 2. flow_through: Automation Pipeline Composition ===")

        def validate_automation_input(
            data: dict[str, object],
        ) -> FlextCore.Result[dict[str, object]]:
            """Validate automation input."""
            task_type = data.get("task_type", "")
            if not isinstance(task_type, str) or not task_type:
                return FlextCore.Result[dict[str, object]].fail(
                    "Task type is required for automation"
                )
            return FlextCore.Result[dict[str, object]].ok(data)

        def enrich_automation_context(
            data: dict[str, object],
        ) -> FlextCore.Result[dict[str, object]]:
            """Enrich with automation context."""
            enriched: dict[str, object] = {
                **data,
                "automation_timestamp": "2025-01-01T12:00:00Z",
                "automation_engine": "flext-core",
            }
            return FlextCore.Result[dict[str, object]].ok(enriched)

        def execute_automation(
            data: dict[str, object],
        ) -> FlextCore.Result[dict[str, object]]:
            """Execute the automation."""
            executed: dict[str, object] = {
                **data,
                "execution_status": "completed",
                "duration_ms": 250,
            }
            return FlextCore.Result[dict[str, object]].ok(executed)

        def finalize_automation(
            data: dict[str, object],
        ) -> FlextCore.Result[dict[str, object]]:
            """Finalize automation execution."""
            final: dict[str, object] = {
                **data,
                "finalized": True,
                "result_id": "RESULT-001",
            }
            return FlextCore.Result[dict[str, object]].ok(final)

        # Flow through automation pipeline
        automation_input: dict[str, object] = {
            "task_type": "batch_processing",
            "source": "database",
        }
        pipeline_result = (
            FlextCore.Result[dict[str, object]]
            .ok(automation_input)
            .flow_through(
                validate_automation_input,
                enrich_automation_context,
                execute_automation,
                finalize_automation,
            )
        )

        if pipeline_result.is_success:
            final_data = pipeline_result.unwrap()
            print(f"✅ Pipeline complete: {final_data.get('task_type', 'N/A')}")
            print(f"   Duration: {final_data.get('duration_ms', 0)}ms")
            print(f"   Result ID: {final_data.get('result_id', 'N/A')}")
        else:
            print(f"❌ Pipeline failed: {pipeline_result.error}")

        # 3. lash - Fallback Automation Strategies
        print("\n=== 3. lash: Fallback Automation Strategies ===")

        def primary_automation_strategy() -> FlextCore.Result[str]:
            """Primary automation strategy that might fail."""
            return FlextCore.Result[str].fail("Primary automation engine unavailable")

        def fallback_automation_strategy(error: str) -> FlextCore.Result[str]:
            """Fallback automation strategy."""
            print(f"   ⚠️  Primary failed: {error}, using fallback...")
            return FlextCore.Result[str].ok("FALLBACK-AUTOMATION-SUCCESS")

        # Try primary, fall back on error
        strategy_result = primary_automation_strategy().lash(
            fallback_automation_strategy
        )
        if strategy_result.is_success:
            value = strategy_result.unwrap()
            print(f"✅ Automation successful: {value}")
        else:
            print(f"❌ All strategies failed: {strategy_result.error}")

        # 4. alt - Alternative Automation Paths
        print("\n=== 4. alt: Alternative Automation Paths ===")

        def get_cached_automation_config() -> FlextCore.Result[dict[str, object]]:
            """Try to get cached automation config."""
            return FlextCore.Result[dict[str, object]].fail("Cache unavailable")

        def get_default_automation_config() -> FlextCore.Result[dict[str, object]]:
            """Provide default automation config."""
            config: dict[str, object] = {
                "automation_mode": "default",
                "batch_size": 100,
                "retry_attempts": 3,
                "timeout_seconds": 30,
            }
            return FlextCore.Result[dict[str, object]].ok(config)

        # Try cached, fall back to default
        config_result = get_cached_automation_config().alt(
            get_default_automation_config()
        )
        if config_result.is_success:
            config = config_result.unwrap()
            print(f"✅ Config acquired: {config.get('automation_mode', 'unknown')}")
            print(f"   Batch size: {config.get('batch_size', 0)}")
        else:
            print(f"❌ No config available: {config_result.error}")

        # 5. value_or_call - Lazy Resource Initialization
        print("\n=== 5. value_or_call: Lazy Resource Initialization ===")

        def create_automation_engine() -> dict[str, object]:
            """Create automation engine (expensive operation)."""
            print("   ⚙️  Initializing automation engine...")
            return {
                "engine_id": "AUTO-ENGINE-001",
                "engine_type": "distributed",
                "initialized": True,
                "worker_count": 8,
            }

        # Try to get existing engine, create if not available
        engine_fail_result = FlextCore.Result[dict[str, object]].fail(
            "No existing engine"
        )
        engine = engine_fail_result.value_or_call(create_automation_engine)
        print(f"✅ Engine acquired: {engine.get('engine_id', 'unknown')}")
        print(f"   Type: {engine.get('engine_type', 'unknown')}")
        print(f"   Workers: {engine.get('worker_count', 0)}")

        # Try again with successful result (lazy function NOT called)
        existing_engine: dict[str, object] = {
            "engine_id": "CACHED-ENGINE-001",
            "engine_type": "local",
            "initialized": True,
            "worker_count": 4,
        }
        engine_success_result = FlextCore.Result[dict[str, object]].ok(existing_engine)
        engine_cached = engine_success_result.value_or_call(create_automation_engine)
        print(f"✅ Existing engine used: {engine_cached.get('engine_id', 'unknown')}")
        print(f"   Workers: {engine_cached.get('worker_count', 0)}")
        print("   No expensive initialization needed")

        print("\n" + "=" * 60)
        print("✅ NEW FlextCore.Result METHODS AUTOMATION DEMO COMPLETE!")
        print("All 5 methods demonstrated with automation patterns")
        print("=" * 60)


# =============================================================================
# DEMONSTRATION
# =============================================================================


def main() -> None:
    """Demonstrate context enrichment in action."""
    print("=" * 80)
    print("FLEXT-CORE CONTEXT ENRICHMENT SHOWCASE")
    print("=" * 80)

    # Example 1: Basic service with automatic context
    print("\n1. BASIC SERVICE WITH CONTEXT ENRICHMENT")
    print("-" * 80)
    user_service = UserService()
    result1 = user_service.create_user("john_doe", "john@example.com")
    print(f"Result: {result1.value if result1.is_success else result1.error}")

    # Example 2: Payment service with correlation ID
    print("\n2. SERVICE WITH CORRELATION ID TRACKING")
    print("-" * 80)
    payment_service = PaymentService()
    result2 = payment_service.process_payment(
        payment_id="pay_123",
        amount=99.99,
        user_id="usr_456",
    )
    print(f"Result: {result2.value if result2.is_success else result2.error}")

    # Example 3: Order service using helper method
    print("\n3. SERVICE USING CONTEXT ENRICHMENT HELPER")
    print("-" * 80)
    order_service = OrderService()
    result3 = order_service.process_order(
        order_id="ord_123",
        customer_id="cust_456",
        correlation_id="corr_abc123",
    )
    print(f"Result: {result3.value if result3.is_success else result3.error}")

    # Example 4: New FlextCore.Result methods (v0.9.9+)
    print("\n4. NEW FlextCore.Result METHODS (v0.9.9+)")
    print("-" * 80)
    automation_service = AutomationService()
    automation_service.demonstrate_new_flextresult_methods()

    print("\n" + "=" * 80)
    print("KEY BENEFITS DEMONSTRATED:")
    print("=" * 80)
    print("✅ Automatic context enrichment in FlextCore.Service")
    print("✅ Correlation ID generation for distributed tracing")
    print("✅ User context enrichment for audit trails")
    print("✅ Operation context tracking")
    print("✅ Automatic context cleanup")
    print("✅ Structured logging with full context")
    print("✅ Helper method for complete automation")
    print("✅ Zero boilerplate infrastructure code")
    print("=" * 80)


if __name__ == "__main__":
    main()
