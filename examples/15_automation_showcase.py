"""Automation Decorators Showcase.

Demonstrates context enrichment, FlextResult advanced methods, and automation
patterns using Python 3.13+ PEP 695 type aliases, collections.abc patterns,
Pydantic 2 with StrEnum, and strict type safety - no backward compatibility.

KEY FEATURES:
- Automatic context enrichment in FlextService
- Correlation ID generation for distributed tracing
- User context enrichment for audit trails
- Operation context tracking
- FlextResult advanced methods: from_callable, flow_through, lash, alt

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import PrivateAttr

from flext_core import (
    FlextConstants,
    FlextResult,
    FlextService,
    t,
)
from flext_core.utilities import u

# =============================================================================
# EXAMPLE 1: Service with Context Enrichment
# =============================================================================


class UserService(FlextService[t.Types.ServiceMetadataMapping]):
    """Service demonstrating automatic context enrichment."""

    def __init__(self, **data: t.GeneralValueType) -> None:
        """Initialize with automatic context enrichment.

        FlextService.__init__ automatically calls:
        - _enrich_context() with service metadata
        """
        super().__init__(**data)
        # Context now includes: service_type, service_module

    def execute(self) -> FlextResult[t.Types.ServiceMetadataMapping]:  # noqa: PLR6301  # Required by FlextService abstract method
        """Required abstract method implementation."""
        return FlextResult[t.Types.ServiceMetadataMapping].ok({"status": "initialized"})

    def create_user(
        self, username: str, email: str
    ) -> FlextResult[t.Types.ServiceMetadataMapping]:
        """Create user with automatic context enrichment."""
        # Context includes service metadata from __init__
        if self.logger:
            self.logger.info("Creating user", username=username, email=email)

        # Business logic
        user_data: t.Types.ServiceMetadataMapping = {
            "id": "usr_123",
            "username": username,
            "email": email,
        }

        return FlextResult[t.Types.ServiceMetadataMapping].ok(user_data)


# =============================================================================
# EXAMPLE 2: Context Enrichment with Correlation ID
# =============================================================================


class PaymentService(FlextService[t.Types.ServiceMetadataMapping]):
    """Service demonstrating correlation ID tracking."""

    def __init__(self, **data: t.GeneralValueType) -> None:
        """Initialize with automatic context enrichment."""
        super().__init__(**data)

    def execute(self) -> FlextResult[t.Types.ServiceMetadataMapping]:  # noqa: PLR6301  # Required by FlextService abstract method
        """Required abstract method implementation."""
        return FlextResult[t.Types.ServiceMetadataMapping].ok({"status": "initialized"})

    def process_payment(
        self,
        payment_id: str,
        amount: float,
        user_id: str,
    ) -> FlextResult[t.Types.ServiceMetadataMapping]:
        """Process payment with correlation tracking.

        Demonstrates:
        1. Correlation ID generation for distributed tracing
        2. User context for audit trail
        3. Operation context for tracking
        """
        # Generate correlation ID for distributed tracing
        correlation_id = self._get_correlation_id() or f"payment_{payment_id}_{user_id}"
        self._set_correlation_id(correlation_id)

        # Set user context for audit trail
        self._enrich_context(
            user_id=user_id,
            payment_id=payment_id,
            operation="process_payment",
        )

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
        payment_data: t.Types.ServiceMetadataMapping = {
            "payment_id": payment_id,
            "amount": amount,
            "status": "completed",
            "correlation_id": correlation_id,
        }

        # Clean up operation context
        self._clear_operation_context()

        return FlextResult[t.Types.ServiceMetadataMapping].ok(payment_data)


# =============================================================================
# EXAMPLE 3: Using execute_with_context_enrichment Helper
# =============================================================================


class OrderService(FlextService[t.Types.ServiceMetadataMapping]):
    """Service demonstrating context enrichment helper method."""

    _order_data: t.Types.ConfigurationMapping = PrivateAttr(default_factory=dict)

    def __init__(self, **data: t.GeneralValueType) -> None:
        """Initialize service."""
        super().__init__(**data)

    def execute(self) -> FlextResult[t.Types.ServiceMetadataMapping]:
        """Process order with business logic."""
        order_data_dict: dict[str, t.GeneralValueType] = dict(self._order_data)
        merge_result = u.merge(
            order_data_dict,
            {
                "order_id": u.get(order_data_dict, "order_id", default="ord_123")
                or "ord_123",
                "status": FlextConstants.Domain.Status.PENDING.value,
            },
            strategy="override",
        )
        if merge_result.is_success:
            order_data_dict = merge_result.value
        # Use u.filter to filter dict items with string keys
        filtered_dict = u.filter(
            order_data_dict,
            predicate=lambda k, _v: isinstance(k, str),
        )
        result_data: t.Types.ServiceMetadataMapping = (
            filtered_dict
            if u.guard(filtered_dict, dict, return_value=True) is not None
            else {}
        )
        return FlextResult[t.Types.ServiceMetadataMapping].ok(result_data)

    def process_order(
        self,
        order_id: str,
        customer_id: str,
        correlation_id: str | None = None,
    ) -> FlextResult[t.Types.ServiceMetadataMapping]:
        """Process order with automatic context enrichment."""
        order_data_dict: dict[str, t.GeneralValueType] = dict(self._order_data)
        order_data_dict["order_id"] = order_id
        order_data_dict["customer_id"] = customer_id

        correlation = correlation_id or f"order_{order_id}_{customer_id}"
        self._set_correlation_id(correlation)
        self._enrich_context(
            user_id=customer_id,
            order_id=order_id,
            operation=FlextConstants.Cqrs.Action.CREATE.value,
        )

        with self.track("process_order"):
            return self.execute()


# =============================================================================
# DEMONSTRATION
# =============================================================================


# =============================================================================
# EXAMPLE 4: New FlextResult Methods (v0.9.9+)
# =============================================================================


class AutomationService(FlextService[t.Types.ServiceMetadataMapping]):
    """Service demonstrating the 5 new FlextResult methods in automation context.

    Shows how the new v0.9.9+ methods work with automated workflows:
    - from_callable: Safe automation task execution
    - flow_through: Automation pipeline composition
    - lash: Fallback automation strategies
    - alt: Alternative automation paths
    - unwrap_or pattern: Lazy resource initialization
    """

    def __init__(self, **data: t.GeneralValueType) -> None:
        """Initialize automation service."""
        super().__init__(**data)

    def execute(  # noqa: PLR6301  # Required by FlextService abstract method
        self, **_kwargs: t.GeneralValueType
    ) -> FlextResult[t.Types.ServiceMetadataMapping]:
        """Required abstract method implementation."""
        return FlextResult[t.Types.ServiceMetadataMapping].ok({
            "status": "automation_ready"
        })

    @staticmethod
    def demonstrate_new_flextresult_methods() -> None:
        """Demonstrate the 5 new FlextResult methods in automation context."""
        print("\n" + "=" * 60)
        print("NEW FlextResult METHODS - AUTOMATION CONTEXT")
        print("Demonstrating v0.9.9+ methods with automated workflows")
        print("=" * 60)

        AutomationService._demo_from_callable()
        AutomationService._demo_flow_through()
        AutomationService._demo_lash()
        AutomationService._demo_alt()
        AutomationService._demo_value_or_call()
        AutomationService._demo_advanced_scenarios()

        print("\n" + "=" * 60)
        print("✅ NEW FlextResult METHODS AUTOMATION DEMO COMPLETE!")
        print("All 5 methods + 3 advanced scenarios demonstrated")
        print("=" * 60)

    @staticmethod
    def _demo_from_callable() -> None:
        """Demo 1: from_callable - Safe Automation Task Execution."""
        print("\n=== 1. from_callable: Safe Automation Task Execution ===")

        def risky_automation_task() -> t.Types.ServiceMetadataMapping:
            task_data: t.Types.ServiceMetadataMapping = {
                "task_id": "AUTO-001",
                "task_type": "data_sync",
                "records_processed": 1000,
                "status": "success",
            }
            records = u.get(task_data, "records_processed", default=0) or 0
            if u.guard(records, int, return_value=True) is None or records == 0:
                msg = "No records to process"
                raise ValueError(msg)
            return task_data

        result = FlextResult[t.Types.ServiceMetadataMapping].create_from_callable(
            risky_automation_task
        )
        if result.is_success:
            data = result.unwrap()
            print(f"✅ Automation successful: {data.get('task_type', 'N/A')}")
            print(f"   Records: {data.get('records_processed', 0)}")
        else:
            print(f"❌ Automation failed: {result.error}")

    @staticmethod
    def _demo_flow_through() -> None:
        """Demo 2: flow_through - Automation Pipeline Composition."""
        print("\n=== 2. flow_through: Automation Pipeline Composition ===")

        def validate(
            data: t.Example.UserDataMapping,
        ) -> FlextResult[t.Example.UserDataMapping]:
            task_type = data.get("task_type", "")
            if not isinstance(task_type, str) or not task_type:
                return FlextResult[t.Example.UserDataMapping].fail("Task type required")
            return FlextResult[t.Example.UserDataMapping].ok(data)

        def enrich(
            data: t.Example.UserDataMapping,
        ) -> FlextResult[t.Example.UserDataMapping]:
            enriched: t.Example.UserDataMapping = {
                **data,
                "automation_timestamp": "2025-01-01T12:00:00Z",
                "duration_ms": 250,
                "result_id": "RESULT-001",
            }
            return FlextResult[t.Example.UserDataMapping].ok(enriched)

        def wrap_dict_fn(
            fn: Callable[
                [t.Example.UserDataMapping],
                FlextResult[t.Example.UserDataMapping],
            ],
        ) -> Callable[
            [t.Example.UserDataMapping],
            FlextResult[t.Example.UserDataMapping],
        ]:
            return fn

        automation_input: t.Example.UserDataMapping = {
            "task_type": FlextConstants.Cqrs.ProcessingMode.BATCH.value,
            "source": "database",
        }
        pipeline_result = (
            FlextResult[t.Example.UserDataMapping]
            .ok(automation_input)
            .flow_through(validate, enrich)
        )

        if pipeline_result.is_success:
            data = pipeline_result.unwrap()
            task_type = data.get("task_type", "")
            duration = data.get("duration_ms", 0)
            print(f"✅ Pipeline complete: {task_type}")
            print(f"   Duration: {duration}ms")
        else:
            print(f"❌ Pipeline failed: {pipeline_result.error}")

    @staticmethod
    def _demo_lash() -> None:
        """Demo 3: lash - Fallback Automation Strategies."""
        print("\n=== 3. lash: Fallback Automation Strategies ===")

        def primary() -> FlextResult[str]:
            return FlextResult[str].fail("Primary automation engine unavailable")

        def fallback(error: str) -> FlextResult[str]:
            print(f"   ⚠️  Primary failed: {error}, using fallback...")
            return FlextResult[str].ok("FALLBACK-AUTOMATION-SUCCESS")

        result = primary().lash(fallback)
        if result.is_success:
            print(f"✅ Automation successful: {result.unwrap()}")
        else:
            print(f"❌ All strategies failed: {result.error}")

    @staticmethod
    def _demo_alt() -> None:
        """Demo 4: alt - Alternative Automation Paths."""
        print("\n=== 4. alt: Alternative Automation Paths ===")

        def get_cached() -> FlextResult[t.Types.ServiceMetadataMapping]:
            return FlextResult[t.Types.ServiceMetadataMapping].fail("Cache unavailable")

        def get_default() -> FlextResult[t.Types.ServiceMetadataMapping]:
            return FlextResult[t.Types.ServiceMetadataMapping].ok({
                "automation_mode": FlextConstants.Cqrs.ProcessingMode.SEQUENTIAL.value,
                "batch_size": FlextConstants.Performance.BatchProcessing.DEFAULT_SIZE,
            })

        cached = get_cached()
        config_result = get_default() if cached.is_failure else cached
        if config_result.is_success:
            config = config_result.unwrap()
            mode = config.get("automation_mode", "unknown")
            batch_size = config.get("batch_size", 0)
            print(f"✅ Config acquired: {mode}")
            print(f"   Batch size: {batch_size}")
        else:
            print(f"❌ No config available: {config_result.error}")

    @staticmethod
    def _demo_value_or_call() -> None:
        """Demo 5: value_or_call - Lazy Resource Initialization."""
        print("\n=== 5. value_or_call: Lazy Resource Initialization ===")

        def create_engine() -> t.Types.ServiceMetadataMapping:
            print("   ⚙️  Initializing automation engine...")
            return {
                "engine_id": "AUTO-ENGINE-001",
                "engine_type": FlextConstants.Cqrs.ProcessingMode.PARALLEL.value,
                "worker_count": FlextConstants.Performance.DEFAULT_DB_POOL_SIZE,
            }

        fail_result = FlextResult[t.Types.ServiceMetadataMapping].fail(
            "No existing engine"
        )
        engine = create_engine() if fail_result.is_failure else fail_result.unwrap()
        engine_id = str(engine.get("engine_id", "unknown"))
        worker_count_val = engine.get("worker_count", 0)
        worker_count = (
            int(worker_count_val) if isinstance(worker_count_val, (int, float)) else 0
        )
        print(f"✅ Engine acquired: {engine_id}")
        print(f"   Workers: {worker_count}")

        existing: t.Types.ServiceMetadataMapping = {
            "engine_id": "CACHED-ENGINE-001",
            "worker_count": FlextConstants.Container.DEFAULT_WORKERS,
        }
        success_result = FlextResult[t.Types.ServiceMetadataMapping].ok(existing)
        cached = (
            success_result.unwrap() if success_result.is_success else create_engine()
        )
        cached_id = str(cached.get("engine_id", "unknown"))
        print(f"✅ Existing engine used: {cached_id}")

    @staticmethod
    def _demo_advanced_scenarios() -> None:
        """Demo 6: Advanced Automation Scenarios."""
        print("\n=== 6. ADVANCED AUTOMATION SCENARIOS ===")
        AutomationService._demo_etl_pipeline()
        AutomationService._demo_service_orchestration()
        AutomationService._demo_lazy_config()
        print("\n" + "=" * 60)
        print("✅ ADVANCED AUTOMATION SCENARIOS COMPLETE!")
        print("=" * 60)

    @staticmethod
    def _demo_etl_pipeline() -> None:
        """ETL Pipeline with Error Recovery."""
        print("\n--- Data Pipeline with Error Recovery ---")

        def extract() -> FlextResult[list[t.Example.UserDataMapping]]:
            return FlextResult[list[t.Example.UserDataMapping]].ok([
                {"id": 1, "name": "Item A", "value": 100},
                {"id": 2, "name": "Item B", "value": 200},
            ])

        def transform(
            data: list[t.Example.UserDataMapping],
        ) -> FlextResult[list[t.Example.UserDataMapping]]:
            transformed: list[t.Example.UserDataMapping] = [
                {**item, "processed": True, "timestamp": "2025-01-01T12:00:00Z"}
                for item in data
            ]
            return FlextResult[list[t.Example.UserDataMapping]].ok(transformed)

        def load(
            data: list[t.Example.UserDataMapping],
        ) -> FlextResult[list[t.Example.UserDataMapping]]:
            print(f"   💾 Loaded {len(data)} records successfully")
            return FlextResult[list[t.Example.UserDataMapping]].ok(data)

        result = extract().flow_through(transform, load)
        if result.is_success:
            print(f"✅ ETL Pipeline: {result.unwrap()}")
        else:
            print(f"❌ ETL Pipeline failed: {result.error}")

    @staticmethod
    def _demo_service_orchestration() -> None:
        """Multi-Service Coordination."""
        print("\n--- Multi-Service Coordination ---")

        def start_a() -> FlextResult[str]:
            return FlextResult[str].ok("Service A started")

        def start_backup(error: str) -> FlextResult[str]:
            print(f"   🔄 Service failed: {error}, starting backup...")
            return FlextResult[str].ok("Backup service started")

        def start_b() -> FlextResult[str]:
            return FlextResult[str].ok("B started")

        result = start_a().flow_through(lambda _: start_b()).lash(start_backup)
        if result.is_success:
            print(f"✅ Service Orchestration: {result.unwrap()}")
        else:
            print(f"❌ Service Orchestration failed: {result.error}")

    @staticmethod
    def _demo_lazy_config() -> None:
        """Configuration with Lazy Loading."""
        print("\n--- Configuration with Lazy Loading ---")

        cache: dict[str, t.GeneralValueType] = {}

        def load_config() -> t.Types.ConfigurationMapping:
            if not cache:
                print("   📄 Loading configuration from file...")
                cache.update({
                    "database_url": FlextConstants.Example.DEFAULT_DATABASE_URL,
                    "cache_ttl": FlextConstants.Defaults.DEFAULT_CACHE_TTL,
                })
            return cache

        fail_attempt = FlextResult[t.Types.ConfigurationMapping].fail(
            "No cached config"
        )
        config = load_config() if fail_attempt.is_failure else fail_attempt.unwrap()
        config_count = len(config)
        print(f"✅ Config loaded: {config_count} settings")

        _ = load_config()
        print("✅ Second config access used cached version (no file loading)")


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

    # Example 4: New FlextResult methods (v0.9.9+)
    print("\n4. NEW FlextResult METHODS (v0.9.9+)")
    print("-" * 80)
    automation_service = AutomationService()
    automation_service.demonstrate_new_flextresult_methods()

    print("\n" + "=" * 80)
    print("KEY BENEFITS DEMONSTRATED:")
    print("=" * 80)
    print("✅ Automatic context enrichment in FlextService")
    print("✅ Correlation ID generation for distributed tracing")
    print("✅ User context enrichment for audit trails")
    print("✅ Operation context tracking")
    print("✅ Automatic context cleanup")
    print("✅ Structured logging with full context")
    print(
        "✅ NEW FlextResult methods: from_callable, flow_through, lash, alt, unwrap_or",
    )
    print(
        "✅ Advanced automation scenarios: ETL pipelines, service orchestration, lazy loading",
    )
    print("✅ Zero boilerplate infrastructure code")
    print("=" * 80)


if __name__ == "__main__":
    main()
