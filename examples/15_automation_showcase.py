"""Automation Decorators Showcase.

Demonstrates context enrichment, r advanced methods, and automation
patterns using Python 3.13+ PEP 695 type aliases, collections.abc patterns,
Pydantic 2 with StrEnum, and strict type safety - no backward compatibility.

**Expected Output:**
- Automatic context enrichment in s
- Correlation ID generation for distributed tracing
- User context enrichment for audit trails
- Operation context tracking
- r advanced methods: from_callable, flow_through, lash, alt
- Service execution with automatic context propagation
- Error recovery and fallback patterns

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import (
    FlextContext,
    c,
    m,
    r,
    s,
    t,
    u,
)
from pydantic import PrivateAttr

# =============================================================================
# EXAMPLE 1: Service with Context Enrichment
# =============================================================================


class UserService(s[m.ConfigMap]):
    """Service demonstrating automatic context enrichment."""

    def __init__(self, **data: t.GeneralValueType) -> None:
        """Initialize with automatic context enrichment.

        s.__init__ automatically calls:
        - _enrich_context() with service metadata
        """
        super().__init__(**data)
        # Context now includes: service_type, service_module

    def execute(self) -> r[m.ConfigMap]:
        """Required abstract method implementation."""
        return r[m.ConfigMap].ok({"status": "initialized"})

    def create_user(
        self,
        username: str,
        email: str,
    ) -> r[m.ConfigMap]:
        """Create user with automatic context enrichment."""
        # Context includes service metadata from __init__
        if self.logger:
            self.logger.info("Creating user", username=username, email=email)

        # Business logic
        user_data: m.ConfigMap = {
            "id": "usr_123",
            "username": username,
            "email": email,
        }

        return r[m.ConfigMap].ok(user_data)


# =============================================================================
# EXAMPLE 2: Context Enrichment with Correlation ID
# =============================================================================


class PaymentService(s[m.ConfigMap]):
    """Service demonstrating correlation ID tracking."""

    def __init__(self, **data: t.GeneralValueType) -> None:
        """Initialize with automatic context enrichment."""
        super().__init__(**data)

    def execute(self) -> r[m.ConfigMap]:
        """Required abstract method implementation."""
        return r[m.ConfigMap].ok({"status": "initialized"})

    def process_payment(
        self,
        payment_id: str,
        amount: float,
        user_id: str,
    ) -> r[m.ConfigMap]:
        """Process payment with correlation tracking.

        Demonstrates:
        1. Correlation ID generation for distributed tracing
        2. User context for audit trail
        3. Operation context for tracking
        """
        # Generate correlation ID for distributed tracing
        correlation_id = (
            FlextContext.Correlation.get_correlation_id()
            or f"payment_{payment_id}_{user_id}"
        )
        FlextContext.Correlation.set_correlation_id(correlation_id)

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
        payment_data: m.ConfigMap = {
            "payment_id": payment_id,
            "amount": amount,
            "status": "completed",
            "correlation_id": correlation_id,
        }

        # Clean up operation context
        self._clear_operation_context()

        return r[m.ConfigMap].ok(payment_data)


# =============================================================================
# EXAMPLE 3: Using execute_with_context_enrichment Helper
# =============================================================================


class OrderService(s[m.ConfigMap]):
    """Service demonstrating context enrichment helper method."""

    _order_data: m.ConfigMap = PrivateAttr(default_factory=dict)

    def __init__(self, **data: t.GeneralValueType) -> None:
        """Initialize service."""
        super().__init__(**data)

    def execute(self) -> r[m.ConfigMap]:
        """Process order with business logic."""
        order_data_dict: dict[str, t.GeneralValueType] = dict(self._order_data)
        # Simple merge: new values override existing ones
        merged: dict[str, t.GeneralValueType] = dict(order_data_dict)
        merged.update(
            {
                "order_id": u.get(order_data_dict, "order_id", default="ord_123")
                or "ord_123",
                "status": c.Cqrs.CommonStatus.PENDING.value,
            },
        )
        order_data_dict = merged

        def is_string_key(_k: str, _v: t.GeneralValueType) -> bool:
            # k is already typed as str from dict[str, t.GeneralValueType]
            return True

        # Use u.filter to filter dict items with string keys
        filtered_dict = u.filter_dict(
            order_data_dict,
            predicate=is_string_key,
        )
        result_data: m.ConfigMap = (
            filtered_dict
            if u.guard(filtered_dict, dict, return_value=True) is not None
            else {}
        )
        return r[m.ConfigMap].ok(result_data)

    def process_order(
        self,
        order_id: str,
        customer_id: str,
        correlation_id: str | None = None,
    ) -> r[m.ConfigMap]:
        """Process order with automatic context enrichment."""
        order_data_dict: dict[str, t.GeneralValueType] = dict(self._order_data)
        order_data_dict["order_id"] = order_id
        order_data_dict["customer_id"] = customer_id

        correlation = correlation_id or f"order_{order_id}_{customer_id}"
        FlextContext.Correlation.set_correlation_id(correlation)
        self._enrich_context(
            user_id=customer_id,
            order_id=order_id,
            operation=c.Cqrs.Action.CREATE.value,
        )

        with self.track("process_order"):
            return self.execute()


# =============================================================================
# DEMONSTRATION
# =============================================================================


# =============================================================================
# EXAMPLE 4: New r Methods (v0.9.9+)
# =============================================================================


class AutomationService(s[m.ConfigMap]):
    """Service demonstrating the 5 new r methods in automation context.

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

    def execute(
        self,
        **_kwargs: t.GeneralValueType,
    ) -> r[m.ConfigMap]:
        """Required abstract method implementation."""
        return r[m.ConfigMap].ok({
            "status": "automation_ready",
        })

    @staticmethod
    def demonstrate_new_r_methods() -> None:
        """Demonstrate the 5 new r methods in automation context."""
        print("\n" + "=" * 60)
        print("NEW r METHODS - AUTOMATION CONTEXT")
        print("Demonstrating v0.9.9+ methods with automated workflows")
        print("=" * 60)

        AutomationService._demo_from_callable()
        AutomationService._demo_flow_through()
        AutomationService._demo_lash()
        AutomationService._demo_alt()
        AutomationService._demo_value_or_call()
        AutomationService._demo_advanced_scenarios()

        print("\n" + "=" * 60)
        print("✅ NEW r METHODS AUTOMATION DEMO COMPLETE!")
        print("All 5 methods + 3 advanced scenarios demonstrated")
        print("=" * 60)

    @staticmethod
    def _demo_from_callable() -> None:
        """Demo 1: from_callable - Safe Automation Task Execution."""
        print("\n=== 1. from_callable: Safe Automation Task Execution ===")

        def risky_automation_task() -> m.ConfigMap:
            task_data: m.ConfigMap = {
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

        result = r[m.ConfigMap].create_from_callable(
            risky_automation_task,
        )
        if result.is_success:
            data = result.value
            print(f"✅ Automation successful: {data.get('task_type', 'N/A')}")
            print(f"   Records: {data.get('records_processed', 0)}")
        else:
            print(f"❌ Automation failed: {result.error}")

    @staticmethod
    def _demo_flow_through() -> None:
        """Demo 2: flow_through - Automation Pipeline Composition."""
        print("\n=== 2. flow_through: Automation Pipeline Composition ===")

        def validate(
            data: m.ConfigMap,
        ) -> r[m.ConfigMap]:
            task_type = data.get("task_type", "")
            if not isinstance(task_type, str) or not task_type:
                return r[m.ConfigMap].fail("Task type required")
            return r[m.ConfigMap].ok(data)

        def enrich(
            data: m.ConfigMap,
        ) -> r[m.ConfigMap]:
            enriched: m.ConfigMap = {
                **data,
                "automation_timestamp": "2025-01-01T12:00:00Z",
                "duration_ms": 250,
                "result_id": "RESULT-001",
            }
            return r[m.ConfigMap].ok(enriched)

        automation_input: m.ConfigMap = {
            "task_type": c.Cqrs.ProcessingMode.BATCH.value,
            "source": "database",
        }
        pipeline_result = (
            r[m.ConfigMap].ok(automation_input).flow_through(validate, enrich)
        )

        if pipeline_result.is_success:
            data = pipeline_result.value
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

        def primary() -> r[str]:
            return r[str].fail("Primary automation engine unavailable")

        def fallback(error: str) -> r[str]:
            print(f"   ⚠️  Primary failed: {error}, using fallback...")
            return r[str].ok("FALLBACK-AUTOMATION-SUCCESS")

        result = primary().lash(fallback)
        if result.is_success:
            print(f"✅ Automation successful: {result.value}")
        else:
            print(f"❌ All strategies failed: {result.error}")

    @staticmethod
    def _demo_alt() -> None:
        """Demo 4: alt - Alternative Automation Paths."""
        print("\n=== 4. alt: Alternative Automation Paths ===")

        def get_cached() -> r[m.ConfigMap]:
            return r[m.ConfigMap].fail("Cache unavailable")

        def get_default() -> r[m.ConfigMap]:
            return r[m.ConfigMap].ok({
                "automation_mode": c.Cqrs.ProcessingMode.SEQUENTIAL.value,
                "batch_size": c.Performance.BatchProcessing.DEFAULT_SIZE,
            })

        cached = get_cached()
        config_result = get_default() if cached.is_failure else cached
        if config_result.is_success:
            config = config_result.value
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

        def create_engine() -> m.ConfigMap:
            print("   ⚙️  Initializing automation engine...")
            return {
                "engine_id": "AUTO-ENGINE-001",
                "engine_type": c.Cqrs.ProcessingMode.PARALLEL.value,
                "worker_count": c.Performance.DEFAULT_DB_POOL_SIZE,
            }

        fail_result = r[m.ConfigMap].fail(
            "No existing engine",
        )
        engine = create_engine() if fail_result.is_failure else fail_result.value
        engine_id = str(engine.get("engine_id", "unknown"))
        worker_count_val = engine.get("worker_count", 0)
        worker_count = (
            int(worker_count_val) if type(worker_count_val) in {int, float} else 0
        )
        print(f"✅ Engine acquired: {engine_id}")
        print(f"   Workers: {worker_count}")

        existing: m.ConfigMap = {
            "engine_id": "CACHED-ENGINE-001",
            "worker_count": c.Container.DEFAULT_WORKERS,
        }
        success_result = r[m.ConfigMap].ok(existing)
        cached = success_result.map_or(create_engine())
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

        def extract() -> r[list[m.ConfigMap]]:
            return r[list[m.ConfigMap]].ok([
                {"id": 1, "name": "Item A", "value": 100},
                {"id": 2, "name": "Item B", "value": 200},
            ])

        def transform(
            data: list[m.ConfigMap],
        ) -> r[list[m.ConfigMap]]:
            transformed: list[m.ConfigMap] = [
                {**item, "processed": True, "timestamp": "2025-01-01T12:00:00Z"}
                for item in data
            ]
            return r[list[m.ConfigMap]].ok(transformed)

        def load(
            data: list[m.ConfigMap],
        ) -> r[list[m.ConfigMap]]:
            print(f"   💾 Loaded {len(data)} records successfully")
            return r[list[m.ConfigMap]].ok(data)

        result = extract().flow_through(transform, load)
        if result.is_success:
            print(f"✅ ETL Pipeline: {result.value}")
        else:
            print(f"❌ ETL Pipeline failed: {result.error}")

    @staticmethod
    def _demo_service_orchestration() -> None:
        """Multi-Service Coordination."""
        print("\n--- Multi-Service Coordination ---")

        def start_a() -> r[str]:
            return r[str].ok("Service A started")

        def start_backup(error: str) -> r[str]:
            print(f"   🔄 Service failed: {error}, starting backup...")
            return r[str].ok("Backup service started")

        def start_b() -> r[str]:
            return r[str].ok("B started")

        result = start_a().flow_through(lambda _: start_b()).lash(start_backup)
        if result.is_success:
            print(f"✅ Service Orchestration: {result.value}")
        else:
            print(f"❌ Service Orchestration failed: {result.error}")

    @staticmethod
    def _demo_lazy_config() -> None:
        """Configuration with Lazy Loading."""
        print("\n--- Configuration with Lazy Loading ---")

        cache: dict[str, t.GeneralValueType] = {}

        def load_config() -> m.ConfigMap:
            if not cache:
                print("   📄 Loading configuration from file...")
                cache.update({
                    "database_url": "postgresql://localhost:5432/testdb",
                    "cache_ttl": c.Defaults.DEFAULT_CACHE_TTL,
                })
            return cache

        fail_attempt = r[m.ConfigMap].fail(
            "No cached config",
        )
        config = load_config() if fail_attempt.is_failure else fail_attempt.value
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

    # Example 4: New r methods (v0.9.9+)
    print("\n4. NEW r METHODS (v0.9.9+)")
    print("-" * 80)
    automation_service = AutomationService()
    automation_service.demonstrate_new_r_methods()

    print("\n" + "=" * 80)
    print("KEY BENEFITS DEMONSTRATED:")
    print("=" * 80)
    print("✅ Automatic context enrichment in s")
    print("✅ Correlation ID generation for distributed tracing")
    print("✅ User context enrichment for audit trails")
    print("✅ Operation context tracking")
    print("✅ Automatic context cleanup")
    print("✅ Structured logging with full context")
    print(
        "✅ NEW r methods: from_callable, flow_through, lash, alt, unwrap_or",
    )
    print(
        "✅ Advanced automation scenarios: ETL pipelines, service orchestration, lazy loading",
    )
    print("✅ Zero boilerplate infrastructure code")
    print("=" * 80)


if __name__ == "__main__":
    main()
