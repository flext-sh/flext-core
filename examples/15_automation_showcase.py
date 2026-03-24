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

from collections.abc import Sequence
from typing import override

from pydantic import PrivateAttr

from flext_core import FlextContext, FlextRuntime, c, r, s, t, u


class UserService(s[t.ConfigMap]):
    """Service demonstrating automatic context enrichment."""

    def create_user(self, username: str, email: str) -> r[t.ConfigMap]:
        """Create user with automatic context enrichment."""
        if self.logger:
            self.logger.info("Creating user", username=username, email=email)
        user_data: t.ConfigMap = t.ConfigMap(
            root={"id": "usr_123", "username": username, "email": email},
        )
        return r[t.ConfigMap].ok(user_data)

    @override
    def execute(self) -> r[t.ConfigMap]:
        """Required abstract method implementation."""
        return r[t.ConfigMap].ok(t.ConfigMap(root={"status": "initialized"}))


class PaymentService(s[t.ConfigMap]):
    """Service demonstrating correlation ID tracking."""

    @override
    def execute(self) -> r[t.ConfigMap]:
        """Required abstract method implementation."""
        return r[t.ConfigMap].ok(t.ConfigMap(root={"status": "initialized"}))

    def process_payment(
        self,
        payment_id: str,
        amount: float,
        user_id: str,
    ) -> r[t.ConfigMap]:
        """Process payment with correlation tracking.

        Demonstrates:
        1. Correlation ID generation for distributed tracing
        2. User context for audit trail
        3. Operation context for tracking
        """
        correlation_id = (
            FlextContext.Correlation.get_correlation_id()
            or f"payment_{payment_id}_{user_id}"
        )
        FlextContext.Correlation.set_correlation_id(correlation_id)
        self._enrich_context(
            user_id=user_id,
            payment_id=payment_id,
            operation="process_payment",
        )
        self._with_operation_context("process_payment", amount=amount)
        if self.logger:
            self.logger.info(
                "Processing payment",
                payment_id=payment_id,
                amount=amount,
                correlation_id=correlation_id,
            )
        payment_data: t.ConfigMap = t.ConfigMap(
            root={
                "payment_id": payment_id,
                "amount": amount,
                "status": "completed",
                "correlation_id": correlation_id,
            },
        )
        self._clear_operation_context()
        return r[t.ConfigMap].ok(payment_data)


class OrderService(s[t.ConfigMap]):
    """Service demonstrating context enrichment helper method."""

    _order_data: t.ConfigMap = PrivateAttr(default_factory=lambda: t.ConfigMap(root={}))

    @override
    def execute(self) -> r[t.ConfigMap]:
        """Process order with business logic."""
        order_id_raw = self._order_data.get("order_id", "ord_123")
        order_id = str(order_id_raw) if order_id_raw else "ord_123"
        result_data = t.ConfigMap(
            root={"order_id": order_id, "status": c.CommonStatus.PENDING.value},
        )
        return r[t.ConfigMap].ok(result_data)

    def process_order(
        self,
        order_id: str,
        customer_id: str,
        correlation_id: str | None = None,
    ) -> r[t.ConfigMap]:
        """Process order with automatic context enrichment."""
        self._order_data = t.ConfigMap(
            root={"order_id": order_id, "customer_id": customer_id},
        )
        correlation = correlation_id or f"order_{order_id}_{customer_id}"
        FlextContext.Correlation.set_correlation_id(correlation)
        self._enrich_context(
            user_id=customer_id,
            order_id=order_id,
            operation=c.Action.CREATE.value,
        )
        with self.track("process_order"):
            return self.execute()


class AutomationService(s[t.ConfigMap]):
    """Service demonstrating the 5 new r methods in automation context.

    Shows how the new v0.9.9+ methods work with automated workflows:
    - from_callable: Safe automation task execution
    - flow_through: Automation pipeline composition
    - lash: Fallback automation strategies
    - alt: Alternative automation paths
    - unwrap_or pattern: Lazy resource initialization
    """

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
    def _demo_alt() -> None:
        """Demo 4: alt - Alternative Automation Paths."""
        print("\n=== 4. alt: Alternative Automation Paths ===")

        def get_cached() -> r[t.ConfigMap]:
            return r[t.ConfigMap].fail("Cache unavailable")

        def get_default() -> r[t.ConfigMap]:
            return r[t.ConfigMap].ok(
                t.ConfigMap(
                    root={
                        "automation_mode": c.ProcessingMode.SEQUENTIAL.value,
                        "batch_size": c.DEFAULT_SIZE,
                    },
                ),
            )

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
    def _demo_etl_pipeline() -> None:
        """ETL Pipeline with Error Recovery."""
        print("\n--- Data Pipeline with Error Recovery ---")

        def extract() -> r[Sequence[t.ConfigMap]]:
            return r[Sequence[t.ConfigMap]].ok([
                t.ConfigMap(root={"id": 1, "name": "Item A", "value": 100}),
                t.ConfigMap(root={"id": 2, "name": "Item B", "value": 200}),
            ])

        def transform(
            data: Sequence[t.ConfigMap],
        ) -> FlextRuntime.RuntimeResult[Sequence[t.ConfigMap]]:
            transformed: Sequence[t.ConfigMap] = [
                t.ConfigMap(
                    root={
                        **item.root,
                        "processed": True,
                        "timestamp": "2025-01-01T12:00:00Z",
                    },
                )
                for item in data
            ]
            return r[Sequence[t.ConfigMap]].ok(transformed)

        def load(
            data: Sequence[t.ConfigMap],
        ) -> FlextRuntime.RuntimeResult[Sequence[t.ConfigMap]]:
            print(f"   💾 Loaded {len(data)} records successfully")
            return r[Sequence[t.ConfigMap]].ok(data)

        extract_result = extract()
        if extract_result.is_failure:
            print(f"❌ ETL Pipeline failed: {extract_result.error}")
            return
        transform_result = transform(extract_result.value)
        if transform_result.is_failure:
            print(f"❌ ETL Pipeline failed: {transform_result.error}")
            return
        result = load(transform_result.value)
        if result.is_success:
            print(f"✅ ETL Pipeline: {result.value}")
        else:
            print(f"❌ ETL Pipeline failed: {result.error}")

    @staticmethod
    def _demo_flow_through() -> None:
        """Demo 2: flow_through - Automation Pipeline Composition."""
        print("\n=== 2. flow_through: Automation Pipeline Composition ===")

        def validate(
            data: t.ConfigMap,
        ) -> FlextRuntime.RuntimeResult[t.ConfigMap]:
            task_type = str(data.get("task_type", ""))
            if not task_type:
                return r[t.ConfigMap].fail("Task type required")
            return r[t.ConfigMap].ok(data)

        def enrich(
            data: t.ConfigMap,
        ) -> FlextRuntime.RuntimeResult[t.ConfigMap]:
            enriched: t.ConfigMap = t.ConfigMap(
                root={
                    **data.root,
                    "automation_timestamp": "2025-01-01T12:00:00Z",
                    "duration_ms": 250,
                    "result_id": "RESULT-001",
                },
            )
            return r[t.ConfigMap].ok(enriched)

        automation_input: t.ConfigMap = t.ConfigMap(
            root={"task_type": c.ProcessingMode.BATCH.value, "source": "database"},
        )
        validate_result = validate(automation_input)
        if validate_result.is_failure:
            print(f"❌ Pipeline failed: {validate_result.error}")
            return
        pipeline_result = enrich(validate_result.value)
        if pipeline_result.is_success:
            data = pipeline_result.value
            task_type = data.get("task_type", "")
            duration = data.get("duration_ms", 0)
            print(f"✅ Pipeline complete: {task_type}")
            print(f"   Duration: {duration}ms")
        else:
            print(f"❌ Pipeline failed: {pipeline_result.error}")

    @staticmethod
    def _demo_from_callable() -> None:
        """Demo 1: from_callable - Safe Automation Task Execution."""
        print("\n=== 1. from_callable: Safe Automation Task Execution ===")

        def risky_automation_task() -> t.ConfigMap:
            task_data: t.ConfigMap = t.ConfigMap(
                root={
                    "task_id": "AUTO-001",
                    "task_type": "data_sync",
                    "records_processed": 1000,
                    "status": "success",
                },
            )
            records_text = str(u.get(task_data, "records_processed", default=0) or 0)
            records = int(records_text) if records_text.isdigit() else 0
            if records == 0:
                msg = "No records to process"
                raise ValueError(msg)
            return task_data

        result = r[t.ConfigMap].create_from_callable(risky_automation_task)
        if result.is_success:
            data = result.value
            print(f"✅ Automation successful: {data.get('task_type', 'N/A')}")
            print(f"   Records: {data.get('records_processed', 0)}")
        else:
            print(f"❌ Automation failed: {result.error}")

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
    def _demo_lazy_config() -> None:
        """Configuration with Lazy Loading."""
        print("\n--- Configuration with Lazy Loading ---")
        cache = t.ConfigMap(root={})

        def load_config() -> t.ConfigMap:
            if not cache.root:
                print("   📄 Loading configuration from file...")
                cache.root["database_url"] = "postgresql://localhost:5432/testdb"
                cache.root["cache_ttl"] = c.DEFAULT_CACHE_TTL
            return cache

        fail_attempt: r[t.ConfigMap] = r[t.ConfigMap].fail("No cached config")
        config: t.ConfigMap = (
            load_config() if fail_attempt.is_failure else fail_attempt.value
        )
        config_count = len(config.root)
        print(f"✅ Config loaded: {config_count} settings")
        cached_config = load_config()
        if not cached_config.root:
            msg = "Cached config should not be empty"
            raise RuntimeError(msg)
        print("✅ Second config access used cached version (no file loading)")

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
    def _demo_value_or_call() -> None:
        """Demo 5: value_or_call - Lazy Resource Initialization."""
        print("\n=== 5. value_or_call: Lazy Resource Initialization ===")

        def create_engine() -> t.ConfigMap:
            print("   ⚙️  Initializing automation engine...")
            return t.ConfigMap(
                root={
                    "engine_id": "AUTO-ENGINE-001",
                    "engine_type": c.ProcessingMode.PARALLEL.value,
                    "worker_count": c.DEFAULT_DB_POOL_SIZE,
                },
            )

        fail_result: r[t.ConfigMap] = r[t.ConfigMap].fail("No existing engine")
        engine: t.ConfigMap = (
            create_engine() if fail_result.is_failure else fail_result.value
        )
        engine_id = str(engine.get("engine_id", "unknown"))
        worker_count_text = str(engine.get("worker_count", 0))
        worker_count = int(worker_count_text) if worker_count_text.isdigit() else 0
        print(f"✅ Engine acquired: {engine_id}")
        print(f"   Workers: {worker_count}")
        existing: t.ConfigMap = t.ConfigMap(
            root={
                "engine_id": "CACHED-ENGINE-001",
                "worker_count": c.DEFAULT_WORKERS,
            },
        )
        success_result = r[t.ConfigMap].ok(existing)
        cached = success_result.map_or(create_engine())
        cached_id = str(cached.get("engine_id", "unknown"))
        print(f"✅ Existing engine used: {cached_id}")

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

    @override
    def execute(self) -> r[t.ConfigMap]:
        """Required abstract method implementation."""
        return r[t.ConfigMap].ok(t.ConfigMap(root={"status": "automation_ready"}))


def main() -> None:
    """Demonstrate context enrichment in action."""
    print("=" * 80)
    print("FLEXT-CORE CONTEXT ENRICHMENT SHOWCASE")
    print("=" * 80)
    print("\n1. BASIC SERVICE WITH CONTEXT ENRICHMENT")
    print("-" * 80)
    user_service = UserService()
    result1 = user_service.create_user("john_doe", "john@example.com")
    print(f"Result: {(result1.value if result1.is_success else result1.error)}")
    print("\n2. SERVICE WITH CORRELATION ID TRACKING")
    print("-" * 80)
    payment_service = PaymentService()
    result2 = payment_service.process_payment(
        payment_id="pay_123",
        amount=99.99,
        user_id="usr_456",
    )
    print(f"Result: {(result2.value if result2.is_success else result2.error)}")
    print("\n3. SERVICE USING CONTEXT ENRICHMENT HELPER")
    print("-" * 80)
    order_service = OrderService()
    result3 = order_service.process_order(
        order_id="ord_123",
        customer_id="cust_456",
        correlation_id="corr_abc123",
    )
    print(f"Result: {(result3.value if result3.is_success else result3.error)}")
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
    print("✅ NEW r methods: from_callable, flow_through, lash, alt, unwrap_or")
    print(
        "✅ Advanced automation scenarios: ETL pipelines, service orchestration, lazy loading",
    )
    print("✅ Zero boilerplate infrastructure code")
    print("=" * 80)


if __name__ == "__main__":
    main()
