"""FlextModels advanced DDD patterns with Pydantic 2 features.

Complete domain-driven design with Value Objects, Entities, Aggregate Roots.
Uses advanced Python 3.13+ patterns, StrEnum validation, railway-oriented programming.

**Expected Output:**
- Value Object creation and immutability demonstrations
- Entity identity and business logic patterns
- Aggregate Root consistency enforcement
- Domain event patterns
- Validation with Pydantic Field constraints
- Railway pattern integration with domain models

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, override

from flext_core import (
    c,
    m,
    r,
    s,
    t,
)
from flext_core._models.base import FlextModelFoundation as F
from flext_core._models.generic import FlextGenericModels as gm
from pydantic import EmailStr, Field, computed_field, model_validator

# Using centralized literals from c (DRY - no local aliases)

# ========== ENHANCED GENERIC MODELS DEMONSTRATION ==========


def demonstrate_enhanced_generic_models() -> None:
    """Demonstrate enhanced generic models with advanced features."""
    print("🎯 ENHANCED GENERIC MODELS - Advanced Features")

    # Enhanced OperationContext with tracking and metadata

    context = gm.Value.OperationContext(
        source="demo",
        user_id="user123",
        tenant_id="tenant456",
        environment="development",
        metadata=t.Dict(root={"session_id": "sess789", "request_id": "req101"}),
    )

    print(f"📊 Context: source={context.source}, env={context.environment}")
    print(
        f"🆔 Correlation: {context.correlation_id[:8]}..., operation={context.operation_id[:8]}..."
    )
    print(f"👤 User: {context.user_id}, Tenant: {context.tenant_id}")
    print(f"🧾 Metadata: {context.metadata.root}")

    # Enhanced Service snapshot with health monitoring
    service = gm.Snapshot.Service(
        name="user-service",
        version="2.1.0",
        status="active",
        uptime_seconds=3600,
        memory_usage_mb=150.5,
        cpu_usage_percent=25.3,
        health_status="healthy",
    )

    print(f"🔧 Service: {service.name} v{service.version}")
    print(f"⚡ Status: {service.status}, Health: {service.health_status}")
    print(f"⏰ Uptime (s): {service.uptime_seconds}")
    print(
        f"📈 Resources: memory={service.memory_usage_mb}MB cpu={service.cpu_usage_percent}%",
    )

    # Enhanced Health check with detailed monitoring
    health = gm.Snapshot.Health(
        healthy=True,
        checks=t.Dict(
            root={
                "database": True,
                "cache": True,
                "external_api": False,
                "filesystem": True,
            },
        ),
        service_name="user-service",
        service_version="2.1.0",
        duration_ms=125.5,
    )

    check_values = [
        value for value in health.checks.root.values() if isinstance(value, bool)
    ]
    total_checks = len(check_values)
    healthy_checks = sum(1 for value in check_values if value)
    health_percentage = (healthy_checks / total_checks * 100.0) if total_checks else 0.0
    failed_checks = total_checks - healthy_checks
    print(f"🏥 Health: {health.healthy} ({health_percentage:.1f}%)")
    print(f"📋 Status: service={health.service_name} version={health.service_version}")
    print(f"🚨 Severity: {'high' if failed_checks > 0 else 'normal'}")
    print(f"❌ Failed Checks: {failed_checks}")

    # Enhanced Operation progress tracking
    operation = gm.Progress.Operation(
        operation_name="user_import",
        estimated_total=1000,
    )
    operation.start_operation()

    # Simulate progress
    for _ in range(750):
        operation.record_success()
    for _ in range(50):
        operation.record_failure()
    for _ in range(25):
        operation.record_warning()

    operation_total = (
        operation.success_count + operation.failure_count + operation.skipped_count
    )
    operation_completion = gm.Progress.safe_percentage(
        operation_total,
        operation.estimated_total,
    )
    operation_success_rate = gm.Progress.safe_rate(
        operation.success_count,
        operation_total,
    )
    print(f"📈 Operation: {operation.operation_name}")
    print(f"✅ Progress: {operation_completion:.1f}%")
    print(f"🎯 Success Rate: {operation_success_rate:.1%}")
    print(f"⚠️ Has Warnings: {operation.warning_count > 0}")
    print(
        f"📊 Status: successes={operation.success_count}, failures={operation.failure_count}"
    )

    # Enhanced Conversion tracking
    conversion = gm.Progress.Conversion(
        source_format="csv",
        target_format="json",
        total_input_count=500,
    )
    conversion.start_conversion()

    # Simulate conversion
    for i in range(480):
        conversion.add_converted(f"record_{i}")
    for i in range(15):
        conversion.add_error(f"Parse error in record_{i}")
    for i in range(5):
        conversion.add_skipped(f"record_{i}", "Duplicate entry")

    conversion.complete_conversion()

    conversion_total = (
        len(conversion.converted) + len(conversion.errors) + len(conversion.skipped)
    )
    conversion_completion = gm.Progress.safe_percentage(
        conversion_total,
        conversion.total_input_count,
    )
    conversion_success_rate = gm.Progress.safe_rate(
        len(conversion.converted),
        conversion_total,
    )
    duration_seconds = (
        (conversion.end_time - conversion.start_time).total_seconds()
        if conversion.start_time is not None and conversion.end_time is not None
        else 0.0
    )
    items_per_second = (
        (conversion_total / duration_seconds) if duration_seconds > 0 else 0.0
    )
    print(f"🔄 Conversion: {conversion.source_format} → {conversion.target_format}")
    print(f"📊 Progress: {conversion_completion:.1f}%")
    print(f"✅ Success Rate: {conversion_success_rate:.1%}")
    print(f"⚡ Processing Rate: {items_per_second:.1f} items/sec")
    print(f"⏱️ Duration: {duration_seconds:.2f}s")
    print(
        f"📋 Status: converted={len(conversion.converted)} errors={len(conversion.errors)} skipped={len(conversion.skipped)}",
    )

    print(
        "✨ Enhanced generic models provide rich monitoring and tracking capabilities!\n",
    )


def demonstrate_advanced_pydantic_mixins() -> None:
    """Demonstrate advanced Pydantic v2 mixins with validation and serialization."""
    print("🔬 ADVANCED PYDANTIC v2 MIXINS - Full Validation & Features")

    # ═══════════════════════════════════════════════════════════════════════════
    # COMPREHENSIVE ENTITY - Using all mixins together
    # ═══════════════════════════════════════════════════════════════════════════

    class AdvancedEntity(
        F.IdentifiableMixin,
        F.TimestampableMixin,
        F.VersionableMixin,
    ):
        """Entity demonstrating all mixins with Pydantic v2 features."""

        name: str
        description: str | None = None
        tags: list[str] = Field(default_factory=list)
        categories: list[str] = Field(default_factory=list)
        labels: dict[str, str] = Field(default_factory=dict)
        is_deleted: bool = False

        @model_validator(mode="after")
        def validate_entity_business_rules(self) -> AdvancedEntity:
            """Custom business rule validation."""
            if self.is_deleted and self.name.startswith("ACTIVE_"):
                msg = "Active entities cannot be deleted"
                raise ValueError(msg)
            return self

    # Create entity with full validation
    entity = AdvancedEntity(
        name="TestEntity",
        description="Advanced entity demo",
        tags=["demo", "advanced"],
        categories=["test"],
        labels={"env": "dev", "priority": "high"},
    )

    print(f"🏷️ Entity: {entity.name}")
    print(f"🆔 ID: {entity.unique_id}")
    print(f"⏰ Created: {entity.created_at.isoformat()}")
    print(f"📊 Version: {entity.version}")
    print(f"🏷️ Tags: {entity.tags} ({len(entity.tags)} total)")
    print(f"📂 Categories: {entity.categories}")
    print(f"🏷️ Labels: {entity.labels}")

    entity.increment_version()
    entity.update_timestamp()
    print(f"✅ Updated version: {entity.version}")
    print(f"🕒 Updated at: {entity.updated_at}")

    # Test serialization
    json_data = entity.model_dump_json(indent=2)
    print(f"📄 JSON length: {len(json_data)} chars")

    # Test deserialization
    restored = AdvancedEntity.model_validate_json(json_data)
    print(f"🔄 Round-trip successful: {restored.name == entity.name}")

    # Test business rule validation
    try:
        _invalid_entity = AdvancedEntity(
            name="ACTIVE_Invalid",
            is_deleted=True,
        )
        print("❌ Should have failed validation")
    except ValueError as e:
        print(f"✅ Business rule validation: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # VALIDATION DEMONSTRATION
    # ═══════════════════════════════════════════════════════════════════════════

    print("\n🔍 VALIDATION FEATURES:")

    # Test field validators
    try:
        F.IdentifiableMixin(unique_id="")
    except ValueError as e:
        print(f"✅ Field validation: {e}")

    # Test cross-field validation
    try:
        invalid_timestamp = F.TimestampableMixin()
        invalid_timestamp.updated_at = invalid_timestamp.created_at.replace(
            second=0,
        )  # Before creation
    except ValueError as e:
        print(f"✅ Cross-field validation: {e}")

    # Test audit consistency
    try:
        F.VersionableMixin(version=0)
    except ValueError as e:
        print(f"✅ Version validation: {e}")

    print(
        "🎯 Pydantic v2 mixins provide enterprise-grade validation and functionality!\n",
    )


# Add to main execution
if __name__ == "__main__":
    demonstrate_advanced_pydantic_mixins()
    # main() is called at the end of the file


# ========== DOMAIN MODELS ==========


class Email(m.Value):
    """Email value object with advanced Pydantic 2 EmailStr validation."""

    model_config = m.Config.DOMAIN_MODEL_CONFIG

    address: Annotated[
        EmailStr,
        Field(
            min_length=5,
            max_length=c.Validation.MAX_EMAIL_LENGTH,
        ),
    ]


class Money(m.Value):
    """Money value object with StrEnum currency and railway operations."""

    model_config = m.Config.DOMAIN_MODEL_CONFIG

    amount: Annotated[Decimal, Field(gt=Decimal(0))]
    currency: c.Domain.Currency | str = Field(
        default=c.Domain.Currency.USD,
    )

    def add(self, other: Money) -> r[Money]:
        """Railway pattern for currency-aware addition."""
        if self.currency != other.currency:
            return r.fail("Currency mismatch")
        return r.ok(
            Money(amount=self.amount + other.amount, currency=self.currency),
        )


class User(m.Entity):
    """User entity with comprehensive validation and domain rules."""

    model_config = m.Config.DOMAIN_MODEL_CONFIG

    name: str = Field(
        min_length=c.Validation.MIN_NAME_LENGTH,
        max_length=c.Validation.MAX_NAME_LENGTH,
    )
    email: Email
    age: Annotated[
        int,
        Field(
            ge=c.Validation.MIN_AGE,
            le=c.Validation.MAX_AGE,
        ),
    ]


class OrderItem(m.Value):
    """Order item with computed fields and railway validation."""

    model_config = m.Config.DOMAIN_MODEL_CONFIG

    product_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    price: Money
    quantity: Annotated[int, Field(gt=0, le=1000)]

    @property
    @computed_field
    def total(self) -> Money:
        """Railway-aware total calculation."""
        return Money(
            amount=self.price.amount * self.quantity,
            currency=self.price.currency,
        )


class Order(m.AggregateRoot):
    """Order aggregate root with advanced business rules."""

    model_config = m.Config.DOMAIN_MODEL_CONFIG

    customer_id: str = Field(min_length=1)
    items: list[OrderItem] = Field(default_factory=list)
    status: c.Domain.OrderStatus = Field(
        default=c.Domain.OrderStatus.PENDING,
    )

    def add_item(self, item: OrderItem) -> r[Order]:
        """Railway pattern for item addition with domain rules."""
        if self.status != c.Domain.OrderStatus.PENDING:
            return r.fail("Cannot modify non-pending order")
        if any(existing.product_id == item.product_id for existing in self.items):
            return r.fail("Product already in order")
        self.items.append(item)
        return r.ok(self)

    def confirm(self) -> r[Order]:
        """Railway pattern for order confirmation."""
        if not self.items:
            return r.fail("Cannot confirm empty order")
        if self.status != c.Domain.OrderStatus.PENDING:
            return r.fail("Order already processed")
        self.status = c.Domain.OrderStatus.CONFIRMED
        return r.ok(self)

    @property
    @computed_field
    def total(self) -> Money:
        """Railway-aware order total calculation."""
        if not self.items:
            return Money(amount=Decimal(0), currency=c.Domain.Currency.USD)
        currency = self.items[0].price.currency
        total_amount = Decimal(
            sum(item.price.amount * item.quantity for item in self.items),
        )
        return Money(amount=total_amount, currency=currency)


# No model_rebuild() needed - Pydantic v2 with 'from __future__ import annotations'
# automatically resolves forward references at runtime


class DomainModelService(s[m.ConfigMap]):
    """Advanced DDD demonstration service with railway-oriented programming."""

    @override
    def execute(self) -> r[m.ConfigMap]:
        """Execute comprehensive DDD demonstrations using railway patterns."""
        # Railway pattern with value objects using traverse (DRY)
        email_result = r[Email].ok(Email(address="Test@Example.Com"))

        def add_money(m: Money) -> r[Money]:
            return m.add(Money(amount=Decimal("5.00"), currency=m.currency))

        money_result = (
            r[Money]
            .ok(
                Money(
                    amount=Decimal("10.00"),
                    currency=c.Domain.Currency.USD,
                ),
            )
            .flat_map(add_money)
        )

        # Combine results using railway pattern (DRY - no manual error collection)
        def combine_email_money(email: Email) -> r[tuple[Email, Money]]:
            def make_tuple(money: Money) -> tuple[Email, Money]:
                return (email, money)

            return money_result.map(make_tuple)

        value_objects_result: r[tuple[Email, Money]] = email_result.flat_map(
            combine_email_money,
        )

        # Entity and aggregate with railway pattern
        user_result = r[User].ok(
            User(
                name="Alice",
                email=Email(address="alice@example.com"),
                age=30,
            ),
        )

        def add_order_item(o: Order) -> r[Order]:
            return o.add_item(
                OrderItem(
                    product_id="prod-001",
                    name="Widget",
                    price=Money(amount=Decimal("29.99")),
                    quantity=2,
                ),
            )

        order_result = (
            r[Order]
            .ok(Order(customer_id="cust-123"))
            .flat_map(add_order_item)
            .flat_map(Order.confirm)
        )

        # Combine all results using railway pattern (DRY)
        def build_result(
            vo_tuple: tuple[Email, Money],
            user: User,
            order: Order,
        ) -> m.ConfigMap:
            order_total = sum(item.price.amount * item.quantity for item in order.items)
            return m.ConfigMap(
                root={
                    "email": vo_tuple[0].address,
                    "money_sum": f"{vo_tuple[1].amount} {vo_tuple[1].currency}",
                    "user_id": user.entity_id,
                    "order_total": float(order_total),
                    "order_status": order.status,
                },
            )

        def combine_with_user(
            vo_tuple: tuple[Email, Money],
        ) -> r[m.ConfigMap]:
            def combine_with_order(
                user: User,
            ) -> r[m.ConfigMap]:
                def finalize(order: Order) -> m.ConfigMap:
                    return build_result(vo_tuple, user, order)

                return order_result.map(finalize)

            return user_result.flat_map(combine_with_order)

        return value_objects_result.flat_map(combine_with_user)


def main() -> None:
    """Advanced main entry point with pattern matching."""
    print("FLEXT MODELS - ADVANCED DDD PATTERNS WITH PYDANTIC 2")

    # Demonstrate enhanced generic models
    demonstrate_enhanced_generic_models()

    service = DomainModelService()
    match service.execute():
        case r(is_success=True, value=data):
            print(f"✅ Email: {data['email']}")
            print(f"✅ Money sum: {data['money_sum']}")
            print(f"✅ User ID: {data['user_id']}")
            print(
                f"✅ Order total: {data['order_total']}, status: {data['order_status']}",
            )
        case r(is_success=False, error=error):
            print(f"❌ Failed: {error}")
        case _:
            pass

    print("🎯 Advanced DDD: Value Objects, Entities, Aggregate Roots")
    print("🎯 Railway Pattern: Comprehensive error handling throughout")
    print("🎯 Pydantic 2: StrEnum, computed_field, AfterValidator, Field validation")
    print("🎯 Python 3.13+: PEP 695 types, collections.abc, advanced patterns")


if __name__ == "__main__":
    main()
