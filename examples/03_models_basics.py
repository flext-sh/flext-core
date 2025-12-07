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
from typing import Annotated

from pydantic import EmailStr, Field, computed_field

from flext_core import (
    FlextConstants,
    FlextModels,
    FlextResult,
    s,
    t,
)

# Using centralized literals from FlextConstants (DRY - no local aliases)

# ========== DOMAIN MODELS ==========


class Email(FlextModels.Value):
    """Email value object with advanced Pydantic 2 EmailStr validation."""

    model_config = FlextConstants.Domain.DOMAIN_MODEL_CONFIG

    address: Annotated[
        EmailStr,
        Field(
            min_length=5,
            max_length=FlextConstants.Validation.MAX_EMAIL_LENGTH,
        ),
    ]


class Money(FlextModels.Value):
    """Money value object with StrEnum currency and railway operations."""

    model_config = FlextConstants.Domain.DOMAIN_MODEL_CONFIG

    amount: Annotated[Decimal, Field(gt=0)]
    currency: FlextConstants.Domain.Currency | str = Field(
        default=FlextConstants.Domain.Currency.USD,
    )

    def add(self, other: Money) -> FlextResult[Money]:
        """Railway pattern for currency-aware addition."""
        if self.currency != other.currency:
            return FlextResult.fail("Currency mismatch")
        return FlextResult.ok(
            Money(amount=self.amount + other.amount, currency=self.currency),
        )


class User(FlextModels.Entity):
    """User entity with comprehensive validation and domain rules."""

    model_config = FlextConstants.Domain.DOMAIN_MODEL_CONFIG

    name: str = Field(
        min_length=FlextConstants.Validation.MIN_NAME_LENGTH,
        max_length=FlextConstants.Validation.MAX_NAME_LENGTH,
    )
    email: Email
    age: Annotated[
        int,
        Field(
            ge=FlextConstants.Validation.MIN_AGE,
            le=FlextConstants.Validation.MAX_AGE,
        ),
    ]


class OrderItem(FlextModels.Value):
    """Order item with computed fields and railway validation."""

    model_config = FlextConstants.Domain.DOMAIN_MODEL_CONFIG

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


class Order(FlextModels.AggregateRoot):
    """Order aggregate root with advanced business rules."""

    model_config = FlextConstants.Domain.DOMAIN_MODEL_CONFIG

    customer_id: str = Field(min_length=1)
    items: list[OrderItem] = Field(default_factory=list)
    status: FlextConstants.Domain.OrderStatus = Field(
        default=FlextConstants.Domain.OrderStatus.PENDING,
    )

    def add_item(self, item: OrderItem) -> FlextResult[Order]:
        """Railway pattern for item addition with domain rules."""
        if self.status != FlextConstants.Domain.OrderStatus.PENDING:
            return FlextResult.fail("Cannot modify non-pending order")
        if any(existing.product_id == item.product_id for existing in self.items):
            return FlextResult.fail("Product already in order")
        self.items.append(item)
        return FlextResult.ok(self)

    def confirm(self) -> FlextResult[Order]:
        """Railway pattern for order confirmation."""
        if not self.items:
            return FlextResult.fail("Cannot confirm empty order")
        if self.status != FlextConstants.Domain.OrderStatus.PENDING:
            return FlextResult.fail("Order already processed")
        self.status = FlextConstants.Domain.OrderStatus.CONFIRMED
        return FlextResult.ok(self)

    @property
    @computed_field
    def total(self) -> Money:
        """Railway-aware order total calculation."""
        if not self.items:
            return Money(amount=Decimal(0), currency=FlextConstants.Domain.Currency.USD)
        currency = self.items[0].price.currency
        total_amount = Decimal(sum(item.total.amount for item in self.items))
        return Money(amount=total_amount, currency=currency)


# No model_rebuild() needed - Pydantic v2 with 'from __future__ import annotations'
# automatically resolves forward references at runtime


class DomainModelService(s[t.Types.ServiceMetadataMapping]):
    """Advanced DDD demonstration service with railway-oriented programming."""

    def execute(self) -> FlextResult[t.Types.ServiceMetadataMapping]:
        """Execute comprehensive DDD demonstrations using railway patterns."""
        # Railway pattern with value objects using traverse (DRY)
        email_result = FlextResult[Email].ok(Email(address="Test@Example.Com"))

        def add_money(m: Money) -> FlextResult[Money]:
            return m.add(Money(amount=Decimal("5.00"), currency=m.currency))

        money_result = (
            FlextResult[Money]
            .ok(
                Money(
                    amount=Decimal("10.00"),
                    currency=FlextConstants.Domain.Currency.USD,
                ),
            )
            .flat_map(add_money)
        )

        # Combine results using railway pattern (DRY - no manual error collection)
        def combine_email_money(email: Email) -> FlextResult[tuple[Email, Money]]:
            def make_tuple(money: Money) -> tuple[Email, Money]:
                return (email, money)

            return money_result.map(make_tuple)

        value_objects_result: FlextResult[tuple[Email, Money]] = email_result.flat_map(
            combine_email_money,
        )

        # Entity and aggregate with railway pattern
        user_result = FlextResult[User].ok(
            User(
                name="Alice",
                email=Email(address="alice@example.com"),
                age=30,
            ),
        )

        def add_order_item(o: Order) -> FlextResult[Order]:
            return o.add_item(
                OrderItem(
                    product_id="prod-001",
                    name="Widget",
                    price=Money(amount=Decimal("29.99")),
                    quantity=2,
                ),
            )

        order_result = (
            FlextResult[Order]
            .ok(Order(customer_id="cust-123"))
            .flat_map(add_order_item)
            .flat_map(Order.confirm)
        )

        # Combine all results using railway pattern (DRY)
        def build_result(
            vo_tuple: tuple[Email, Money],
            user: User,
            order: Order,
        ) -> t.Types.ServiceMetadataMapping:
            return {
                "email": vo_tuple[0].address,
                "money_sum": f"{vo_tuple[1].amount} {vo_tuple[1].currency}",
                "user_id": user.entity_id,
                "order_total": float(order.total.amount),
                "order_status": order.status,
            }

        def combine_with_user(
            vo_tuple: tuple[Email, Money],
        ) -> FlextResult[t.Types.ServiceMetadataMapping]:
            def combine_with_order(
                user: User,
            ) -> FlextResult[t.Types.ServiceMetadataMapping]:
                def finalize(order: Order) -> t.Types.ServiceMetadataMapping:
                    return build_result(vo_tuple, user, order)

                return order_result.map(finalize)

            return user_result.flat_map(combine_with_order)

        return value_objects_result.flat_map(combine_with_user)


def main() -> None:
    """Advanced main entry point with pattern matching."""
    print("FLEXT MODELS - ADVANCED DDD PATTERNS WITH PYDANTIC 2")

    service = DomainModelService()
    match service.execute():
        case FlextResult(is_success=True, value=data):
            print(f"✅ Email: {data['email']}")
            print(f"✅ Money sum: {data['money_sum']}")
            print(f"✅ User ID: {data['user_id']}")
            print(
                f"✅ Order total: {data['order_total']}, status: {data['order_status']}",
            )
        case FlextResult(is_success=False, error=error):
            print(f"❌ Failed: {error}")
        case _:
            pass

    print("🎯 Advanced DDD: Value Objects, Entities, Aggregate Roots")
    print("🎯 Railway Pattern: Comprehensive error handling throughout")
    print("🎯 Pydantic 2: StrEnum, computed_field, AfterValidator, Field validation")
    print("🎯 Python 3.13+: PEP 695 types, collections.abc, advanced patterns")


if __name__ == "__main__":
    main()
