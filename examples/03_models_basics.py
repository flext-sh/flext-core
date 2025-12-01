"""FlextModels advanced DDD patterns with Pydantic 2 features.

Complete domain-driven design with Value Objects, Entities, Aggregate Roots.
Uses advanced Python 3.13+ patterns, StrEnum validation, railway-oriented programming.

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
    FlextService,
    FlextTypes,
)

# Using centralized literals from FlextConstants (DRY - no local aliases)

# ========== DOMAIN MODELS ==========


class Email(FlextModels.Value):  # type: ignore[misc,valid-type]  # FlextModels.Value is assignment alias, valid for inheritance
    """Email value object with advanced Pydantic 2 EmailStr validation."""

    model_config = FlextConstants.Domain.DOMAIN_MODEL_CONFIG

    address: Annotated[
        EmailStr,
        Field(
            min_length=5,
            max_length=FlextConstants.Validation.MAX_EMAIL_LENGTH,
        ),
    ]


class Money(FlextModels.Value):  # type: ignore[misc,valid-type]  # FlextModels.Value is assignment alias, valid for inheritance
    """Money value object with StrEnum currency and railway operations."""

    model_config = FlextConstants.Domain.DOMAIN_MODEL_CONFIG

    amount: Annotated[Decimal, Field(gt=0)]
    currency: FlextConstants.Domain.Currency | str = Field(
        default=FlextConstants.Domain.Currency.USD
    )

    def add(self, other: Money) -> FlextResult[Money]:
        """Railway pattern for currency-aware addition."""
        if self.currency != other.currency:
            return FlextResult.fail("Currency mismatch")
        return FlextResult.ok(
            Money(amount=self.amount + other.amount, currency=self.currency)
        )


class User(FlextModels.Entity):  # type: ignore[misc,valid-type]  # FlextModels.Entity is assignment alias, valid for inheritance
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
            ge=FlextConstants.Validation.MIN_AGE, le=FlextConstants.Validation.MAX_AGE
        ),
    ]


class OrderItem(FlextModels.Value):  # type: ignore[misc,valid-type]  # FlextModels.Value is assignment alias, valid for inheritance
    """Order item with computed fields and railway validation."""

    model_config = FlextConstants.Domain.DOMAIN_MODEL_CONFIG

    product_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    price: Money
    quantity: Annotated[int, Field(gt=0, le=1000)]

    @computed_field  # type: ignore[prop-decorator]  # Pydantic 2 requires @property with @computed_field
    @property
    def total(self) -> Money:
        """Railway-aware total calculation."""
        return Money(
            amount=self.price.amount * self.quantity,
            currency=self.price.currency,
        )


class Order(FlextModels.AggregateRoot):  # type: ignore[misc,valid-type]  # FlextModels.AggregateRoot is assignment alias, valid for inheritance
    """Order aggregate root with advanced business rules."""

    model_config = FlextConstants.Domain.DOMAIN_MODEL_CONFIG

    customer_id: str = Field(min_length=1)
    items: list[OrderItem] = Field(default_factory=list)
    status: FlextConstants.Domain.OrderStatus = Field(
        default=FlextConstants.Domain.OrderStatus.PENDING
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

    @computed_field  # type: ignore[prop-decorator]  # Pydantic 2 requires @property with @computed_field
    @property
    def total(self) -> Money:
        """Railway-aware order total calculation."""
        if not self.items:
            return Money(amount=Decimal(0), currency=FlextConstants.Domain.Currency.USD)
        currency = self.items[0].price.currency
        total_amount = Decimal(sum(item.total.amount for item in self.items))
        return Money(amount=total_amount, currency=currency)


# Rebuild models to resolve forward references after all definitions
# Use FlextModels.Entity directly (no private imports)
# FlextModels.Entity is already the correct type (FlextModelsEntity.Core)
_types_namespace = {**globals(), "FlextModels": FlextModels}
User.model_rebuild(_types_namespace=_types_namespace)
Order.model_rebuild(_types_namespace=_types_namespace)
OrderItem.model_rebuild(_types_namespace=_types_namespace)


class DomainModelService(FlextService[FlextTypes.Types.ServiceMetadataMapping]):
    """Advanced DDD demonstration service with railway-oriented programming."""

    def execute(self) -> FlextResult[FlextTypes.Types.ServiceMetadataMapping]:  # noqa: PLR6301  # Required by FlextService abstract method
        """Execute comprehensive DDD demonstrations using railway patterns."""
        # Railway pattern with value objects using traverse (DRY)
        email_result = FlextResult.ok(Email(address="Test@Example.Com"))
        money_result = FlextResult.ok(
            Money(amount=Decimal("10.00"), currency=FlextConstants.Domain.Currency.USD)
        ).flat_map(lambda m: m.add(Money(amount=Decimal("5.00"), currency=m.currency)))

        # Combine results using railway pattern (DRY - no manual error collection)
        value_objects_result = email_result.flat_map(
            lambda email: money_result.map(lambda money: (email, money))
        )

        # Entity and aggregate with railway pattern
        user_result = FlextResult.ok(
            User(
                name="Alice",
                email=Email(address="alice@example.com"),
                age=30,
            )
        )

        order_result = (
            FlextResult.ok(Order(customer_id="cust-123"))
            .flat_map(
                lambda o: o.add_item(
                    OrderItem(
                        product_id="prod-001",
                        name="Widget",
                        price=Money(amount=Decimal("29.99")),
                        quantity=2,
                    )
                )
            )
            .flat_map(Order.confirm)
        )

        # Combine all results using railway pattern (DRY)
        return value_objects_result.flat_map(
            lambda vo_tuple: user_result.flat_map(
                lambda user: order_result.map(
                    lambda order: {
                        "email": vo_tuple[0].address,
                        "money_sum": f"{vo_tuple[1].amount} {vo_tuple[1].currency}",
                        "user_id": user.entity_id,
                        "order_total": float(order.total.amount),
                        "order_status": order.status,
                    }
                )
            )
        )


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
                f"✅ Order total: {data['order_total']}, status: {data['order_status']}"
            )
        case FlextResult(is_success=False, error=error):
            print(f"❌ Failed: {error}")

    print("🎯 Advanced DDD: Value Objects, Entities, Aggregate Roots")
    print("🎯 Railway Pattern: Comprehensive error handling throughout")
    print("🎯 Pydantic 2: StrEnum, computed_field, AfterValidator, Field validation")
    print("🎯 Python 3.13+: PEP 695 types, collections.abc, advanced patterns")


if __name__ == "__main__":
    main()
