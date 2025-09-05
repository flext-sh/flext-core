#!/usr/bin/env python3
"""FLEXT Core - Complete Integration Example.

Comprehensive example showing all FLEXT components working together with shared
domain models.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations
from decimal import Decimal
from typing import cast

from flext_core import (
    FlextContainer,
    FlextFields,
    FlextLogger,
    FlextModels,
    FlextResult,
    FlextTypes,
)

# =============================================================================
# VALIDATION CONSTANTS - Integration example constraints using FlextTypes
# =============================================================================

# Email validation constants using proper FlextTypes annotations
MIN_EMAIL_LENGTH: int = 5  # Minimum characters for basic email validation


# Simple classes for the example
class Money:
    """Simple money value object for demonstration."""

    def __init__(self, amount: Decimal, currency: str) -> None:
        self.amount = amount
        self.currency = currency

    def __str__(self) -> str:
        """String representation of money."""
        return f"{self.amount} {self.currency}"


class Order:
    """Simple order class for demonstration."""

    def __init__(
        self,
        order_id: str,
        customer_id: str,
        items: list[dict[str, object]],
    ) -> None:
        self.id = order_id
        self.customer_id = customer_id
        self.items = items

    def calculate_total(self) -> FlextResult[Money]:
        """Calculate order total."""
        # Simple calculation for demo
        total = Decimal(0)
        for item in self.items:
            price = item.get("price", 0)
            if isinstance(price, (int, float, str)):
                total += Decimal(str(price))
        return FlextResult[Money].ok(Money(total, "USD"))

    def __str__(self) -> str:
        """String representation of order."""
        return (
            f"Order({self.id}, customer: {self.customer_id}, items: {len(self.items)})"
        )


class User:
    """Simple user class for demonstration."""

    def __init__(self, user_id: str, name: str, email: str) -> None:
        self.id = user_id
        self.name = name
        self.email = email

    def __str__(self) -> str:
        """String representation of user."""
        return f"User({self.id}, {self.name}, {self.email})"


def _print_header() -> None:
    pass


def _setup_logger() -> FlextLogger:
    logger = FlextLogger("integration_example")
    logger.info("Starting complete integration example")
    return logger


def _shared_domain_vo_demo() -> tuple[FlextModels.EmailAddress, Money]:
    email = FlextModels.EmailAddress(root="customer@example.com")
    money = Money(amount=Decimal("100.0"), currency="USD")
    return email, money


def _create_customer(
    email: FlextModels.EmailAddress,
) -> FlextResult[User]:
    user = User(user_id="customer_123", name="John Customer", email=str(email.root))
    return FlextResult[User].ok(user)


def _print_customer(customer: User) -> None:
    pass


def _create_order(customer_id: str, money: Money) -> FlextResult[Order]:
    order_items = [
        {
            "product_id": "product123",
            "product_name": "Test Product",
            "quantity": "1",
            "unit_price": str(money.amount),
            "currency": money.currency,
        },
    ]
    # Add price for calculate_total method
    order_items_with_price: list[dict[str, object]] = []
    for item in order_items:
        item_copy: dict[str, object] = {
            "product_id": item["product_id"],
            "product_name": item["product_name"],
            "quantity": item["quantity"],
            "unit_price": item["unit_price"],
            "currency": item["currency"],
            "price": float(item["unit_price"]),  # Convert to float for calculation
        }
        order_items_with_price.append(item_copy)

    order = Order(
        order_id="order_456",
        customer_id=customer_id,
        items=order_items_with_price,
    )
    return FlextResult[Order].ok(order)


def _print_order(order: Order) -> None:
    pass


def _demo_command_pattern(
    customer_id: str,
    order_items: list[dict[str, str]],
) -> None:
    class CreateOrderCommand:
        """Command pattern for order creation with FlextTypes annotations."""

        def __init__(
            self,
            customer_id: str,
            items: list[dict[str, str]],
        ) -> None:
            self.customer_id = customer_id
            self.items = items

        def validate_command(self) -> FlextResult[None]:
            """Validate command parameters."""
            if not self.customer_id.strip():
                return FlextResult[None].fail("Customer ID required")
            if not self.items:
                return FlextResult[None].fail("Order items required")
            return FlextResult[None].ok(None)

    create_command = CreateOrderCommand(customer_id=customer_id, items=order_items)
    create_command.validate_command()


def _demo_repository_pattern(order: Order) -> object:
    """Demonstrate repository pattern with FlextResult."""

    class OrderRepository:
        """Repository pattern implementation with FlextTypes."""

        def __init__(self) -> None:
            self.orders: dict[str, Order] = {}

        def save(self, order: Order) -> FlextResult[Order]:
            """Save order with FlextResult error handling."""
            order_id = order.id
            self.orders[order_id] = order
            return FlextResult[Order].ok(order)

        def get_by_id(self, order_id: str) -> FlextResult[Order]:
            """Retrieve order by ID with FlextResult."""
            if order_id in self.orders:
                return FlextResult[Order].ok(self.orders[order_id])
            return FlextResult[Order].fail(f"Order {order_id} not found")

    repository = OrderRepository()
    repository.save(order)
    fetch_result = repository.get_by_id(order.id)
    if fetch_result.success:
        pass
    return repository


class _OrderRepositoryProtocol:
    def save(self, order: Order) -> object: ...  # pragma: no cover


def _demo_container(customer: User, repository: object) -> None:
    container = FlextContainer.get_global()
    container.register("order_repository", repository)
    container.register("current_customer", customer)
    repo_result = container.get("order_repository")
    customer_result = container.get("current_customer")
    if repo_result.success and customer_result.success:
        retrieved_customer = customer_result.value
        if hasattr(retrieved_customer, "name"):
            pass


def _demo_fields_validation() -> None:
    """Demo fields validation using correct FlextFields API."""
    # Use the actual FlextFields.Factory.create_field API with correct signature
    email_field_result = FlextFields.Factory.create_field(
        field_type="string",
        name="email",
        pattern=r"^[^@]+@[^@]+\.[^@]+$",
        required=True,
        field_id="customer_email",
    )

    # Test validation using the created field with FlextResult pattern
    if email_field_result.success:
        # Field created successfully, ready for use
        pass


def _demo_complete_flow(customer: User, order: Order, logger: FlextLogger) -> None:
    # Create order using simple constructor instead of non-existent factory
    order2_result = FlextResult[Order].ok(
        Order(
            order_id="order_456",
            customer_id=str(customer.id),
            items=[
                {
                    "product_id": "product456",
                    "product_name": "Another Product",
                    "quantity": "2",
                    "unit_price": "50.0",
                    "price": 100.0,  # Added price for calculate_total
                    "currency": "USD",
                },
            ],
        ),
    )
    # Modern pattern: Check success and use value directly
    if order2_result.success:
        order2 = order2_result.value
        repository_any = FlextContainer.get_global().get("order_repository").value
        # Hint to type checker
        repo: _OrderRepositoryProtocol = cast(
            "_OrderRepositoryProtocol",
            repository_any,
        )
        repo.save(order2)
        total1_result = order.calculate_total()
        total2_result = order2.calculate_total()
        if total1_result.success and total2_result.success:
            total1 = total1_result.value
            total2 = total2_result.value
            if total1 is not None and total2 is not None:
                _ = total1.amount + total2.amount
    logger.info("Integration example completed successfully")


def main() -> None:
    """Execute main function for integration example."""
    _print_header()
    logger = _setup_logger()
    email, money = _shared_domain_vo_demo()
    user_result = _create_customer(email)
    if user_result.is_failure or user_result.value is None:
        return
    customer = user_result.value
    _print_customer(customer)
    order_result = _create_order(str(customer.id), money)
    if order_result.is_failure or order_result.value is None:
        return
    order = order_result.value
    _print_order(order)
    _demo_command_pattern(
        str(customer.id),
        [
            {
                "product_id": "product123",
                "product_name": "Test Product",
                "quantity": "1",
                "unit_price": str(money.amount),
                "currency": money.currency,
            },
        ],
    )
    repository = _demo_repository_pattern(order)
    _demo_container(customer, repository)
    _demo_fields_validation()
    _demo_complete_flow(customer, order, logger)


if __name__ == "__main__":
    main()
