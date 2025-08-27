#!/usr/bin/env python3
"""FLEXT Core - Complete Integration Example.

Comprehensive example showing all FLEXT components working together with shared
domain models.
"""

from decimal import Decimal
from typing import cast

from shared_domain import EmailAddress, Money, Order, SharedDomainFactory, User

from flext_core import (
    FlextContainer,
    FlextFields,
    FlextLogger,
    FlextResult,
    FlextTypes,
)

# =============================================================================
# VALIDATION CONSTANTS - Integration example constraints using FlextTypes
# =============================================================================

# Email validation constants using proper FlextTypes annotations
MIN_EMAIL_LENGTH: FlextTypes.Core.Integer = (
    5  # Minimum characters for basic email validation
)


def _print_header() -> None:
    pass


def _setup_logger() -> FlextLogger:
    logger = FlextLogger("integration_example")
    logger.info("Starting complete integration example")
    return logger


def _shared_domain_vo_demo() -> tuple[EmailAddress, Money]:
    email = EmailAddress(email="customer@example.com")
    money = Money(amount=Decimal("100.0"), currency="USD")
    return email, money


def _create_customer(email: EmailAddress) -> FlextResult[User]:
    return SharedDomainFactory.create_user(
        name="John Customer",
        email=email.email,
        age=35,
    )


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
    return SharedDomainFactory.create_order(
        customer_id=customer_id,
        items=cast("list[dict[str, object]]", order_items),
    )


def _print_order(order: Order) -> None:
    pass


def _demo_command_pattern(
    customer_id: FlextTypes.Core.String,
    order_items: list[dict[FlextTypes.Core.String, FlextTypes.Core.String]],
) -> None:
    class CreateOrderCommand:
        """Command pattern for order creation with FlextTypes annotations."""

        def __init__(
            self,
            customer_id: FlextTypes.Core.String,
            items: list[dict[FlextTypes.Core.String, FlextTypes.Core.String]],
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
            self.orders: dict[FlextTypes.Core.String, object] = {}

        def save(self, order: object) -> FlextResult[object]:
            """Save order with FlextResult error handling."""
            if hasattr(order, "id"):
                self.orders[order.id] = order  # type: ignore[attr-defined]
                return FlextResult[object].ok(order)
            return FlextResult[object].fail("Order must have an id")

        def get_by_id(self, order_id: FlextTypes.Core.String) -> FlextResult[object]:
            """Retrieve order by ID with FlextResult."""
            if order_id in self.orders:
                return FlextResult[object].ok(self.orders[order_id])
            return FlextResult[object].fail(f"Order {order_id} not found")

    repository = OrderRepository()
    repository.save(order)
    fetch_result = repository.get_by_id(order.id.root)  # type: ignore[attr-defined]
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
    order2_result = SharedDomainFactory.create_order(
        customer_id=str(customer.id),
        items=[
            {
                "product_id": "product456",
                "product_name": "Another Product",
                "quantity": "2",
                "unit_price": "50.0",
                "currency": "USD",
            },
        ],
    )
    # Modern pattern: Check success and use value directly
    if order2_result.success:
        order2 = order2_result.value
        repository_any = FlextContainer.get_global().get("order_repository").value
        # Hint to type checker
        repo: _OrderRepositoryProtocol = cast(
            "_OrderRepositoryProtocol", repository_any
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
