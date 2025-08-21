#!/usr/bin/env python3
"""FLEXT Core - Complete Integration Example.

Comprehensive example showing all FLEXT components working together with shared
domain models.
"""

from decimal import Decimal
from typing import cast

# use .shared_domain with dot to access local module
from shared_domain import EmailAddress, Money, Order, SharedDomainFactory, User

from flext_core import (
    FlextFields,
    FlextLogger,
    FlextResult,
    get_flext_container,
    get_logger,
)

# =============================================================================
# VALIDATION CONSTANTS - Integration example constraints
# =============================================================================

# Email validation constants
MIN_EMAIL_LENGTH = 5  # Minimum characters for basic email validation


def _print_header() -> None:
    pass


def _setup_logger() -> FlextLogger:
    logger = get_logger("integration_example")
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


def _demo_command_pattern(customer_id: str, order_items: list[dict[str, str]]) -> None:
    class CreateOrderCommand:
        def __init__(self, customer_id: str, items: list[dict[str, str]]) -> None:
            self.customer_id = customer_id
            self.items = items

        def validate_command(self) -> FlextResult[None]:
            if not self.customer_id.strip():
                return FlextResult.fail("Customer ID required")
            if not self.items:
                return FlextResult.fail("Order items required")
            return FlextResult.ok(None)

    create_command = CreateOrderCommand(customer_id=customer_id, items=order_items)
    create_command.validate_command()


def _demo_repository_pattern(order: Order) -> object:
    class OrderRepository:
        def __init__(self) -> None:
            self.orders: dict[str, object] = {}

        def save(self, order: object) -> FlextResult[object]:
            if hasattr(order, "id"):
                self.orders[order.id] = order
                return FlextResult.ok(order)
            return FlextResult.fail("Order must have an id")

        def get_by_id(self, order_id: str) -> FlextResult[object]:
            if order_id in self.orders:
                return FlextResult.ok(self.orders[order_id])
            return FlextResult.fail(f"Order {order_id} not found")

    repository = OrderRepository()
    repository.save(order)
    fetch_result = repository.get_by_id(order.id.root)
    if fetch_result.success:
        pass
    return repository


class _OrderRepositoryProtocol:
    def save(self, order: Order) -> object: ...  # pragma: no cover


def _demo_container(customer: User, repository: object) -> None:
    container = get_flext_container()
    container.register("order_repository", repository)
    container.register("current_customer", customer)
    repo_result = container.get("order_repository")
    customer_result = container.get("current_customer")
    if repo_result.success and customer_result.success:
        retrieved_customer = customer_result.value
        if hasattr(retrieved_customer, "name"):
            pass


def _demo_fields_validation() -> None:
    email_field = FlextFields.create_string_field(
        field_id="customer_email",
        field_name="email",
        pattern=r"^[^@]+@[^@]+\.[^@]+$",
        required=True,
    )
    email_field.validate_value("alice@example.com")
    invalid_test = email_field.validate_value("invalid-email")
    if invalid_test.is_failure:
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
    if order2_result.success and order2_result.value is not None:
        order2 = order2_result.value
        repository_any = get_flext_container().get("order_repository").value
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
