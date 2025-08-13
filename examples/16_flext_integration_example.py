#!/usr/bin/env python3
"""FLEXT Core - Complete Integration Example.

Comprehensive example showing all FLEXT components working together with shared
domain models.
"""

from decimal import Decimal
from typing import cast

from flext_core import (
    FlextFields,
    FlextLogger,
    FlextResult,
    get_flext_container,
    get_logger,
)

from .shared_domain import EmailAddress, Money, Order, SharedDomainFactory, User

# =============================================================================
# VALIDATION CONSTANTS - Integration example constraints
# =============================================================================

# Email validation constants
MIN_EMAIL_LENGTH = 5  # Minimum characters for basic email validation


def _print_header() -> None:
    print("=== FLEXT Core Complete Integration Example ===\n")


def _setup_logger() -> FlextLogger:
    logger = get_logger("integration_example")
    logger.info("Starting complete integration example")
    return logger


def _shared_domain_vo_demo() -> tuple[EmailAddress, Money]:
    email = EmailAddress(email="customer@example.com")
    money = Money(amount=Decimal("100.0"), currency="USD")
    print("1. Shared Domain Value Objects:")
    print(f"  Email: {email.email}")
    print(f"  Money: {money.amount} {money.currency}")
    print()
    return email, money


def _create_customer(email: EmailAddress) -> FlextResult[User]:
    print("2. Using Shared Domain User:")
    return SharedDomainFactory.create_user(
        name="John Customer",
        email=email.email,
        age=35,
    )


def _print_customer(customer: User) -> None:
    print(f"  Customer: {customer.name} ({customer.email_address.email})")
    print()


def _create_order(customer_id: str, money: Money) -> FlextResult[Order]:
    print("3. Using Shared Domain Order:")
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
    print(f"  Order: {order.id}, Status: {order.status}")
    print(f"  Customer: {order.customer_id}")
    print(f"  Events: {len(order.domain_events)} domain events")
    print()


def _demo_command_pattern(customer_id: str, order_items: list[dict[str, str]]) -> None:
    print("4. Command Pattern with Shared Domain:")

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
    validation = create_command.validate_command()
    print(f"  Command validation: {validation.success}")
    print()


def _demo_repository_pattern(order: Order) -> object:
    print("5. Repository Pattern with Shared Domain:")

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
    save_result = repository.save(order)
    print(f"  Order saved: {save_result.success}")
    fetch_result = repository.get_by_id(order.id)
    if fetch_result.success:
        fetched_order = fetch_result.data
        print(f"  Order fetched: {hasattr(fetched_order, 'id')}")
    print()
    return repository


class _OrderRepositoryProtocol:
    def save(self, order: Order) -> object: ...  # pragma: no cover


def _demo_container(customer: User, repository: object) -> None:
    print("6. Dependency Injection Example:")
    container = get_flext_container()
    container.register("order_repository", repository)
    container.register("current_customer", customer)
    repo_result = container.get("order_repository")
    customer_result = container.get("current_customer")
    if repo_result.success and customer_result.success:
        print("  Services registered and retrieved successfully")
        retrieved_customer = customer_result.data
        if hasattr(retrieved_customer, "name"):
            print(f"  Retrieved customer: {retrieved_customer.name}")
        else:
            print(f"  Retrieved customer: {retrieved_customer}")
    print()


def _demo_fields_validation() -> None:
    print("7. Field Validation Example:")
    email_field = FlextFields.create_string_field(
        field_id="customer_email",
        field_name="email",
        pattern=r"^[^@]+@[^@]+\.[^@]+$",
        required=True,
    )
    valid_test = email_field.validate_value("alice@example.com")
    invalid_test = email_field.validate_value("invalid-email")
    print(f"  Valid email validation: {valid_test.success}")
    print(f"  Invalid email validation: {invalid_test.success}")
    if invalid_test.is_failure:
        print(f"    Error: {invalid_test.error}")
    print()


def _demo_complete_flow(
    customer: User, order: Order, logger: FlextLogger
) -> None:
    print("8. Complete Integration Flow:")
    order2_result = SharedDomainFactory.create_order(
        customer_id=customer.id,
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
    if order2_result.success and order2_result.data is not None:
        order2 = order2_result.data
        repository_any = get_flext_container().get("order_repository").unwrap()
        # Hint to type checker
        repo: _OrderRepositoryProtocol = repository_any  # type: ignore[assignment]
        repo.save(order2)
        print(f"  Created second order: {order2.id}")
        total1_result = order.calculate_total()
        total2_result = order2.calculate_total()
        if total1_result.success and total2_result.success:
            total1 = total1_result.data
            total2 = total2_result.data
            if total1 is not None and total2 is not None:
                combined_amount = total1.amount + total2.amount
                print(f"  Order 1 total: {total1.amount} {total1.currency}")
                print(f"  Order 2 total: {total2.amount} {total2.currency}")
                print(f"  Combined total: {combined_amount} USD")
    logger.info("Integration example completed successfully")
    print("\n=== Complete Integration Example Finished! ===")


def main() -> None:
    """Execute main function for integration example."""
    _print_header()
    logger = _setup_logger()
    email, money = _shared_domain_vo_demo()
    user_result = _create_customer(email)
    if user_result.is_failure or user_result.data is None:
        print(f"Failed to create user: {user_result.error}")
        return
    customer = user_result.data
    _print_customer(customer)
    order_result = _create_order(customer.id, money)
    if order_result.is_failure or order_result.data is None:
        print(f"Failed to create order: {order_result.error}")
        return
    order = order_result.data
    _print_order(order)
    _demo_command_pattern(customer.id, [
        {
            "product_id": "product123",
            "product_name": "Test Product",
            "quantity": "1",
            "unit_price": str(money.amount),
            "currency": money.currency,
        },
    ])
    repository = _demo_repository_pattern(order)
    _demo_container(customer, repository)
    _demo_fields_validation()
    _demo_complete_flow(customer, order, logger)


if __name__ == "__main__":
    main()
