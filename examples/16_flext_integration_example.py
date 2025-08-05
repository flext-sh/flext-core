#!/usr/bin/env python3
"""FLEXT Core - Complete Integration Example.

Comprehensive example showing all FLEXT components working together with shared
domain models.
"""

from decimal import Decimal
from typing import cast

# Import shared domain models to eliminate duplication
from shared_domain import (
    EmailAddress,
    Money,
    SharedDomainFactory,
)

from flext_core import (
    FlextCommands,
    FlextFields,
    FlextResult,
    get_flext_container,
    get_logger,
)

# =============================================================================
# VALIDATION CONSTANTS - Integration example constraints
# =============================================================================

# Email validation constants
MIN_EMAIL_LENGTH = 5  # Minimum characters for basic email validation


def main() -> None:  # noqa: PLR0915, PLR0912
    """Execute main function for integration example."""
    print("=== FLEXT Core Complete Integration Example ===\n")

    # Setup logging
    logger = get_logger("integration_example")
    logger.info("Starting complete integration example")

    # 1. Use Shared Domain Value Objects
    # Using EmailAddress and Money from shared_domain instead of local definitions
    email = EmailAddress(email="customer@example.com")
    money = Money(amount=Decimal("100.0"), currency="USD")

    print("1. Shared Domain Value Objects:")
    print(f"  Email: {email.email}")
    print(f"  Money: {money.amount} {money.currency}")
    print()

    # 2. Use Shared Domain Entities
    print("2. Using Shared Domain User:")
    user_result = SharedDomainFactory.create_user(
        name="John Customer",
        email=email.email,
        age=35,
    )

    if user_result.is_failure:
        print(f"Failed to create user: {user_result.error}")
        return

    customer = user_result.data

    if customer is None:
        print("❌ Operation returned None data")

        return
    print(f"  Customer: {customer.name} ({customer.email_address.email})")
    print()

    # 3. Use Shared Domain Order
    print("3. Using Shared Domain Order:")
    order_items = [
        {
            "product_id": "product123",
            "product_name": "Test Product",
            "quantity": "1",  # String format for command validation
            "unit_price": str(money.amount),
            "currency": money.currency,
        },
    ]

    order_result = SharedDomainFactory.create_order(
        customer_id=customer.id,
        items=cast("list[dict[str, object]]", order_items),
    )

    if order_result.is_failure:
        print(f"Failed to create order: {order_result.error}")
        return

    order = order_result.data

    if order is None:
        print("❌ Operation returned None data")

        return
    print(f"  Order: {order.id}, Status: {order.status.value}")
    print(f"  Customer: {order.customer_id}")
    print(f"  Events: {len(order.domain_events)} domain events")
    print()

    # 4. Command Pattern Example
    print("4. Command Pattern with Shared Domain:")

    class CreateOrderCommand(FlextCommands.Command):
        customer_id: str
        items: list[dict[str, str]]

        def validate_command(self) -> FlextResult[None]:
            if not self.customer_id.strip():
                return FlextResult.fail("Customer ID required")
            if not self.items:
                return FlextResult.fail("Order items required")
            return FlextResult.ok(None)

    # Test command
    create_command = CreateOrderCommand(
        customer_id=customer.id,
        items=order_items,
    )

    validation = create_command.validate_command()
    print(f"  Command validation: {validation.success}")
    print()

    # 5. Simple Repository Pattern using shared domain
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

    # Test repository with shared domain order
    repository = OrderRepository()
    save_result = repository.save(order)
    print(f"  Order saved: {save_result.success}")

    fetch_result = repository.get_by_id(order.id)
    if fetch_result.success:
        fetched_order = fetch_result.data
        print(f"  Order fetched: {hasattr(fetched_order, 'id')}")
    print()

    # 6. Dependency Injection with Container
    print("6. Dependency Injection Example:")

    container = get_flext_container()

    # Register repository
    container.register("order_repository", repository)

    # Register customer
    container.register("current_customer", customer)

    # Retrieve services
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

    # 7. Field Validation Example
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

    # 8. Complete Integration Flow
    print("8. Complete Integration Flow:")

    # Create multiple orders for the customer
    order2_result = SharedDomainFactory.create_order(
        customer_id=customer.id,
        items=[
            {
                "product_id": "product456",
                "product_name": "Another Product",
                "quantity": "2",  # String format
                "unit_price": "50.0",
                "currency": "USD",
            },
        ],
    )

    if order2_result.success:
        order2 = order2_result.data

        if order2 is None:
            print("❌ Operation returned None data")

            return
        repository.save(order2)
        print(f"  Created second order: {order2.id}")

        # Calculate totals
        total1_result = order.calculate_total()
        total2_result = order2.calculate_total()

        if total1_result.success and total2_result.success:
            total1 = total1_result.data
            total2 = total2_result.data
            if total1 is None or total2 is None:
                print("❌ Total calculation returned None data")
                return

            combined_amount = total1.amount + total2.amount
            print(f"  Order 1 total: {total1.amount} {total1.currency}")
            print(f"  Order 2 total: {total2.amount} {total2.currency}")
            print(f"  Combined total: {combined_amount} USD")

    logger.info("Integration example completed successfully")
    print("\n=== Complete Integration Example Finished! ===")


if __name__ == "__main__":
    main()
