#!/usr/bin/env python3
"""FLEXT Core - Advanced Examples.

Advanced patterns and enterprise scenarios using FLEXT Core with shared domain models.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, cast

from flext_core import (
    FlextCommands,
    FlextDecorators,
    FlextResult,
    FlextSettings,
    get_logger,
)

from .shared_domain import (
    EmailAddress,
    Money,
    Order as SharedOrder,
    SharedDomainFactory,
    User as SharedUser,
)

if TYPE_CHECKING:
    from flext_core import FlextDecoratedFunction

# =============================================================================
# VALIDATION CONSTANTS - Domain rule constraints
# =============================================================================

# Currency validation constants
CURRENCY_CODE_LENGTH = 3  # Standard ISO 4217 currency code length


def main() -> None:
    """Execute main function for advanced examples using railway-oriented programming."""
    print("=== FLEXT Core Advanced Examples ===\n")

    # Chain all demonstration functions using railway-oriented programming
    result = (
        _demonstrate_value_objects()
        .flat_map(lambda _: _demonstrate_aggregate_root())
        .flat_map(_demonstrate_query_pattern)
        .flat_map(lambda _: _demonstrate_decorators())
        .flat_map(lambda _: _demonstrate_logging())
        .flat_map(lambda _: _demonstrate_configuration())
    )

    if result.success:
        print("\n=== Advanced Examples Completed Successfully! ===")
    else:
        print(f"\n❌ Advanced examples failed: {result.error}")


def _demonstrate_value_objects() -> FlextResult[None]:
    """Demonstrate value objects using shared domain models."""
    print("1. FlextValueObject Examples (using shared domain):")

    # Use shared domain models instead of local definitions
    email = EmailAddress(email="user@example.com")
    money = Money(amount=Decimal("100.50"), currency="USD")

    print(f"  Email: {email.email}")
    print(f"  Money: {money.amount} {money.currency}")

    # Test equality (value objects are equal by value, not identity)
    email2 = EmailAddress(email="user@example.com")
    print(f"  Email equality: {email == email2}")  # Should be True
    print()

    return FlextResult.ok(None)


def _demonstrate_aggregate_root() -> FlextResult[SharedOrder]:
    """Demonstrate aggregate root pattern using shared domain models."""
    print("2. FlextAggregateRoot Examples (using shared domain):")

    email = EmailAddress(email="user@example.com")
    money = Money(amount=Decimal("100.50"), currency="USD")

    return (
        _create_test_user(email)
        .flat_map(lambda user: _create_test_order(user, money))
        .flat_map(_display_order_information)
    )


def _create_test_user(email: EmailAddress) -> FlextResult[SharedUser]:
    """Create test user for demonstration."""
    user_result = SharedDomainFactory.create_user(
        name="Test User",
        email=email.email,
        age=30,
    )

    if user_result.success:
        user = user_result.data
        if user is None:
            return FlextResult.fail("User creation returned None data")
        return FlextResult.ok(user)
    return FlextResult.fail(f"Failed to create user: {user_result.error}")


def _create_test_order(user: SharedUser, money: Money) -> FlextResult[SharedOrder]:
    """Create test order for demonstration."""
    order_items = [
        {
            "product_id": "product123",
            "product_name": "Test Product",
            "quantity": 1,
            "unit_price": str(money.amount),
            "currency": money.currency,
        },
    ]

    order_result = SharedDomainFactory.create_order(
        customer_id=user.id,
        items=order_items,
    )

    if order_result.success:
        order = order_result.data
        if order is None:
            return FlextResult.fail("Order creation returned None data")
        return FlextResult.ok(order)
    return FlextResult.fail(f"Failed to create order: {order_result.error}")


def _display_order_information(order: SharedOrder) -> FlextResult[SharedOrder]:
    """Display order information and return order for chaining."""
    print(f"  Order: {order.id}, Status: {order.status.value}")
    print(f"  Customer: {order.customer_id}")

    total_result = order.calculate_total()
    if total_result.success:
        total = total_result.data
        if total is None:
            return FlextResult.fail("Total calculation returned None data")
        print(f"  Total: {total.amount} {total.currency}")

    # SharedOrder has built-in domain events
    print(f"  Events: {len(order.domain_events)} domain events")
    print()

    return FlextResult.ok(order)


def _demonstrate_query_pattern(_order: SharedOrder) -> FlextResult[None]:
    """Demonstrate CQRS query pattern."""
    print("3. FlextCommands Query Examples:")

    class GetOrdersQuery(FlextCommands.Query):
        customer_email: str
        status: str | None = None

        def validate_query(self) -> FlextResult[None]:
            """Validate custom query."""
            result = super().validate_query()  # Call base validation
            if result.is_failure:
                return result

            if not self.customer_email or "@" not in self.customer_email:
                return FlextResult.fail("Invalid customer email")
            return FlextResult.ok(None)

    class GetOrdersHandler(
        FlextCommands.QueryHandler[GetOrdersQuery, list[SharedOrder]],
    ):
        def handle(self, query: GetOrdersQuery) -> FlextResult[list[SharedOrder]]:
            # Create user for order simulation
            user_result = SharedDomainFactory.create_user(
                name="Query User",
                email=query.customer_email,
                age=30,
            )

            if user_result.is_failure:
                return FlextResult.fail(f"Failed to create user: {user_result.error}")

            user = user_result.data

            if user is None:
                print("❌ Operation returned None data")
                return FlextResult.fail("User creation returned None data")

            # Simulate database query with SharedOrder
            order1_result = SharedDomainFactory.create_order(
                customer_id=user.id,
                items=[
                    {
                        "product_id": "product1",
                        "product_name": "Product 1",
                        "quantity": 1,
                        "unit_price": "50.0",
                        "currency": "USD",
                    },
                ],
            )

            order2_result = SharedDomainFactory.create_order(
                customer_id=user.id,
                items=[
                    {
                        "product_id": "product2",
                        "product_name": "Product 2",
                        "quantity": 1,
                        "unit_price": "75.0",
                        "currency": "USD",
                    },
                ],
            )

            orders = []
            if order1_result.success:
                order1 = order1_result.data

                if order1 is None:
                    print("❌ Operation returned None data")
                    return FlextResult.fail("Order creation returned None data")
                orders.append(order1)
            if order2_result.success:
                order2 = order2_result.data

                if order2 is None:
                    print("❌ Operation returned None data")
                    return FlextResult.fail("Order creation returned None data")
                orders.append(order2)

            # Filter by status if provided
            if query.status:
                orders = [
                    o
                    for o in orders
                    if o is not None and o.status.value == query.status
                ]

            return FlextResult.ok(orders)

    query = GetOrdersQuery(
        customer_email="user@example.com",
        status="confirmed",
        page_size=10,
    )

    query_handler = GetOrdersHandler()

    # Validate query
    validation = query.validate_query()
    print(f"  Query validation: {validation.success}")

    if validation.success:
        query_result = query_handler.handle(query)
        if query_result.success:
            orders = query_result.data

            if orders is None:
                print("❌ Operation returned None data")
                return FlextResult.fail("Orders query returned None data")
            print(f"  Found {len(orders)} orders")
            for o in orders:
                if hasattr(o, "total") and o.total is not None:
                    print(f"    Order {o.id}: {o.status} - {o.total.amount}")
                else:
                    print(f"    Order {o.id}: {o.status} - No total available")
    print()
    return FlextResult.ok(None)


def _demonstrate_decorators() -> FlextResult[None]:
    """Demonstrate decorator patterns."""
    print("4. FlextDecorators Examples:")

    def risky_calculation(x: float, y: float) -> float:
        if y == 0:
            msg = "Division by zero"
            raise ValueError(msg)
        return x / y

    # Apply decorator using casting for protocol compliance
    safe_calculation = FlextDecorators.safe_result(
        cast("FlextDecoratedFunction", risky_calculation)
    )

    # Test safe execution
    safe_result = safe_calculation(10.0, 2.0)
    print(
        f"  Safe calculation: {getattr(safe_result, 'success', False)}, Result: {getattr(safe_result, 'data', 'N/A')}"
    )

    error_result = safe_calculation(10.0, 0.0)
    print(
        f"  Error handling: {getattr(error_result, 'is_failure', False)}, Error: {getattr(error_result, 'error', 'N/A')}"
    )
    print()
    return FlextResult.ok(None)


def _demonstrate_logging() -> FlextResult[None]:
    """Demonstrate structured logging patterns."""
    print("5. FlextLogger Examples:")

    logger = get_logger("advanced_examples")

    # Structured logging with context
    logger.info(
        "Starting advanced processing",
        operation="advanced_example",
        user_id="user123",
        request_id="req456",
    )

    # Context logging
    logger.set_context({"component": "order_service", "version": "1.0"})
    logger.info("Order processing started", order_id="order123")

    try:
        # Simulate some work
        result = 100 / 10
        logger.info("Calculation completed", result=result)
    except (RuntimeError, ValueError, TypeError) as e:
        logger.exception("Calculation failed", error=str(e))

    print("  Check logs above for structured logging output")
    print()
    return FlextResult.ok(None)


def _demonstrate_configuration() -> FlextResult[None]:
    """Demonstrate configuration management patterns."""
    print("6. FlextConfig Examples:")

    class AppSettings(FlextSettings):
        database_url: str = "sqlite:///app.db"
        debug: bool = False
        max_workers: int = 4
        api_key: str = "default-key"

        class Config:
            env_prefix = "APP_"

    # Create settings
    try:
        settings = AppSettings(
            debug=True,
            max_workers=8,
        )
        settings_result = FlextResult.ok(settings)
    except Exception as e:
        settings_result = FlextResult.fail(f"Configuration creation failed: {e}")

    if settings_result.success:
        settings = settings_result.unwrap()

        print(f"  Database URL: {getattr(settings, 'database_url', 'N/A')}")
        print(f"  Debug mode: {getattr(settings, 'debug', 'N/A')}")
        print(f"  Max workers: {getattr(settings, 'max_workers', 'N/A')}")
    print()
    return FlextResult.ok(None)


if __name__ == "__main__":
    main()
