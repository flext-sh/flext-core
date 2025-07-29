#!/usr/bin/env python3
"""FLEXT Core - Advanced Examples.

Advanced patterns and enterprise scenarios using FLEXT Core with shared domain models.
"""

# Import shared domain models to eliminate duplication
from shared_domain import (
    EmailAddress,
    Money,
    Order as SharedOrder,
    SharedDomainFactory,
)

from flext_core import (
    FlextCommands,
    FlextDecorators,
    FlextResult,
    get_logger,
)

# =============================================================================
# VALIDATION CONSTANTS - Domain rule constraints
# =============================================================================

# Currency validation constants
CURRENCY_CODE_LENGTH = 3  # Standard ISO 4217 currency code length


def main() -> None:  # noqa: PLR0915
    """Execute main function for advanced examples."""
    print("=== FLEXT Core Advanced Examples ===\n")

    # 1. Value Objects
    print("1. FlextValueObject Examples (using shared domain):")

    # Use shared domain models instead of local definitions
    email = EmailAddress(email="user@example.com")
    money = Money(amount=100.50, currency="USD")

    print(f"  Email: {email.email}")
    print(f"  Money: {money.amount} {money.currency}")

    # Test equality (value objects are equal by value, not identity)
    email2 = EmailAddress(email="user@example.com")
    print(f"  Email equality: {email == email2}")  # Should be True
    print()

    # 2. Aggregate Root (using shared domain)
    print("2. FlextAggregateRoot Examples (using shared domain):")

    # Use SharedOrder from shared_domain instead of local Order class

    # Create order using SharedDomainFactory
    user_result = SharedDomainFactory.create_user(
        name="Test User",
        email=email.email,
        age=30,
    )

    if user_result.is_success:
        user = user_result.data
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

        if order_result.is_success:
            order = order_result.data
        else:
            print(f"Failed to create order: {order_result.error}")
            return
    else:
        print(f"Failed to create user: {user_result.error}")
        return

    print(f"  Order: {order.id}, Status: {order.status.value}")
    print(f"  Customer: {order.customer_id}")
    total_result = order.calculate_total()
    if total_result.is_success:
        total = total_result.data
        print(f"  Total: {total.amount} {total.currency}")

    # SharedOrder has built-in domain events
    print(f"  Events: {len(order.domain_events)} domain events")
    print()

    # 3. Query Pattern
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

            # Simulate database query with SharedOrder
            order1_result = SharedDomainFactory.create_order(
                customer_id=user.id,
                items=[{
                    "product_id": "product1",
                    "product_name": "Product 1",
                    "quantity": 1,
                    "unit_price": "50.0",
                    "currency": "USD",
                }],
            )

            order2_result = SharedDomainFactory.create_order(
                customer_id=user.id,
                items=[{
                    "product_id": "product2",
                    "product_name": "Product 2",
                    "quantity": 1,
                    "unit_price": "75.0",
                    "currency": "USD",
                }],
            )

            orders = []
            if order1_result.is_success:
                orders.append(order1_result.data)
            if order2_result.is_success:
                orders.append(order2_result.data)

            # Filter by status if provided
            if query.status:
                orders = [o for o in orders if o.status.value == query.status]

            return FlextResult.ok(orders)

    query = GetOrdersQuery(
        customer_email="user@example.com",
        status="confirmed",
        page_size=10,
    )

    query_handler = GetOrdersHandler()

    # Validate query
    validation = query.validate_query()
    print(f"  Query validation: {validation.is_success}")

    if validation.is_success:
        query_result = query_handler.handle(query)
        if query_result.is_success:
            orders = query_result.data
            print(f"  Found {len(orders)} orders")
            for o in orders:
                print(f"    Order {o.id}: {o.status} - {o.total.amount}")
    print()

    # 4. Decorators
    print("4. FlextDecorators Examples:")

    @FlextDecorators.safe_result
    def risky_calculation(x: float, y: float) -> float:
        if y == 0:
            msg = "Division by zero"
            raise ValueError(msg)
        return x / y

    # Test safe execution
    safe_result = risky_calculation(10.0, 2.0)
    print(f"  Safe calculation: {safe_result.is_success}, Result: {safe_result.data}")

    error_result = risky_calculation(10.0, 0.0)
    print(f"  Error handling: {error_result.is_failure}, Error: {error_result.error}")
    print()

    # 5. Logging
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
    except Exception as e:
        logger.exception("Calculation failed", error=str(e))

    print("  Check logs above for structured logging output")
    print()

    # 6. Configuration
    print("6. FlextConfig Examples:")

    from flext_core.config import FlextBaseSettings  # noqa: PLC0415

    class AppSettings(FlextBaseSettings):
        database_url: str = "sqlite:///app.db"
        debug: bool = False
        max_workers: int = 4
        api_key: str = "default-key"

        class Config:
            env_prefix = "APP_"

    # Create settings
    settings_result = AppSettings.create_with_validation(
        {
            "debug": True,
            "max_workers": 8,
        },
    )

    if settings_result.is_success:
        settings = settings_result.data
        print(f"  Database URL: {settings.database_url}")
        print(f"  Debug mode: {settings.debug}")
        print(f"  Max workers: {settings.max_workers}")
    print()

    print("=== Advanced Examples Completed Successfully! ===")


if __name__ == "__main__":
    main()
