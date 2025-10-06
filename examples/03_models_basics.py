# !/usr/bin/env python3
"""03 - FlextModels Fundamentals: Complete Domain-Driven Design Patterns.

This example demonstrates the COMPLETE FlextModels API - the foundation
for domain modeling across the entire FLEXT ecosystem. FlextModels provides
DDD patterns with Value Objects, Entities, and Aggregate Roots.

Key Concepts Demonstrated:
- Value Objects: Flext.Models.Value for immutable value types
- Entities: Flext.Models.Entity with identity and lifecycle
- Aggregate Roots: Flext.Models.AggregateRoot for consistency boundaries
- Domain Events: Event sourcing patterns (AggregateRoot only)
- Business Rules: Invariant enforcement and validation
- Model Serialization: to_dict(), from_dict(), model_dump()
- Comparison & Equality: Value-based vs identity-based
- Factory Methods: Static creation methods with validation
- Repository Patterns: Integration with persistence

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import warnings
from datetime import UTC, datetime
from decimal import Decimal
from typing import cast
from uuid import uuid4

from pydantic import Field

from flext_core import Flext

from .example_scenarios import ExampleScenarios, RealisticDataDict, RealisticOrderDict

# ========== VALUE OBJECTS ==========


class Email(Flext.Models.Value):
    """Email value object - immutable and compared by value."""

    address: str

    def validate_email(self) -> Flext.Result[None]:
        """Business validation for email format."""
        if "@" not in self.address:
            return Flext.Result[None].fail("Invalid email format")
        if self.address.lower() != self.address:
            return Flext.Result[None].fail("Email must be lowercase")
        return Flext.Result[None].ok(None)

    @classmethod
    def create_email(cls, *args: object, **kwargs: object) -> Flext.Result[Email]:
        """Factory method with validation."""
        address_raw = args[0] if args else kwargs.get("address", "")
        address = str(address_raw) if address_raw is not None else ""
        email = cls(address=address.lower().strip())
        validation = email.validate_email()
        if validation.is_failure:
            return Flext.Result[Email].fail(
                f"Email creation failed: {validation.error}"
            )
        return Flext.Result[Email].ok(email)


class Money(Flext.Models.Value):
    """Money value object with currency and amount."""

    amount: Decimal
    currency: str = "USD"

    def add(self, other: Money) -> Flext.Result[Money]:
        """Add money with currency check."""
        if self.currency != other.currency:
            return Flext.Result[Money].fail("Currency mismatch")
        return Flext.Result[Money].ok(
            Money(amount=self.amount + other.amount, currency=self.currency),
        )

    def subtract(self, other: Money) -> Flext.Result[Money]:
        """Subtract money with currency check."""
        if self.currency != other.currency:
            return Flext.Result[Money].fail("Currency mismatch")
        if self.amount < other.amount:
            return Flext.Result[Money].fail("Insufficient funds")
        return Flext.Result[Money].ok(
            Money(amount=self.amount - other.amount, currency=self.currency),
        )


class Address(Flext.Models.Value):
    """Complex value object with multiple fields."""

    street: str
    city: str
    state: str
    zip_code: str
    country: str = "USA"

    def format_for_shipping(self) -> str:
        """Format address for shipping label."""
        return (
            f"{self.street}\n{self.city}, {self.state} {self.zip_code}\n{self.country}"
        )


# ========== ENTITIES ==========


class Product(Flext.Models.Entity):
    """Product entity with identity and business logic."""

    name: str
    price: Money
    sku: str
    stock: int = 0
    is_active: bool = True

    def adjust_price(self, new_price: Money) -> Flext.Result[None]:
        """Adjust product price with validation."""
        if new_price.amount <= Decimal(0):
            return Flext.Result[None].fail("Price must be positive")

        self.price = new_price

        # Note: Entity doesn't have domain events, only AggregateRoot does

        return Flext.Result[None].ok(None)

    def add_stock(self, quantity: int) -> Flext.Result[None]:
        """Add stock with validation."""
        if quantity <= 0:
            return Flext.Result[None].fail("Quantity must be positive")

        self.stock += quantity
        # Note: Entity doesn't have domain events
        return Flext.Result[None].ok(None)

    def remove_stock(self, quantity: int) -> Flext.Result[None]:
        """Remove stock with availability check."""
        if quantity <= 0:
            return Flext.Result[None].fail("Quantity must be positive")
        if quantity > self.stock:
            return Flext.Result[None].fail("Insufficient stock")

        self.stock -= quantity
        # Note: Entity doesn't have domain events
        return Flext.Result[None].ok(None)


class Customer(Flext.Models.Entity):
    """Customer entity with complex business rules."""

    name: str
    email: Email
    shipping_address: Address
    billing_address: Address
    credit_limit: Money
    current_balance: Money
    is_vip: bool = False
    created_at: datetime = datetime.now(UTC)

    def model_post_init(self, /, __context: object) -> None:
        """Initialize with timestamp after Pydantic model creation."""
        if not hasattr(self, "created_at"):
            self.created_at = datetime.now(UTC)

    def can_purchase(self, amount: Money) -> Flext.Result[bool]:
        """Check if customer can make purchase."""
        # Check credit limit
        available_credit_result = self.credit_limit.subtract(self.current_balance)
        if available_credit_result.is_failure:
            return Flext.Result[bool].ok(False)

        available_credit = available_credit_result.unwrap()
        if amount.amount > available_credit.amount:
            return Flext.Result[bool].fail("Exceeds credit limit")

        return Flext.Result[bool].ok(True)

    def make_purchase(self, amount: Money) -> Flext.Result[None]:
        """Process a purchase."""
        can_purchase = self.can_purchase(amount)
        if can_purchase.is_failure or not can_purchase.unwrap():
            return Flext.Result[None].fail("Cannot complete purchase")

        # Update balance
        new_balance_result = self.current_balance.add(amount)
        if new_balance_result.is_failure:
            return Flext.Result[None].fail(new_balance_result.error or "Unknown error")

        self.current_balance = new_balance_result.unwrap()

        # Note: Entity doesn't have domain events

        return Flext.Result[None].ok(None)

    def upgrade_to_vip(self) -> Flext.Result[None]:
        """Upgrade customer to VIP status."""
        if self.is_vip:
            return Flext.Result[None].fail("Already VIP")

        self.is_vip = True
        # VIP gets higher credit limit
        self.credit_limit = Money(
            amount=self.credit_limit.amount * Decimal(2),
            currency=self.credit_limit.currency,
        )

        # Note: Entity doesn't have domain events

        return Flext.Result[None].ok(None)


# ========== AGGREGATE ROOTS ==========


class OrderLine(Flext.Models.Value):
    """Order line value object."""

    product_id: str
    product_name: str
    quantity: int
    unit_price: Money

    def calculate_total(self) -> Money:
        """Calculate line total."""
        return Money(
            amount=self.unit_price.amount * Decimal(self.quantity),
            currency=self.unit_price.currency,
        )


class Order(Flext.Models.AggregateRoot):
    """Order aggregate root - maintains consistency boundary."""

    customer_id: str
    order_number: str
    lines: list[OrderLine] = Field(default_factory=list)
    total_amount: Money = Money(amount=Decimal(0), currency="USD")
    status: str = "DRAFT"
    created_at: datetime = datetime.now(UTC)
    shipped_at: datetime | None = None

    def model_post_init(self, /, __context: object) -> None:
        """Initialize with defaults after Pydantic model creation."""
        if not self.lines:
            self.lines = []
        if not hasattr(self, "created_at"):
            self.created_at = datetime.now(UTC)
        if not hasattr(self, "total_amount"):
            self.total_amount = Money(amount=Decimal(0), currency="USD")

    def add_line(self, product: Product, quantity: int) -> Flext.Result[None]:
        """Add order line with stock validation."""
        # Check if order is modifiable
        if self.status not in {"DRAFT", "PENDING"}:
            return Flext.Result[None].fail(
                f"Cannot modify order in {self.status} status",
            )

        # Check stock availability
        if product.stock < quantity:
            return Flext.Result[None].fail(f"Insufficient stock for {product.name}")

        # Create order line
        line = OrderLine(
            product_id=product.id,
            product_name=product.name,
            quantity=quantity,
            unit_price=product.price,
        )

        self.lines.append(line)

        # Recalculate total
        self._recalculate_total()

        # Add domain event
        self.add_domain_event(
            "OrderLineAdded",
            {
                "order_id": str(self.id),
                "product_id": str(product.id),
                "quantity": quantity,
                "line_total": str(line.calculate_total().amount),
            },
        )

        return Flext.Result[None].ok(None)

    def remove_line(self, product_id: str) -> Flext.Result[None]:
        """Remove order line."""
        if self.status not in {"DRAFT", "PENDING"}:
            return Flext.Result[None].fail(
                f"Cannot modify order in {self.status} status",
            )

        # Find and remove line
        original_count = len(self.lines)
        self.lines = [line for line in self.lines if line.product_id != product_id]

        if len(self.lines) == original_count:
            return Flext.Result[None].fail("Product not found in order")

        # Recalculate total
        self._recalculate_total()

        # Add domain event
        self.add_domain_event(
            "OrderLineRemoved",
            {
                "order_id": str(self.id),
                "product_id": str(product_id),
            },
        )

        return Flext.Result[None].ok(None)

    def _recalculate_total(self) -> None:
        """Recalculate order total (internal consistency)."""
        total = Decimal(0)
        for line in self.lines:
            total += line.calculate_total().amount

        self.total_amount = Money(amount=total, currency="USD")

    def submit(self) -> Flext.Result[None]:
        """Submit order for processing."""
        if self.status != "DRAFT":
            return Flext.Result[None].fail("Only draft orders can be submitted")

        if not self.lines:
            return Flext.Result[None].fail("Cannot submit empty order")

        self.status = "PENDING"

        # Add domain event
        self.add_domain_event(
            "OrderSubmitted",
            {
                "order_id": str(self.id),
                "order_number": self.order_number,
                "total_amount": str(self.total_amount.amount),
                "line_count": len(self.lines),
            },
        )

        return Flext.Result[None].ok(None)

    def ship(self) -> Flext.Result[None]:
        """Mark order as shipped."""
        if self.status != "PENDING":
            return Flext.Result[None].fail("Only pending orders can be shipped")

        self.status = "SHIPPED"
        self.shipped_at = datetime.now(UTC)

        # Add domain event
        self.add_domain_event(
            "OrderShipped",
            {
                "order_id": str(self.id),
                "order_number": self.order_number,
                "shipped_at": self.shipped_at.isoformat(),
            },
        )

        return Flext.Result[None].ok(None)

    def cancel(self, reason: str) -> Flext.Result[None]:
        """Cancel order with reason."""
        if self.status in {"SHIPPED", "DELIVERED", "CANCELLED"}:
            return Flext.Result[None].fail(
                f"Cannot cancel order in {self.status} status",
            )

        self.status = "CANCELLED"

        # Add domain event
        self.add_domain_event(
            "OrderCancelled",
            {
                "order_id": str(self.id),
                "order_number": self.order_number,
                "reason": reason,
                "cancelled_at": datetime.now(UTC).isoformat(),
            },
        )

        return Flext.Result[None].ok(None)


class ComprehensiveModelsService(Flext.Service[Order]):
    """Service demonstrating ALL FlextModels patterns and methods."""

    def __init__(self) -> None:
        """Initialize with dependencies."""
        super().__init__()
        manager: Flext.Container.GlobalManager = Flext.Container.ensure_global_manager()
        self._container = manager.get_or_create()
        self._logger = Flext.Logger(__name__)
        self._scenarios: type[ExampleScenarios] = ExampleScenarios
        self._dataset: Flext.Types.Dict = self._scenarios.dataset()
        self._realistic: RealisticDataDict = self._scenarios.realistic_data()
        self._validation: Flext.Types.Dict = self._scenarios.validation_data()

    def execute(self) -> Flext.Result[Order]:
        """Execute method required by FlextService."""
        # This is a demonstration service, execute creates a sample order
        order = Order(
            id=str(uuid4()),
            customer_id=str(uuid4()),
            order_number=f"ORD-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}",
        )
        return Flext.Result[Order].ok(order)

    # ========== VALUE OBJECTS DEMONSTRATION ==========

    def demonstrate_value_objects(self) -> None:
        """Show value object patterns."""
        print("\n=== Value Objects ===")

        validation_data = self._validation
        sample_email = cast("Flext.Types.List", validation_data["valid_emails"])[0]
        email_result = Email.create_email(sample_email)
        if email_result.is_success:
            email: Email = email_result.unwrap()
            print(f"✅ Email created: {email.address}")

        order_data: RealisticOrderDict = self._realistic["order"]
        money = Money(amount=Decimal(order_data["total"]), currency="USD")
        print(f"Money: ${money.amount} {money.currency}")

        user_reg = self._realistic["user_registration"]
        address_data = cast("Flext.Types.Dict", user_reg["address"])
        address = Address(
            street=str(address_data.get("street", "")),
            city=str(address_data.get("city", "")),
            state=str(address_data.get("state", "")),
            zip_code=str(address_data.get("zip", "")),
        )
        print(f"Address:\n{address.format_for_shipping()}")

        email1 = Email(address=sample_email)
        email2 = Email(address=sample_email)
        print(f"Value equality: {email1 == email2}")
        print(f"Same instance: {email1 is email2}")  # False  # False

    def demonstrate_value_object_operations(self) -> None:
        """Show value object business operations."""
        print("\n=== Value Object Operations ===")

        order_data = self._realistic["order"]
        first_item = order_data["items"][0]
        second_item = (
            order_data["items"][0]
            if len(order_data["items"]) == 1
            else order_data["items"][1]
        )

        money1 = Money(amount=Decimal(first_item["price"]), currency="USD")
        money2 = Money(amount=Decimal(second_item["price"]), currency="USD")

        result = money1.add(money2)
        if result.is_success:
            total = result.unwrap()
            print(f"${money1.amount} + ${money2.amount} = ${total.amount}")

        result = money1.subtract(money2)
        if result.is_success:
            difference = result.unwrap()
            print(f"${money1.amount} - ${money2.amount} = ${difference.amount}")

        euro = Money(amount=money2.amount, currency="EUR")
        result = money1.add(euro)
        if result.is_failure:
            print(f"✅ Currency check: {result.error}")

    # ========== ENTITIES DEMONSTRATION ==========

    def demonstrate_entities(self) -> None:
        """Show entity patterns."""
        print("\n=== Entities ===")

        product_data = self._realistic["order"]["items"][0]
        product = Product(
            id=str(product_data["product_id"]),
            name=str(product_data["name"]),
            price=Money(amount=Decimal(product_data["price"]), currency="USD"),
            sku=str(product_data["product_id"])[:8],
            stock=10,
        )
        print(f"Product: {product.name} (ID: {product.id})")

        result = product.add_stock(5)
        if result.is_success:
            print(f"✅ Stock added, new stock: {product.stock}")

        new_price = Money(
            amount=product.price.amount - Decimal("50.00"),
            currency="USD",
        )
        result = product.adjust_price(new_price)
        if result.is_success:
            print(f"✅ Price adjusted to ${new_price.amount}")

        print("Note: Entity doesn't track domain events (use AggregateRoot)")

    def demonstrate_entity_identity(self) -> None:
        """Show entity identity vs value comparison."""
        print("\n=== Entity Identity ===")

        base_item = self._realistic["order"]["items"][0]
        product1 = Product(
            id=str(base_item["product_id"]),
            name=str(base_item["name"]),
            price=Money(amount=Decimal(base_item["price"]), currency="USD"),
            sku=str(base_item["product_id"])[:8],
        )

        product2 = Product(
            id=product1.id,
            name=f"{base_item['name']} Variant",
            price=Money(amount=Decimal(base_item["price"]), currency="USD"),
            sku=f"{product1.sku}-VAR",
        )

        print(f"Same ID equality: {product1 == product2}")

        alt_item = self._realistic["order"]["items"][0]
        product3 = Product(
            id=str(uuid4()),
            name=str(alt_item["name"]),
            price=Money(amount=Decimal(alt_item["price"]), currency="USD"),
            sku=str(uuid4())[:8],
        )

        print(f"Different ID equality: {product1 == product3}")  # False (different ID)

    # ========== AGGREGATE ROOTS DEMONSTRATION ==========

    def demonstrate_aggregate_roots(self) -> None:
        """Show aggregate root patterns."""
        print("\n=== Aggregate Roots ===")

        order_data = self._realistic["order"]
        order = Order(
            id=str(order_data["order_id"]),
            customer_id=str(order_data["customer_id"]),
            order_number=str(order_data["order_id"])[:12],
        )
        print(f"Order created: {order.order_number}")

        for item in order_data["items"]:
            product = Product(
                id=str(item["product_id"]),
                name=str(item["name"]),
                price=Money(amount=Decimal(item["price"]), currency="USD"),
                sku=str(item["product_id"])[:8],
                stock=max(item.get("quantity", 1) * 5, 5),
            )
            result = order.add_line(product, int(item.get("quantity", 1)))
            if result.is_success:
                print(f"✅ Added {item.get('quantity', 1)}x {product.name}")

        print(f"Order total: ${order.total_amount.amount}")

        result = order.submit()
        if result.is_success:
            print(f"✅ Order submitted, status: {order.status}")

        events = order.domain_events
        print(f"Aggregate events: {len(events)} events")
        for event in events:
            if isinstance(event, dict) and "event_name" in event:
                print(f"  - {event['event_name']}: {event['data']}")

    def demonstrate_business_invariants(self) -> None:
        """Show how aggregates maintain business invariants."""
        print("\n=== Business Invariants ===")

        users_list = cast("list", self._dataset["users"])
        user_data = cast("Flext.Types.Dict", users_list[0])
        user_reg2 = self._realistic["user_registration"]
        address_data2 = cast("Flext.Types.Dict", user_reg2["address"])

        customer = Customer(
            id=str(uuid4()),
            name=str(user_data["name"]),
            email=Email(address=str(user_data["email"])),
            shipping_address=Address(
                street=str(address_data2["street"]),
                city=str(address_data2["city"]),
                state=str(address_data2["state"]),
                zip_code=str(address_data2["zip"]),
            ),
            billing_address=Address(
                street=str(address_data2["street"]),
                city=str(address_data2["city"]),
                state=str(address_data2["state"]),
                zip_code=str(address_data2["zip"]),
            ),
            credit_limit=Money(amount=Decimal(5000), currency="USD"),
            current_balance=Money(amount=Decimal(1000), currency="USD"),
        )

        purchase_amount = Money(
            amount=Decimal(self._realistic["order"]["total"]),
            currency="USD",
        )
        can_purchase = customer.can_purchase(purchase_amount)
        if can_purchase.is_success:
            print(f"Can purchase ${purchase_amount.amount}: {can_purchase.unwrap()}")

        result = customer.make_purchase(purchase_amount)
        if result.is_success:
            print(
                f"✅ Purchase completed, new balance: ${customer.current_balance.amount}",
            )

        excessive = Money(amount=Decimal(2000), currency="USD")
        result = customer.make_purchase(excessive)
        if result.is_failure:
            print(f"✅ Invariant enforced: {result.error}")

    # ========== SERIALIZATION ==========

    def demonstrate_serialization(self) -> None:
        """Show model serialization patterns."""
        print("\n=== Model Serialization ===")

        order_data: RealisticOrderDict = self._realistic["order"]
        order = Order(
            id=str(order_data["order_id"]),
            customer_id=str(order_data["customer_id"]),
            order_number=str(order_data["order_id"])[:12],
            lines=[
                OrderLine(
                    product_id=str(item["product_id"]),
                    product_name=str(item["name"]),
                    quantity=int(item["quantity"]),
                    unit_price=Money(amount=Decimal(item["price"]), currency="USD"),
                )
                for item in order_data["items"]
            ],
            total_amount=Money(amount=Decimal(order_data["total"]), currency="USD"),
        )

        order_dict = order.model_dump()
        print(f"Serialized keys: {list(order_dict.keys())}")

        order_dict_minimal = order.model_dump(exclude={"lines", "domain_events"})
        print(f"Minimal serialization: {list(order_dict_minimal.keys())}")

        new_order = Order(**order_dict)
        print(f"✅ Deserialized order: {new_order.order_number}")

    # ========== REPOSITORY PATTERN ==========

    def demonstrate_repository_pattern(self) -> None:
        """Show repository pattern integration."""
        print("\n=== Repository Pattern ===")

        class OrderRepository:
            """Repository for Order aggregates."""

            def __init__(self) -> None:
                self._storage: dict[str, Order] = {}
                self._logger = Flext.Logger(__name__)

            def save(self, order: Order) -> Flext.Result[None]:
                """Save order aggregate."""
                self._storage[order.id] = order
                self._logger.info(f"Order saved: {order.order_number}")
                return Flext.Result[None].ok(None)

            def find_by_id(self, order_id: str) -> Flext.Result[Order]:
                """Find order by ID."""
                if order_id in self._storage:
                    return Flext.Result[Order].ok(self._storage[order_id])
                return Flext.Result[Order].fail(f"Order not found: {order_id}")

            def find_by_customer(self, customer_id: str) -> Flext.Result[list[Order]]:
                """Find orders by customer."""
                orders = [
                    order
                    for order in self._storage.values()
                    if order.customer_id == customer_id
                ]
                return Flext.Result[list[Order]].ok(orders)

        repo = OrderRepository()

        order_data = self._realistic["order"]
        order = Order(
            id=str(order_data["order_id"]),
            customer_id=str(order_data["customer_id"]),
            order_number=str(order_data["order_id"])[:12],
        )

        result = repo.save(order)
        if result.is_success:
            print("✅ Order saved to repository")

        find_result = repo.find_by_id(order.id)
        if find_result.is_success:
            retrieved = find_result.unwrap()
            print(f"✅ Order retrieved: {retrieved.order_number}")

    # ========== DEPRECATED PATTERNS ==========

    def demonstrate_deprecated_patterns(self) -> None:
        """Show deprecated patterns with warnings."""
        print("\n=== ⚠️ DEPRECATED PATTERNS ===")

        # OLD: Mutable models (DEPRECATED)
        warnings.warn(
            "Mutable domain models are DEPRECATED! Use Flext.Models.Value for immutability.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("❌ OLD WAY (mutable models):")
        print("class Price:")
        print("    def __init__(self, amount):")
        print("        self.amount = amount  # Mutable!")

        print("\n✅ CORRECT WAY (Flext.Models.Value):")
        print("class Price(Flext.Models.Value):")
        print("    amount: Decimal  # Immutable value object")

        # OLD: Anemic domain models (DEPRECATED)
        warnings.warn(
            "Anemic domain models are ANTI-PATTERN! Include business logic in models.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("\n❌ OLD WAY (anemic model):")
        print("class Order:")
        print("    # Just data, no behavior")
        print("    id: str")
        print("    total: float")

        print("\n✅ CORRECT WAY (rich domain model):")
        print("class Order(Flext.Models.AggregateRoot):")
        print("    def submit(self) -> Flext.Result[None]:")
        print("        # Business logic in the model")

        # OLD: Direct mutation (DEPRECATED)
        warnings.warn(
            "Direct state mutation is DEPRECATED! Use methods that return Flext.Result.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("\n❌ OLD WAY (direct mutation):")
        print("order.status = 'SHIPPED'  # Direct mutation!")

        print("\n✅ CORRECT WAY (method with result):")
        print("result = order.ship()  # Returns FlextResult")
        print("if result.is_success:")
        print("    # Handle success")


def main() -> None:
    """Main entry point demonstrating all FlextModels capabilities."""
    service = ComprehensiveModelsService()

    print("=" * 60)
    print("FLEXTMODELS COMPLETE API DEMONSTRATION")
    print("Foundation for Domain-Driven Design in FLEXT Ecosystem")
    print("=" * 60)

    # Core patterns
    service.demonstrate_value_objects()
    service.demonstrate_value_object_operations()
    service.demonstrate_entities()
    service.demonstrate_entity_identity()

    # Advanced patterns
    service.demonstrate_aggregate_roots()
    service.demonstrate_business_invariants()

    # Professional patterns
    service.demonstrate_serialization()
    service.demonstrate_repository_pattern()

    # Deprecation warnings
    service.demonstrate_deprecated_patterns()

    print("\n" + "=" * 60)
    print("✅ ALL FlextModels methods demonstrated!")
    print("🎯 Next: See 04_config_basics.py for FlextConfig patterns")
    print("=" * 60)


if __name__ == "__main__":
    main()
