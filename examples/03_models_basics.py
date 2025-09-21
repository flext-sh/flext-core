#!/usr/bin/env python3
"""03 - FlextModels Fundamentals: Complete Domain-Driven Design Patterns.

This example demonstrates the COMPLETE FlextModels API - the foundation
for domain modeling across the entire FLEXT ecosystem. FlextModels provides
DDD patterns with Value Objects, Entities, and Aggregate Roots.

Key Concepts Demonstrated:
- Value Objects: FlextModels.Value for immutable value types
- Entities: FlextModels.Entity with identity and lifecycle
- Aggregate Roots: FlextModels.AggregateRoot for consistency boundaries
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
from uuid import uuid4

from pydantic import Field

from flext_core import (
    FlextContainer,
    FlextDomainService,
    FlextLogger,
    FlextModels,
    FlextResult,
)

# ========== VALUE OBJECTS ==========


class Email(FlextModels.Value):
    """Email value object - immutable and compared by value."""

    address: str

    def validate_email(self) -> FlextResult[None]:
        """Business validation for email format."""
        if "@" not in self.address:
            return FlextResult[None].fail("Invalid email format")
        if self.address.lower() != self.address:
            return FlextResult[None].fail("Email must be lowercase")
        return FlextResult[None].ok(None)

    @classmethod
    def create(cls, address: str) -> FlextResult[Email]:
        """Factory method with validation."""
        email = cls(address=address.lower().strip())
        validation = email.validate_email()
        if validation.is_failure:
            return FlextResult[Email].fail(f"Email creation failed: {validation.error}")
        return FlextResult[Email].ok(email)


class Money(FlextModels.Value):
    """Money value object with currency and amount."""

    amount: Decimal
    currency: str = "USD"

    def add(self, other: Money) -> FlextResult[Money]:
        """Add money with currency check."""
        if self.currency != other.currency:
            return FlextResult[Money].fail("Currency mismatch")
        return FlextResult[Money].ok(
            Money(amount=self.amount + other.amount, currency=self.currency)
        )

    def subtract(self, other: Money) -> FlextResult[Money]:
        """Subtract money with currency check."""
        if self.currency != other.currency:
            return FlextResult[Money].fail("Currency mismatch")
        if self.amount < other.amount:
            return FlextResult[Money].fail("Insufficient funds")
        return FlextResult[Money].ok(
            Money(amount=self.amount - other.amount, currency=self.currency)
        )


class Address(FlextModels.Value):
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


class Product(FlextModels.Entity):
    """Product entity with identity and business logic."""

    name: str
    price: Money
    sku: str
    stock: int = 0
    is_active: bool = True

    def adjust_price(self, new_price: Money) -> FlextResult[None]:
        """Adjust product price with validation."""
        if new_price.amount <= Decimal(0):
            return FlextResult[None].fail("Price must be positive")

        self.price = new_price

        # Note: Entity doesn't have domain events, only AggregateRoot does

        return FlextResult[None].ok(None)

    def add_stock(self, quantity: int) -> FlextResult[None]:
        """Add stock with validation."""
        if quantity <= 0:
            return FlextResult[None].fail("Quantity must be positive")

        self.stock += quantity
        # Note: Entity doesn't have domain events
        return FlextResult[None].ok(None)

    def remove_stock(self, quantity: int) -> FlextResult[None]:
        """Remove stock with availability check."""
        if quantity <= 0:
            return FlextResult[None].fail("Quantity must be positive")
        if quantity > self.stock:
            return FlextResult[None].fail("Insufficient stock")

        self.stock -= quantity
        # Note: Entity doesn't have domain events
        return FlextResult[None].ok(None)


class Customer(FlextModels.Entity):
    """Customer entity with complex business rules."""

    name: str
    email: Email
    shipping_address: Address
    billing_address: Address
    credit_limit: Money
    current_balance: Money
    is_vip: bool = False
    created_at: datetime = datetime.now(UTC)

    def __init__(self, **data: object) -> None:
        """Initialize with timestamp."""
        if "created_at" not in data:
            data["created_at"] = datetime.now(UTC)

        # Initialize parent class with type-safe data
        # Convert object values to proper types for Pydantic
        typed_data: dict[str, object] = dict(data.items())

        super().__init__(**typed_data)  # type: ignore[arg-type]

    def can_purchase(self, amount: Money) -> FlextResult[bool]:
        """Check if customer can make purchase."""
        # Check credit limit
        available_credit_result = self.credit_limit.subtract(self.current_balance)
        if available_credit_result.is_failure:
            return FlextResult[bool].ok(False)

        available_credit = available_credit_result.unwrap()
        if amount.amount > available_credit.amount:
            return FlextResult[bool].fail("Exceeds credit limit")

        return FlextResult[bool].ok(True)

    def make_purchase(self, amount: Money) -> FlextResult[None]:
        """Process a purchase."""
        can_purchase = self.can_purchase(amount)
        if can_purchase.is_failure or not can_purchase.unwrap():
            return FlextResult[None].fail("Cannot complete purchase")

        # Update balance
        new_balance_result = self.current_balance.add(amount)
        if new_balance_result.is_failure:
            return FlextResult[None].fail(new_balance_result.error or "Unknown error")

        self.current_balance = new_balance_result.unwrap()

        # Note: Entity doesn't have domain events

        return FlextResult[None].ok(None)

    def upgrade_to_vip(self) -> FlextResult[None]:
        """Upgrade customer to VIP status."""
        if self.is_vip:
            return FlextResult[None].fail("Already VIP")

        self.is_vip = True
        # VIP gets higher credit limit
        self.credit_limit = Money(
            amount=self.credit_limit.amount * Decimal(2),
            currency=self.credit_limit.currency,
        )

        # Note: Entity doesn't have domain events

        return FlextResult[None].ok(None)


# ========== AGGREGATE ROOTS ==========


class OrderLine(FlextModels.Value):
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


class Order(FlextModels.AggregateRoot):
    """Order aggregate root - maintains consistency boundary."""

    customer_id: str
    order_number: str
    lines: list[OrderLine] = Field(default_factory=list)
    total_amount: Money = Money(amount=Decimal(0), currency="USD")
    status: str = "DRAFT"
    created_at: datetime = datetime.now(UTC)
    shipped_at: datetime | None = None

    def __init__(self, **data: object) -> None:
        """Initialize with defaults."""
        if "lines" not in data:
            data["lines"] = []
        if "created_at" not in data:
            data["created_at"] = datetime.now(UTC)
        if "total_amount" not in data:
            data["total_amount"] = Money(amount=Decimal(0), currency="USD")

        # Initialize parent class with type-safe data
        # Convert object values to proper types for Pydantic
        typed_data: dict[str, object] = dict(data.items())

        super().__init__(**typed_data)  # type: ignore[arg-type]

    def add_line(self, product: Product, quantity: int) -> FlextResult[None]:
        """Add order line with stock validation."""
        # Check if order is modifiable
        if self.status not in {"DRAFT", "PENDING"}:
            return FlextResult[None].fail(
                f"Cannot modify order in {self.status} status"
            )

        # Check stock availability
        if product.stock < quantity:
            return FlextResult[None].fail(f"Insufficient stock for {product.name}")

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

        return FlextResult[None].ok(None)

    def remove_line(self, product_id: str) -> FlextResult[None]:
        """Remove order line."""
        if self.status not in {"DRAFT", "PENDING"}:
            return FlextResult[None].fail(
                f"Cannot modify order in {self.status} status"
            )

        # Find and remove line
        original_count = len(self.lines)
        self.lines = [line for line in self.lines if line.product_id != product_id]

        if len(self.lines) == original_count:
            return FlextResult[None].fail("Product not found in order")

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

        return FlextResult[None].ok(None)

    def _recalculate_total(self) -> None:
        """Recalculate order total (internal consistency)."""
        total = Decimal(0)
        for line in self.lines:
            total += line.calculate_total().amount

        self.total_amount = Money(amount=total, currency="USD")

    def submit(self) -> FlextResult[None]:
        """Submit order for processing."""
        if self.status != "DRAFT":
            return FlextResult[None].fail("Only draft orders can be submitted")

        if not self.lines:
            return FlextResult[None].fail("Cannot submit empty order")

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

        return FlextResult[None].ok(None)

    def ship(self) -> FlextResult[None]:
        """Mark order as shipped."""
        if self.status != "PENDING":
            return FlextResult[None].fail("Only pending orders can be shipped")

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

        return FlextResult[None].ok(None)

    def cancel(self, reason: str) -> FlextResult[None]:
        """Cancel order with reason."""
        if self.status in {"SHIPPED", "DELIVERED", "CANCELLED"}:
            return FlextResult[None].fail(
                f"Cannot cancel order in {self.status} status"
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

        return FlextResult[None].ok(None)


# ========== COMPREHENSIVE MODELS SERVICE ==========


class ComprehensiveModelsService(FlextDomainService[Order]):
    """Service demonstrating ALL FlextModels patterns and methods."""

    def __init__(self) -> None:
        """Initialize with dependencies."""
        super().__init__()
        self._container = FlextContainer.get_global()
        self._logger = FlextLogger(__name__)

    def execute(self) -> FlextResult[Order]:
        """Execute method required by FlextDomainService."""
        # This is a demonstration service, execute creates a sample order
        order = Order(
            id=str(uuid4()),
            customer_id=str(uuid4()),
            order_number=f"ORD-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}",
        )
        return FlextResult[Order].ok(order)

    # ========== VALUE OBJECTS DEMONSTRATION ==========

    def demonstrate_value_objects(self) -> None:
        """Show value object patterns."""
        print("\n=== Value Objects ===")

        # Create value objects
        email_result = Email.create("john@example.com")
        if email_result.is_success:
            email = email_result.unwrap()
            print(f"✅ Email created: {email.address}")

        money = Money(amount=Decimal("100.50"), currency="USD")
        print(f"Money: ${money.amount} {money.currency}")

        address = Address(
            street="123 Main St",
            city="Springfield",
            state="IL",
            zip_code="62701",
        )
        print(f"Address:\n{address.format_for_shipping()}")

        # Value object equality (by value, not identity)
        email1 = Email(address="test@example.com")
        email2 = Email(address="test@example.com")
        print(f"Value equality: {email1 == email2}")  # True
        print(f"Same instance: {email1 is email2}")  # False

    def demonstrate_value_object_operations(self) -> None:
        """Show value object business operations."""
        print("\n=== Value Object Operations ===")

        # Money operations
        money1 = Money(amount=Decimal(100), currency="USD")
        money2 = Money(amount=Decimal(50), currency="USD")

        # Addition
        result = money1.add(money2)
        if result.is_success:
            total = result.unwrap()
            print(f"$100 + $50 = ${total.amount}")

        # Subtraction
        result = money1.subtract(money2)
        if result.is_success:
            difference = result.unwrap()
            print(f"$100 - $50 = ${difference.amount}")

        # Currency mismatch
        euro = Money(amount=Decimal(50), currency="EUR")
        result = money1.add(euro)
        if result.is_failure:
            print(f"✅ Currency check: {result.error}")

    # ========== ENTITIES DEMONSTRATION ==========

    def demonstrate_entities(self) -> None:
        """Show entity patterns."""
        print("\n=== Entities ===")

        # Create entities with identity
        product = Product(
            id=str(uuid4()),
            name="Laptop",
            price=Money(amount=Decimal("999.99"), currency="USD"),
            sku="LAP-001",
            stock=10,
        )
        print(f"Product: {product.name} (ID: {product.id})")

        # Entity operations
        result = product.add_stock(5)
        if result.is_success:
            print(f"✅ Stock added, new stock: {product.stock}")

        # Price adjustment
        new_price = Money(amount=Decimal("899.99"), currency="USD")
        result = product.adjust_price(new_price)
        if result.is_success:
            print(f"✅ Price adjusted to ${new_price.amount}")

        # Note: Entity class doesn't have domain events
        # Only AggregateRoot has domain event support
        print("Note: Entity doesn't track domain events (use AggregateRoot)")

    def demonstrate_entity_identity(self) -> None:
        """Show entity identity vs value comparison."""
        print("\n=== Entity Identity ===")

        # Entities are compared by ID, not values
        product1 = Product(
            id=str(uuid4()),
            name="Same Product",
            price=Money(amount=Decimal(100), currency="USD"),
            sku="PROD-001",
        )

        product2 = Product(
            id=product1.id,  # Same ID
            name="Different Name",
            price=Money(amount=Decimal(200), currency="USD"),
            sku="PROD-002",
        )

        print(f"Same ID equality: {product1 == product2}")  # True (same ID)

        product3 = Product(
            id=str(uuid4()),  # Different ID
            name="Same Product",
            price=Money(amount=Decimal(100), currency="USD"),
            sku="PROD-001",
        )

        print(f"Different ID equality: {product1 == product3}")  # False (different ID)

    # ========== AGGREGATE ROOTS DEMONSTRATION ==========

    def demonstrate_aggregate_roots(self) -> None:
        """Show aggregate root patterns."""
        print("\n=== Aggregate Roots ===")

        # Create aggregate root
        order = Order(
            id=str(uuid4()),
            customer_id=str(uuid4()),
            order_number="ORD-2025-001",
        )
        print(f"Order created: {order.order_number}")

        # Create products for the order
        laptop = Product(
            id=str(uuid4()),
            name="Laptop",
            price=Money(amount=Decimal("999.99"), currency="USD"),
            sku="LAP-001",
            stock=5,
        )

        mouse = Product(
            id=str(uuid4()),
            name="Mouse",
            price=Money(amount=Decimal("29.99"), currency="USD"),
            sku="MOU-001",
            stock=20,
        )

        # Add lines to order (aggregate maintains consistency)
        result = order.add_line(laptop, 2)
        if result.is_success:
            print(f"✅ Added 2x {laptop.name}")

        result = order.add_line(mouse, 3)
        if result.is_success:
            print(f"✅ Added 3x {mouse.name}")

        print(f"Order total: ${order.total_amount.amount}")

        # Submit order (state transition)
        result = order.submit()
        if result.is_success:
            print(f"✅ Order submitted, status: {order.status}")

        # Aggregate events (AggregateRoot has domain event support!)
        events = order.domain_events  # Access the events list directly
        print(f"Aggregate events: {len(events)} events")
        for event in events:
            if isinstance(event, dict) and "event_name" in event:
                print(f"  - {event['event_name']}: {event['data']}")

    def demonstrate_business_invariants(self) -> None:
        """Show how aggregates maintain business invariants."""
        print("\n=== Business Invariants ===")

        # Customer with credit limit
        customer = Customer(
            id=str(uuid4()),
            name="John Doe",
            email=Email(address="john@example.com"),
            shipping_address=Address(
                street="123 Main St",
                city="Springfield",
                state="IL",
                zip_code="62701",
            ),
            billing_address=Address(
                street="123 Main St",
                city="Springfield",
                state="IL",
                zip_code="62701",
            ),
            credit_limit=Money(amount=Decimal(5000), currency="USD"),
            current_balance=Money(amount=Decimal(1000), currency="USD"),
        )

        # Check purchase ability
        purchase_amount = Money(amount=Decimal(3000), currency="USD")
        can_purchase = customer.can_purchase(purchase_amount)
        if can_purchase.is_success:
            print(f"Can purchase ${purchase_amount.amount}: {can_purchase.unwrap()}")

        # Attempt purchase
        result = customer.make_purchase(purchase_amount)
        if result.is_success:
            print(
                f"✅ Purchase completed, new balance: ${customer.current_balance.amount}"
            )

        # Try to exceed limit
        excessive = Money(amount=Decimal(2000), currency="USD")
        result = customer.make_purchase(excessive)
        if result.is_failure:
            print(f"✅ Invariant enforced: {result.error}")

    # ========== SERIALIZATION ==========

    def demonstrate_serialization(self) -> None:
        """Show model serialization patterns."""
        print("\n=== Model Serialization ===")

        # Create complex model
        order = Order(
            id=str(uuid4()),
            customer_id=str(uuid4()),
            order_number="ORD-2025-002",
            lines=[
                OrderLine(
                    product_id=str(uuid4()),
                    product_name="Product 1",
                    quantity=2,
                    unit_price=Money(amount=Decimal(50), currency="USD"),
                ),
            ],
            total_amount=Money(amount=Decimal(100), currency="USD"),
        )

        # Serialize to dict
        order_dict = order.model_dump()
        print(f"Serialized keys: {list(order_dict.keys())}")

        # Serialize with exclusion
        order_dict_minimal = order.model_dump(exclude={"lines", "domain_events"})
        print(f"Minimal serialization: {list(order_dict_minimal.keys())}")

        # Create from dict (deserialization)
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
                self._logger = FlextLogger(__name__)

            def save(self, order: Order) -> FlextResult[None]:
                """Save order aggregate."""
                # In real implementation, this would persist to database
                self._storage[order.id] = order
                self._logger.info(f"Order saved: {order.order_number}")
                return FlextResult[None].ok(None)

            def find_by_id(self, order_id: str) -> FlextResult[Order]:
                """Find order by ID."""
                if order_id in self._storage:
                    return FlextResult[Order].ok(self._storage[order_id])
                return FlextResult[Order].fail(f"Order not found: {order_id}")

            def find_by_customer(self, customer_id: str) -> FlextResult[list[Order]]:
                """Find orders by customer."""
                orders = [
                    order
                    for order in self._storage.values()
                    if order.customer_id == customer_id
                ]
                return FlextResult[list[Order]].ok(orders)

        # Use repository
        repo = OrderRepository()

        order = Order(
            id=str(uuid4()),
            customer_id=str(uuid4()),
            order_number="ORD-2025-003",
        )

        # Save
        result = repo.save(order)
        if result.is_success:
            print("✅ Order saved to repository")

        # Retrieve
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
            "Mutable domain models are DEPRECATED! Use FlextModels.Value for immutability.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("❌ OLD WAY (mutable models):")
        print("class Price:")
        print("    def __init__(self, amount):")
        print("        self.amount = amount  # Mutable!")

        print("\n✅ CORRECT WAY (FlextModels.Value):")
        print("class Price(FlextModels.Value):")
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
        print("class Order(FlextModels.AggregateRoot):")
        print("    def submit(self) -> FlextResult[None]:")
        print("        # Business logic in the model")

        # OLD: Direct mutation (DEPRECATED)
        warnings.warn(
            "Direct state mutation is DEPRECATED! Use methods that return FlextResult.",
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
