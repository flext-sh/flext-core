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

import json
import warnings
from datetime import UTC, datetime
from decimal import Decimal
from typing import cast
from uuid import uuid4

from pydantic import Field

from flext_core import (
    Flext,
    FlextConstants,
    FlextExceptions,
    FlextRuntime,
)

from .example_scenarios import ExampleScenarios, RealisticDataDict, RealisticOrderDict

# ========== VALUE OBJECTS ==========


class Email(Flext.Models.Value):
    """Email value object - immutable and compared by value."""

    address: str

    def validate_email(self) -> Flext.Result[bool]:
        """Business validation for email format."""
        if "@" not in self.address:
            return Flext.Result[bool].fail("Invalid email format")
        if self.address.lower() != self.address:
            return Flext.Result[bool].fail("Email must be lowercase")
        return Flext.Result[bool].ok(True)

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

    def adjust_price(self, new_price: Money) -> Flext.Result[bool]:
        """Adjust product price with validation."""
        if new_price.amount <= Decimal(0):
            return Flext.Result[bool].fail("Price must be positive")

        self.price = new_price

        # Note: Entity doesn't have domain events, only AggregateRoot does

        return Flext.Result[bool].ok(True)

    def add_stock(self, quantity: int) -> Flext.Result[bool]:
        """Add stock with validation."""
        if quantity <= 0:
            return Flext.Result[bool].fail("Quantity must be positive")

        self.stock += quantity
        # Note: Entity doesn't have domain events
        return Flext.Result[bool].ok(True)

    def remove_stock(self, quantity: int) -> Flext.Result[bool]:
        """Remove stock with availability check."""
        if quantity <= 0:
            return Flext.Result[bool].fail("Quantity must be positive")
        if quantity > self.stock:
            return Flext.Result[bool].fail("Insufficient stock")

        self.stock -= quantity
        # Note: Entity doesn't have domain events
        return Flext.Result[bool].ok(True)


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

    def make_purchase(self, amount: Money) -> Flext.Result[bool]:
        """Process a purchase."""
        can_purchase = self.can_purchase(amount)
        if can_purchase.is_failure or not can_purchase.unwrap():
            return Flext.Result[bool].fail("Cannot complete purchase")

        # Update balance
        new_balance_result = self.current_balance.add(amount)
        if new_balance_result.is_failure:
            return Flext.Result[bool].fail(new_balance_result.error or "Unknown error")

        self.current_balance = new_balance_result.unwrap()

        # Note: Entity doesn't have domain events

        return Flext.Result[bool].ok(True)

    def upgrade_to_vip(self) -> Flext.Result[bool]:
        """Upgrade customer to VIP status."""
        if self.is_vip:
            return Flext.Result[bool].fail("Already VIP")

        self.is_vip = True
        # VIP gets higher credit limit
        self.credit_limit = Money(
            amount=self.credit_limit.amount * Decimal(2),
            currency=self.credit_limit.currency,
        )

        # Note: Entity doesn't have domain events

        return Flext.Result[bool].ok(True)


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

    def add_line(self, product: Product, quantity: int) -> Flext.Result[bool]:
        """Add order line with stock validation."""
        # Check if order is modifiable
        if self.status not in {"DRAFT", "PENDING"}:
            return Flext.Result[bool].fail(
                f"Cannot modify order in {self.status} status",
            )

        # Check stock availability
        if product.stock < quantity:
            return Flext.Result[bool].fail(f"Insufficient stock for {product.name}")

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

        return Flext.Result[bool].ok(True)

    def remove_line(self, product_id: str) -> Flext.Result[bool]:
        """Remove order line."""
        if self.status not in {"DRAFT", "PENDING"}:
            return Flext.Result[bool].fail(
                f"Cannot modify order in {self.status} status",
            )

        # Find and remove line
        original_count = len(self.lines)
        self.lines = [line for line in self.lines if line.product_id != product_id]

        if len(self.lines) == original_count:
            return Flext.Result[bool].fail("Product not found in order")

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

        return Flext.Result[bool].ok(True)

    def _recalculate_total(self) -> None:
        """Recalculate order total (internal consistency)."""
        total = Decimal(0)
        for line in self.lines:
            total += line.calculate_total().amount

        self.total_amount = Money(amount=total, currency="USD")

    def submit(self) -> Flext.Result[bool]:
        """Submit order for processing."""
        if self.status != "DRAFT":
            return Flext.Result[bool].fail("Only draft orders can be submitted")

        if not self.lines:
            return Flext.Result[bool].fail("Cannot submit empty order")

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

        return Flext.Result[bool].ok(True)

    def ship(self) -> Flext.Result[bool]:
        """Mark order as shipped."""
        if self.status != "PENDING":
            return Flext.Result[bool].fail("Only pending orders can be shipped")

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

        return Flext.Result[bool].ok(True)

    def cancel(self, reason: str) -> Flext.Result[bool]:
        """Cancel order with reason."""
        if self.status in {"SHIPPED", "DELIVERED", "CANCELLED"}:
            return Flext.Result[bool].fail(
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

        return Flext.Result[bool].ok(True)


class ComprehensiveModelsService(Flext.Service[Order]):
    """Service demonstrating ALL FlextModels patterns with FlextMixins.Service infrastructure.

    This service inherits from Flext.Service to demonstrate:
    - Inherited container property (FlextContainer singleton)
    - Inherited logger property (FlextLogger with service context)
    - Inherited context property (FlextContext for request tracking)
    - Inherited config property (FlextConfig with settings)
    - Inherited metrics property (FlextMetrics for observability)

    The focus is on demonstrating DDD patterns (Value Objects, Entities,
    Aggregate Roots) while leveraging complete FlextMixins.Service infrastructure.
    """

    def __init__(self) -> None:
        """Initialize with inherited FlextMixins.Service infrastructure.

        Note: No manual logger or container initialization needed!
        All infrastructure is inherited from Flext.Service base class:
        - self.logger: FlextLogger with service context
        - self.container: FlextContainer global singleton
        - self.context: FlextContext for request tracking
        - self.config: FlextConfig with application settings
        - self.metrics: FlextMetrics for observability
        """
        super().__init__()
        self._scenarios: type[ExampleScenarios] = ExampleScenarios
        self._dataset: Flext.Types.Dict = self._scenarios.dataset()
        self._realistic: RealisticDataDict = self._scenarios.realistic_data()
        self._validation: Flext.Types.Dict = self._scenarios.validation_data()

        # Demonstrate inherited logger (no manual instantiation needed!)
        self.logger.info(
            "ComprehensiveModelsService initialized with inherited infrastructure",
            extra={
                "dataset_keys": list(self._dataset.keys()),
                "realistic_data_keys": list(self._realistic.keys()),
                "service_type": "FlextModels DDD demonstration",
            },
        )

    def execute(self) -> Flext.Result[Order]:
        """Execute method required by FlextService.

        This method satisfies the FlextService abstract interface while
        demonstrating FlextModels DDD patterns. Uses inherited infrastructure:
        - self.logger for structured logging throughout execution
        - self.container for dependency resolution (if needed)
        - self.context for request tracking (if needed)

        Returns:
            FlextResult containing Order aggregate root entity

        """
        # This is a demonstration service, execute creates a sample order
        order = Order(
            id=str(uuid4()),
            customer_id=str(uuid4()),
            order_number=f"ORD-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}",
        )

        self.logger.info(
            "Sample order created for demonstration",
            extra={
                "order_id": order.id,
                "order_number": order.order_number,
            },
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

        address = Address(
            street="123 Main St",
            city="Springfield",
            state="IL",
            zip_code="62701",
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

        address = Address(
            street="123 Main St",
            city="Springfield",
            state="IL",
            zip_code="62701",
        )

        customer = Customer(
            id=str(uuid4()),
            name=str(user_data["name"]),
            email=Email(address=str(user_data["email"])),
            shipping_address=address,
            billing_address=address,
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

            def save(self, order: Order) -> Flext.Result[bool]:
                """Save order aggregate."""
                self._storage[order.id] = order
                return Flext.Result[bool].ok(True)

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

    # ========== FOUNDATION LAYER INTEGRATION ==========

    def demonstrate_flext_runtime_integration(self) -> None:
        """Show FlextRuntime (Layer 0.5) type guards with DDD patterns."""
        print("\n=== FlextRuntime Integration (Layer 0.5) ===")

        # Type guard validation for value objects
        email_str = "user@example.com"
        if FlextRuntime.is_valid_email(email_str):
            email = Email(address=email_str)
            print(f"✅ Valid email via FlextRuntime: {email.address}")

        # UUID validation for entity IDs
        entity_id = "550e8400-e29b-41d4-a716-446655440000"
        if FlextRuntime.is_valid_uuid(entity_id):
            product = Product(
                id=entity_id,
                name="Validated Product",
                price=Money(amount=Decimal("99.99"), currency="USD"),
                sku="VAL-PROD-001",
            )
            print(f"✅ Valid UUID for entity: {product.id[:8]}...")

        # JSON validation for serialized models
        order_json = json.dumps(
            {
                "id": str(uuid4()),
                "customer_id": str(uuid4()),
                "order_number": "ORD-001",
                "status": "DRAFT",
            },
        )
        if FlextRuntime.is_valid_json(order_json):
            print("✅ Valid JSON for model serialization")

        # Configuration defaults for domain limits
        max_stock = FlextRuntime.DEFAULT_BATCH_SIZE
        print(f"✅ Domain limit from FlextRuntime: max_stock={max_stock}")

    def demonstrate_flext_exceptions_integration(self) -> None:
        """Show FlextExceptions (Layer 2) with domain validation."""
        print("\n=== FlextExceptions Integration (Layer 2) ===")

        # ValidationError with value object validation
        try:
            invalid_email = "not-an-email"
            if not FlextRuntime.is_valid_email(invalid_email):
                error_msg = "Invalid email format for value object"
                raise FlextExceptions.ValidationError(
                    error_msg,
                    field="address",
                    value=invalid_email,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
        except FlextExceptions.ValidationError as e:
            print(f"✅ ValidationError: {e.error_code} - {e.message}")
            print(f"   Field: {e.field}, Value: {e.value}")

        # NotFoundError with entity repository
        try:
            missing_id = str(uuid4())
            error_msg = "Entity not found in repository"
            raise FlextExceptions.NotFoundError(
                error_msg,
                resource_type="Product",
                resource_id=missing_id,
            )
        except FlextExceptions.NotFoundError as e:
            print(f"✅ NotFoundError: {e.error_code}")
            print(f"   Resource: {e.resource_type}, ID: {e.resource_id[:8]}...")

        # ConflictError for business rule violations
        try:
            error_msg = "Cannot ship order without items"
            raise FlextExceptions.ConflictError(
                error_msg,
                conflict_reason="Order must have at least one line item",
            )
        except FlextExceptions.ConflictError as e:
            print(f"✅ ConflictError: {e.error_code}")
            print(f"   Reason: {e.conflict_reason}")

    # ========== DEPRECATED PATTERNS ==========

    # ========== NEW FLEXTRESULT METHODS (v0.9.9+) ==========

    def demonstrate_from_callable(self) -> None:
        """Show from_callable for safe value object/entity creation."""
        print("\n=== from_callable(): Safe Domain Model Creation ===")

        # Safe email creation with exception handling
        def risky_email_creation() -> Email:
            """Simulate risky email creation that might raise."""
            email_result = Email.create_email("invalid-email")
            if email_result.is_failure:
                msg = f"Invalid email: {email_result.error}"
                raise ValueError(msg)
            return email_result.unwrap()

        email_result = Flext.Result.from_callable(risky_email_creation)
        if email_result.is_failure:
            print(f"✅ Caught email validation error safely: {email_result.error}")

        # Safe entity creation
        def create_product() -> Product:
            """Create product with validation."""
            return Product(
                id=str(uuid4()),
                name="Test Product",
                price=Money(amount=Decimal("99.99"), currency="USD"),
                sku="SKU-001",
                stock=10,
            )

        product_result = Flext.Result.from_callable(create_product)
        if product_result.is_success:
            product = product_result.unwrap()
            print(f"✅ Product created safely: {product.name}")

    def demonstrate_flow_through(self) -> None:
        """Show pipeline composition for multi-step entity operations."""
        print("\n=== flow_through(): Domain Model Validation Pipeline ===")

        def create_customer(data: dict[str, object]) -> Flext.Result[Customer]:
            """Step 1: Create customer."""
            email_result = Email.create_email(str(data.get("email", "")))
            if email_result.is_failure:
                return Flext.Result[Customer].fail(f"Email error: {email_result.error}")

            email = email_result.unwrap()

            address = Address(
                street="123 Main St",
                city="Springfield",
                state="IL",
                zip_code="62701",
            )

            customer = Customer(
                id=str(uuid4()),
                name=str(data.get("name", "")),
                email=email,
                shipping_address=address,
                billing_address=address,
                credit_limit=Money(amount=Decimal(5000), currency="USD"),
                current_balance=Money(amount=Decimal(0), currency="USD"),
            )
            return Flext.Result[Customer].ok(customer)

        def validate_credit(customer: Customer) -> Flext.Result[Customer]:
            """Step 2: Validate credit limit."""
            if customer.credit_limit.amount <= Decimal(0):
                return Flext.Result[Customer].fail("Invalid credit limit")
            return Flext.Result[Customer].ok(customer)

        def assign_vip_status(customer: Customer) -> Flext.Result[Customer]:
            """Step 3: Check VIP eligibility."""
            if customer.credit_limit.amount >= Decimal(10000):
                upgrade_result = customer.upgrade_to_vip()
                if upgrade_result.is_success:
                    return Flext.Result[Customer].ok(customer)
            return Flext.Result[Customer].ok(customer)

        # Pipeline: create → validate → assign VIP
        validation_data = self._validation
        sample_email = cast("Flext.Types.List", validation_data["valid_emails"])[0]
        users_list = cast("list", self._dataset["users"])
        user_data = cast("Flext.Types.Dict", users_list[0])

        result = (
            Flext.Result[dict[str, object]]
            .ok({"email": sample_email, "name": str(user_data["name"])})
            .flow_through(
                create_customer,
                validate_credit,
                assign_vip_status,
            )
        )

        if result.is_success:
            customer = result.unwrap()
            print(
                f"✅ Customer pipeline success: {customer.name}, VIP: {customer.is_vip}"
            )

    def demonstrate_lash(self) -> None:
        """Show error recovery in aggregate operations."""
        print("\n=== lash(): Aggregate Error Recovery ===")

        def try_create_order() -> Flext.Result[Order]:
            """Attempt to create order (might fail)."""
            return Flext.Result[Order].fail("Primary order creation failed")

        def recover_with_draft(error: str) -> Flext.Result[Order]:
            """Recover by creating draft order."""
            print(f"  Recovering from: {error}")
            order = Order(
                id=str(uuid4()),
                customer_id=str(uuid4()),
                order_number=f"DRAFT-{uuid4().hex[:8]}",
                status="DRAFT",
            )
            return Flext.Result[Order].ok(order)

        result = try_create_order().lash(recover_with_draft)
        if result.is_success:
            order = result.unwrap()
            print(f"✅ Recovered with draft order: {order.order_number}")

    def demonstrate_alt(self) -> None:
        """Show fallback pattern for repository operations."""
        print("\n=== alt(): Repository Fallback Pattern ===")

        # Primary repository (simulated failure)
        primary = Flext.Result[Order].fail("Primary database unavailable")

        # Fallback with cached order
        order_data = self._realistic["order"]
        fallback_order = Order(
            id=str(order_data["order_id"]),
            customer_id=str(order_data["customer_id"]),
            order_number=f"CACHE-{uuid4().hex[:8]}",
        )
        fallback = Flext.Result[Order].ok(fallback_order)

        result = primary.alt(fallback)
        if result.is_success:
            order = result.unwrap()
            print(f"✅ Got fallback order: {order.order_number}")

    def demonstrate_value_or_call(self) -> None:
        """Show lazy default evaluation for expensive domain object creation."""
        print("\n=== value_or_call(): Lazy Domain Object Creation ===")

        # Success case - no expensive creation needed
        order_data = self._realistic["order"]
        success_order = Order(
            id=str(order_data["order_id"]),
            customer_id=str(order_data["customer_id"]),
            order_number=str(order_data["order_id"])[:12],
        )
        success = Flext.Result[Order].ok(success_order)

        expensive_created = False

        def expensive_default() -> Order:
            """Expensive default order creation (only if needed)."""
            nonlocal expensive_created
            expensive_created = True
            print("  Creating expensive default order...")
            return Order(
                id=str(uuid4()),
                customer_id=str(uuid4()),
                order_number=f"DEFAULT-{uuid4().hex[:8]}",
            )

        # Success case - expensive_default NOT called
        order = success.value_or_call(expensive_default)
        print(
            f"✅ Success: {order.order_number}, expensive_created={expensive_created}"
        )

        # Failure case - expensive_default IS called
        expensive_created = False
        failure = Flext.Result[Order].fail("Order not found")
        order = failure.value_or_call(expensive_default)
        print(
            f"✅ Failure recovered: {order.order_number}, expensive_created={expensive_created}"
        )

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
        print("    def submit(self) -> Flext.Result[bool]:")
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

    # Foundation layer integration (NEW in Phase 1)
    service.demonstrate_flext_runtime_integration()
    service.demonstrate_flext_exceptions_integration()

    # New FlextResult methods (v0.9.9+)
    service.demonstrate_from_callable()
    service.demonstrate_flow_through()
    service.demonstrate_lash()
    service.demonstrate_alt()
    service.demonstrate_value_or_call()

    # Deprecation warnings
    service.demonstrate_deprecated_patterns()

    print("\n" + "=" * 60)
    print("✅ ALL FlextModels methods demonstrated!")
    print(
        "✨ Including new v0.9.9+ methods: from_callable, flow_through, lash, alt, value_or_call"
    )
    print(
        "🔧 Including foundation integration: FlextRuntime (Layer 0.5), FlextExceptions (Layer 2)"
    )
    print("🎯 Next: See 04_config_basics.py for FlextConfig patterns")
    print("=" * 60)


if __name__ == "__main__":
    main()
