#!/usr/bin/env python3
"""Modern FLEXT patterns with boilerplate elimination.

Demonstrates railway-oriented programming, semantic types,
zero-configuration patterns, and type safety compliance.
Building an e-commerce order processing system with user management,
inventory tracking, payment processing, and notification systems.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Self, cast, override

from pydantic_settings import SettingsConfigDict
from shared_domain import (
    Age,
    EmailAddress as Email,
    Money,
    User as SharedUser,
)

from flext_core import (
    FlextContainer,
    FlextEntity,
    FlextEntityId,
    FlextResult,
    FlextSettings,
    FlextUtilities,
    FlextValue,
    get_flext_container,
)

# =============================================================================
# BUSINESS CONSTANTS
# =============================================================================

# Product validation
MIN_PRODUCT_NAME_LENGTH = 2

# Order validation
MAX_ORDER_ITEM_QUANTITY = 100
MAX_ORDER_ITEMS = 50

# =============================================================================
# CONFIGURATION: Environment-Aware Settings (Zero Boilerplate)
# =============================================================================


class AppConfig(FlextSettings):
    """Application configuration with automatic environment loading."""

    # Database settings - automatically loaded from env vars
    database_url: str = "postgresql://localhost/ecommerce"
    redis_url: str = "redis://localhost:6379"

    # Payment settings
    payment_api_key: str = "demo_payment_key_12345"
    payment_endpoint: str = "https://api.payments.com"

    # Business rules
    max_order_value: int = 100000  # cents
    min_order_value: int = 100  # cents

    model_config = SettingsConfigDict(env_prefix="ECOMMERCE_")


# =============================================================================
# VALUE OBJECTS: Immutable Domain Concepts (Minimal Boilerplate)
# =============================================================================


# Using Money, Email, and other value objects from shared_domain
# This eliminates ~50 lines of duplicate code!


# =============================================================================
# DOMAIN ENTITIES: Rich Business Logic (Framework Handles Infrastructure)
# =============================================================================


class OrderStatus(StrEnum):
    """Order status enumeration."""

    DRAFT = "draft"
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class Customer(SharedUser):
    """Customer entity using shared domain - zero boilerplate."""

    is_premium: bool = False

    def promote_to_premium(self) -> FlextResult[Self]:
        """Promote customer to premium status."""
        result = self.copy_with(is_premium=True)
        # Modern pattern: Check success and add domain event
        if result.success:
            customer = result.value
            customer.add_domain_event(
                "CustomerPromoted",
                {
                    "customer_id": self.id,
                    "timestamp": FlextUtilities.generate_iso_timestamp(),
                },
            )
            return FlextResult[Self].ok(customer)
        return result


class Product(FlextEntity):
    """Product entity with inventory management."""

    name: str
    price: Money
    stock_quantity: int
    category: str

    def is_available(self, quantity: int = 1) -> bool:
        """Check if product is available in requested quantity."""
        return self.stock_quantity >= quantity

    def reserve_stock(self, quantity: int) -> FlextResult[None]:
        """Reserve stock for an order."""
        if not self.is_available(quantity):
            return FlextResult[None].fail(
                f"Insufficient stock: {self.stock_quantity} available, {quantity} requested",
            )

        # Update stock (in real implementation would persist to database)
        self.stock_quantity -= quantity
        return FlextResult[None].ok(None)

    @override
    def validate_business_rules(self) -> FlextResult[None]:
        """Validate product business rules."""
        if not self.name or len(self.name.strip()) < MIN_PRODUCT_NAME_LENGTH:
            return FlextResult[None].fail("Product name must be at least 2 characters")

        if self.stock_quantity < 0:
            return FlextResult[None].fail("Stock quantity cannot be negative")

        if not self.category:
            return FlextResult[None].fail("Product category is required")

        # Validate price
        price_validation = self.price.validate_business_rules()
        if price_validation.is_failure:
            return FlextResult[None].fail(
                f"Price validation failed: {price_validation.error}",
            )

        return FlextResult[None].ok(None)


class OrderItem(FlextValue):
    """Order item with quantity and pricing."""

    product_id: str
    quantity: int
    unit_price: Money

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate order item business rules."""
        if not self.product_id:
            return FlextResult[None].fail("Product ID cannot be empty")

        if self.quantity <= 0:
            return FlextResult[None].fail("Quantity must be positive")

        if self.quantity > MAX_ORDER_ITEM_QUANTITY:
            return FlextResult[None].fail("Quantity cannot exceed 100")

        # Validate unit price
        price_validation = self.unit_price.validate_business_rules()
        if price_validation.is_failure:
            return FlextResult[None].fail(
                f"Unit price validation failed: {price_validation.error}",
            )

        return FlextResult[None].ok(None)

    def total_price(self) -> Money:
        """Calculate total price for this item."""
        result = self.unit_price.multiply(Decimal(str(self.quantity)))
        # Modern pattern: Use .value with fallback for cleaner error handling
        return result.unwrap_or(
            Money(amount=Decimal(0), currency=self.unit_price.currency)
        )


class Order(FlextEntity):
    """Order entity with complete business logic."""

    customer_id: str
    items: list[OrderItem]
    status: OrderStatus = OrderStatus.DRAFT
    total: Money = Money(amount=Decimal(0), currency="USD")

    def add_item(self, product: Product, quantity: int) -> FlextResult[Self]:
        """Add item to order with validation."""
        if not product.is_available(quantity):
            return FlextResult[Self].fail(
                f"Product {product.name} not available in quantity {quantity}",
            )

        item = OrderItem(
            product_id=str(product.id),
            quantity=quantity,
            unit_price=product.price,
        )

        updated_items = [*self.items, item]
        new_total = self._calculate_total(updated_items)

        # Modern pattern: Use map for transformation, preserving error structure
        return self.copy_with(items=updated_items, total=new_total)

    def confirm(self) -> FlextResult[Self]:
        """Confirm the order with business rule validation."""
        config = cast("AppConfig", get_flext_container().get("config").value)

        if self.total.amount < config.min_order_value:
            return FlextResult[Self].fail(f"Order below minimum value: {self.total}")

        if self.total.amount > config.max_order_value:
            return FlextResult[Self].fail(f"Order exceeds maximum value: {self.total}")

        # Modern pattern: Use map for cleaner transformation
        return self.copy_with(status=OrderStatus.CONFIRMED)

    def _calculate_total(self, items: list[OrderItem]) -> Money:
        """Calculate total from all items."""
        total_amount = sum(item.total_price().amount for item in items)
        return Money(amount=Decimal(str(total_amount)), currency="USD")

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate order business rules."""
        if not self.customer_id:
            return FlextResult[None].fail("Customer ID is required")

        if not self.items:
            return FlextResult[None].fail("Order must have at least one item")

        if len(self.items) > MAX_ORDER_ITEMS:
            return FlextResult[None].fail("Order cannot have more than 50 items")

        # Validate all items
        for i, item in enumerate(self.items):
            item_validation = item.validate_business_rules()
            if item_validation.is_failure:
                return FlextResult[None].fail(
                    f"Item {i + 1} validation failed: {item_validation.error}",
                )

        # Validate total is consistent
        calculated_total = self._calculate_total(self.items)
        if abs(self.total.amount - calculated_total.amount) > Decimal("0.01"):
            return FlextResult[None].fail(
                "Order total is inconsistent with item totals"
            )

        return FlextResult[None].ok(None)


# =============================================================================
# SERVICES: Business Logic Orchestration (Clean & Testable)
# =============================================================================


class InventoryService:
    """Inventory management service."""

    def reserve_items(self, items: list[OrderItem]) -> FlextResult[None]:
        """Reserve inventory for order items."""
        container = get_flext_container()

        for item in items:
            product_result = container.get(f"product_{item.product_id}")
            if not product_result.success:
                return FlextResult[None].fail(f"Product not found: {item.product_id}")

            product = cast("Product", product_result.value)
            reserve_result = product.reserve_stock(item.quantity)
            if not reserve_result.success:
                return reserve_result

        return FlextResult[None].ok(None)


class PaymentService:
    """Payment processing service."""

    def charge(self, customer_id: str, amount: Money) -> FlextResult[str]:
        """Process payment and return transaction ID."""
        # Simulate payment processing
        if amount.amount <= 0:
            return FlextResult[str].fail("Invalid payment amount")

        # In real implementation, call payment API
        transaction_id = f"txn_{customer_id}_{amount.amount}"
        return FlextResult[str].ok(transaction_id)


class NotificationService:
    """Notification service for customer communications."""

    def send_order_confirmation(
        self,
        customer: Customer,  # noqa: ARG002
        order: Order,  # noqa: ARG002
    ) -> FlextResult[None]:
        """Send order confirmation email."""
        # Simulate email sending (send_email function not available in shared_domain)
        return FlextResult[None].ok(None)


# =============================================================================
# FACTORIES: Zero-Boilerplate Object Creation
# =============================================================================


def create_customer(name: str, email_address: str) -> FlextResult[Customer]:
    """Create customer with validation."""
    try:
        # Create email and validate
        email = Email(email=email_address)
        validation_result = email.validate_business_rules()
        if validation_result.is_failure:
            return FlextResult[Customer].fail(
                validation_result.error or "Email validation failed"
            )

        # Create customer
        customer = Customer(
            id=FlextEntityId(FlextUtilities.generate_entity_id()),
            name=name,
            email_address=email,
            age=Age(value=25),  # Default age
        )
        return FlextResult[Customer].ok(customer)
    except Exception as e:
        return FlextResult[Customer].fail(f"Customer creation failed: {e}")


def create_product(
    name: str,
    price_cents: int,
    stock: int,
    category: str,
) -> FlextResult[Product]:
    """Create product with business rules."""
    if price_cents <= 0:
        return FlextResult[Product].fail("Price must be positive")

    if stock < 0:
        return FlextResult[Product].fail("Stock cannot be negative")

    price = Money(amount=Decimal(str(price_cents)), currency="USD")

    product = Product(
        id=FlextEntityId(FlextUtilities.generate_entity_id()),
        name=name,
        price=price,
        stock_quantity=stock,
        category=category,
    )
    return FlextResult[Product].ok(product)


# =============================================================================
# APPLICATION SERVICE: Complete Business Flow (Railway-Oriented)
# =============================================================================


class OrderProcessingService:
    """Order processing service with railway-oriented programming."""

    def __init__(self) -> None:
        self.inventory = InventoryService()
        self.payment = PaymentService()
        self.notifications = NotificationService()

    def process_order(
        self,
        customer_id: str,
        order_data: dict[str, object],
    ) -> FlextResult[Order]:
        """Process complete order with automatic error handling.

        Traditional approach would require 50+ lines of try/catch boilerplate.
        Railway-oriented approach: 8 lines, automatic error propagation!
        """
        return (
            self._create_order(customer_id, order_data)
            .flat_map(self._validate_inventory)
            .flat_map(self._confirm_order)
            .flat_map(self._process_payment)
            .flat_map(self._reserve_inventory)
            .flat_map(self._send_notifications)
            .tap(self._log_success)
        )

    def _create_order(
        self,
        customer_id: str,
        order_data: dict[str, object],
    ) -> FlextResult[Order]:
        """Create order from raw data."""

        def _start_empty_order() -> Order:
            return Order(
                id=FlextEntityId(FlextUtilities.generate_entity_id()),
                customer_id=customer_id,
                items=[],
            )

        def _ensure_items(
            data: dict[str, object],
        ) -> FlextResult[list[dict[str, object]]]:
            items_value = data.get("items", [])
            if not isinstance(items_value, list):
                return FlextResult[list[dict[str, object]]].fail("Items must be a list")
            normalized: list[dict[str, object]] = []
            for entry in items_value:
                if not isinstance(entry, dict):
                    return FlextResult[list[dict[str, object]]].fail(
                        "Item data must be a dictionary"
                    )
                normalized.append(entry)
            return FlextResult[list[dict[str, object]]].ok(normalized)

        def _parse_quantity(raw_quantity: object) -> FlextResult[int]:
            if not isinstance(raw_quantity, (int, str)):
                return FlextResult[int].fail("Quantity must be a number")
            try:
                return FlextResult[int].ok(int(str(raw_quantity)))
            except (ValueError, TypeError):
                return FlextResult[int].fail("Invalid quantity format")

        def _get_product_for_item(
            item: dict[str, object],
        ) -> FlextResult[tuple[object, int]]:
            product_id = item.get("product_id")

            if not isinstance(product_id, str):
                return FlextResult[tuple[object, int]].fail(
                    "Product ID must be a string"
                )
            product_result = self._get_product(product_id)

            if product_result.is_failure or product_result.value is None:
                return FlextResult[tuple[object, int]].fail(
                    product_result.error or "Product not found"
                )
            quantity_result = _parse_quantity(item.get("quantity", 1))

            if quantity_result.is_failure or quantity_result.value is None:
                return FlextResult[tuple[object, int]].fail(
                    quantity_result.error or "Invalid quantity"
                )
            return FlextResult[tuple[object, int]].ok((
                product_result.value,
                quantity_result.value,
            ))

        items_result = _ensure_items(order_data)
        if items_result.is_failure or items_result.value is None:
            return FlextResult[Order].fail(items_result.error or "Invalid items")

        order = _start_empty_order()
        for item_data in items_result.value:
            pair_result = _get_product_for_item(item_data)

            if pair_result.is_failure or pair_result.value is None:
                return FlextResult[Order].fail(pair_result.error or "Invalid item")
            product, quantity_int = cast("tuple[Product, int]", pair_result.value)
            order_result = order.add_item(product, quantity_int)

            if not order_result.success:
                return FlextResult[Order].fail(
                    order_result.error or "Failed to add item"
                )

            order = order_result.value
        return FlextResult[Order].ok(order)

    def _validate_inventory(self, order: Order) -> FlextResult[Order]:
        """Validate inventory availability."""
        validation_result = self.inventory.reserve_items(order.items)
        return validation_result.map(lambda _: order)

    def _confirm_order(self, order: Order) -> FlextResult[Order]:
        """Confirm order with business rules."""
        return order.confirm()

    def _process_payment(self, order: Order) -> FlextResult[Order]:
        """Process payment for order."""
        payment_result = self.payment.charge(order.customer_id, order.total)
        return payment_result.map(lambda _: order)

    def _reserve_inventory(self, order: Order) -> FlextResult[Order]:
        """Reserve inventory for confirmed order."""
        reserve_result = self.inventory.reserve_items(order.items)
        return reserve_result.map(lambda _: order)

    def _send_notifications(self, order: Order) -> FlextResult[Order]:
        """Send order confirmation notifications."""
        container = get_flext_container()
        customer_result = container.get(f"customer_{order.customer_id}")

        if not customer_result.success:
            return FlextResult[Order].fail("Customer not found for notification")

        if customer_result.value is None:
            return FlextResult[Order].fail("Customer data is None")

        customer = cast("Customer", customer_result.value)
        notification_result = self.notifications.send_order_confirmation(
            customer,
            order,
        )
        return notification_result.map(lambda _: order)

    def _get_product(self, product_id: str) -> FlextResult[Product]:
        """Get product by ID."""
        container = get_flext_container()
        result = container.get(f"product_{product_id}")
        if result.is_failure:
            return FlextResult[Product].fail(result.error or "Product not found")

        if result.value is None:
            return FlextResult[Product].fail("Product data is None")

        return FlextResult[Product].ok(cast("Product", result.value))

    def _log_success(self, order: Order) -> None:
        """Log successful order processing."""


# =============================================================================
# DEMONSTRATION: Real-World Usage
# =============================================================================


def _print_showcase_header() -> None:
    pass


def _setup_environment() -> FlextContainer:
    config = AppConfig()
    container = get_flext_container()
    container.register("config", config)
    return container


def _create_and_register_customer(container: FlextContainer) -> Customer | None:
    customer_result: FlextResult[Customer] = create_customer(
        "John Doe", "john@example.com"
    )
    if not customer_result.success:
        raise ValueError(customer_result.error)
    customer = customer_result.value
    container.register(f"customer_{customer.id}", customer)
    return customer


def _create_and_register_products(container: FlextContainer) -> None:
    products_data = [
        ("Laptop", 99900, 10, "Electronics", "product_laptop"),
        ("Mouse", 2500, 50, "Electronics", "product_mouse"),
        ("Keyboard", 7500, 25, "Electronics", "product_keyboard"),
    ]
    for name, price, stock, category, key in products_data:
        product_result = create_product(name, price, stock, category)
        if product_result.success:
            product = product_result.value
            container.register(key, product)


def _select_first_available_product_id(
    container: FlextContainer,
    keys: list[str],
) -> str | None:
    for key in keys:
        result = container.get(key)
        if result.success:
            product = cast("Product", result.value)
            return str(product.id)
    return None


def _process_order_and_print(
    _: FlextContainer, customer: Customer, product_id: str
) -> None:
    order_service = OrderProcessingService()
    order_data: dict[str, object] = {
        "items": [{"product_id": product_id, "quantity": 1}],
    }
    order_result = order_service.process_order(str(customer.id), order_data)
    if order_result.success:
        pass


def _type_system_demo(customer: Customer) -> None:
    def validator(c: Customer) -> bool:
        return c.is_premium

    def transformer(m: Money) -> str:
        return str(m)

    # Use the customer parameter
    validator(customer)


def _print_benefits() -> None:
    """Print benefits of modern patterns."""


def demonstrate_modern_patterns() -> None:
    """Demonstrate the power of modern FLEXT patterns."""
    _print_showcase_header()
    container: FlextContainer = _setup_environment()
    _create_and_register_products(container)
    customer = _create_and_register_customer(container)
    if customer is None:
        return
    product_keys = ["product_laptop", "product_mouse", "product_keyboard"]
    product_id = _select_first_available_product_id(container, product_keys)
    if product_id is None:
        return
    _process_order_and_print(container, customer, product_id)
    _type_system_demo(customer)
    _print_benefits()


def demonstrate_boilerplate_comparison() -> float:
    """Show before/after comparison of boilerplate reduction."""
    traditional_lines = 150  # Estimated lines for traditional approach
    modern_lines = 25  # Actual lines in modern approach
    return ((traditional_lines - modern_lines) / traditional_lines) * 100


if __name__ == "__main__":
    demonstrate_modern_patterns()
    demonstrate_boilerplate_comparison()
