#!/usr/bin/env python3
"""Modern FLEXT patterns with boilerplate elimination.

Demonstrates railway-oriented programming, semantic types,
zero-configuration patterns, and type safety compliance.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations
from flext_core import FlextTypes
import sys
from decimal import Decimal
from enum import StrEnum
from typing import Self, NotRequired, TypedDict, Unpack

from pydantic import ConfigDict, Field
from pydantic_settings import SettingsConfigDict

from flext_core import FlextConfig, FlextModels, FlextResult


MIN_AGE = 18
MAX_AGE = 120
MIN_PRICE = Decimal("0.01")
MAX_PRICE = Decimal("100000.00")
CURRENCY_CODE_LENGTH = 3


class OrderStatus(StrEnum):
    """Order status enumeration."""

    PENDING = "pending"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class PaymentStatus(StrEnum):
    """Payment status enumeration."""

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"


class Age(FlextModels.Value):
    """Age value object with validation."""

    value: int = Field(..., ge=MIN_AGE, le=MAX_AGE)

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate age business rules."""
        if self.value < MIN_AGE:
            return FlextResult[None].fail(f"Age must be at least {MIN_AGE}")
        if self.value > MAX_AGE:
            return FlextResult[None].fail(f"Age cannot exceed {MAX_AGE}")
        return FlextResult[None].ok(None)

    @classmethod
    def create(cls, age: int) -> FlextResult[Self]:
        """Create age with validation."""
        try:
            instance = cls(value=age)
            return FlextResult[Self].ok(instance)
        except Exception as e:
            return FlextResult[Self].fail(f"Invalid age: {e}")


class Money(FlextModels.Value):
    """Money value object with currency support."""

    amount: Decimal = Field(..., ge=MIN_PRICE, le=MAX_PRICE)
    currency: str = Field(default="USD", min_length=3, max_length=3)

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate money business rules."""
        if self.amount < MIN_PRICE:
            return FlextResult[None].fail(f"Amount must be at least {MIN_PRICE}")
        if self.amount > MAX_PRICE:
            return FlextResult[None].fail(f"Amount cannot exceed {MAX_PRICE}")
        if len(self.currency) != CURRENCY_CODE_LENGTH:
            return FlextResult[None].fail("Currency must be 3 characters")
        return FlextResult[None].ok(None)

    @classmethod
    def create(
        cls,
        amount: Decimal | float | str,
        currency: str = "USD",
    ) -> FlextResult[Self]:
        """Create money with validation."""
        try:
            decimal_amount = Decimal(str(amount))
            instance = cls(amount=decimal_amount, currency=currency.upper())
            return FlextResult[Self].ok(instance)
        except Exception as e:
            return FlextResult[Self].fail(f"Invalid money: {e}")

    def add(self, other: Money) -> FlextResult[Money]:
        """Add two money values."""
        if self.currency != other.currency:
            return FlextResult[Money].fail(
                f"Currency mismatch: {self.currency} vs {other.currency}",
            )
        return Money.create(self.amount + other.amount, self.currency)

    def multiply(self, factor: float) -> FlextResult[Money]:
        """Multiply money by a factor."""
        return Money.create(self.amount * Decimal(str(factor)), self.currency)


class EmailAddress(FlextModels.Value):
    """Email address value object with validation."""

    address: str = Field(..., min_length=5, max_length=254)

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate email business rules."""
        if "@" not in self.address:
            return FlextResult[None].fail("Email must contain @")
        if "." not in self.address.split("@")[1]:
            return FlextResult[None].fail("Email domain must contain .")
        return FlextResult[None].ok(None)

    @classmethod
    def create(cls, email: str) -> FlextResult[Self]:
        """Create email with validation."""
        try:
            if "@" not in email or "." not in email.rsplit("@", maxsplit=1)[-1]:
                return FlextResult[Self].fail("Invalid email format")
            instance = cls(address=email.lower())
            return FlextResult[Self].ok(instance)
        except Exception as e:
            return FlextResult[Self].fail(f"Invalid email: {e}")


class ECommerceConfig(FlextConfig):
    """E-commerce system configuration."""

    # Service configuration
    service_name: str = "modern-ecommerce"
    service_version: str = "1.0.0"
    debug_mode: bool = False

    # Business rules
    max_order_items: int = Field(default=50, ge=1, le=100)
    min_order_value: Decimal = Field(default=MIN_PRICE)
    max_order_value: Decimal = Field(default=MAX_PRICE)

    # Payment settings
    payment_timeout_seconds: int = Field(default=300, ge=60, le=3600)
    enable_refunds: bool = True

    # Notification settings
    enable_email_notifications: bool = True
    enable_sms_notifications: bool = False

    model_config = SettingsConfigDict(
        extra="allow",
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
    )

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate configuration business rules."""
        if self.min_order_value >= self.max_order_value:
            return FlextResult[None].fail(
                "min_order_value must be less than max_order_value",
            )
        if self.max_order_items <= 0:
            return FlextResult[None].fail("max_order_items must be positive")
        return FlextResult[None].ok(None)


class User(FlextModels.Entity):
    """User entity with comprehensive validation."""

    email: EmailAddress
    name: str = Field(..., min_length=1, max_length=100)
    age: Age
    is_active: bool = True

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate user business rules."""
        email_result = self.email.validate_business_rules()
        if email_result.failure:
            return email_result

        age_result = self.age.validate_business_rules()
        if age_result.failure:
            return age_result

        return FlextResult[None].ok(None)

    @classmethod
    def create(
        cls,
        email: str,
        name: str,
        age: int,
        user_id: str | None = None,
    ) -> FlextResult[User]:
        """Create user with validation."""
        try:
            # Validate email
            email_result = EmailAddress.create(email)
            if not email_result.success:
                return FlextResult[User].fail(f"Invalid email: {email_result.error}")

            # Validate age
            age_result = Age.create(age)
            if not age_result.success:
                return FlextResult[User].fail(f"Invalid age: {age_result.error}")

            # Generate ID if not provided
            if user_id is None:
                user_id = f"USER-{hash(email) % 100000:05d}"

            user = cls(
                id=user_id,
                email=email_result.value,
                name=name.strip(),
                age=age_result.value,
            )

            return FlextResult[User].ok(user)

        except Exception as e:
            return FlextResult[User].fail(f"Failed to create user: {e}")


class Product(FlextModels.Entity):
    """Product entity with pricing and inventory."""

    name: str = Field(..., min_length=1, max_length=200)
    price: Money
    stock_quantity: int = Field(default=0, ge=0)
    is_available: bool = True

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate product business rules."""
        price_result = self.price.validate_business_rules()
        if price_result.failure:
            return price_result

        if self.stock_quantity < 0:
            return FlextResult[None].fail("Stock quantity cannot be negative")

        return FlextResult[None].ok(None)

    # Python 3.13 TypedDict for parameter reduction - ELIMINATES BOILERPLATE
    class CreateProductParams(TypedDict):
        """TypedDict for Product.create parameters."""

        name: str
        price: Decimal | float | str
        stock_quantity: NotRequired[int]
        currency: NotRequired[str]
        product_id: NotRequired[str | None]

    @classmethod
    def create(cls, **params: Unpack[CreateProductParams]) -> FlextResult[Product]:
        """Create product with validation using Python 3.13 TypedDict."""
        # Extract parameters with defaults
        name = params["name"]
        price = params["price"]
        stock_quantity = params.get("stock_quantity", 0)
        currency = params.get("currency", "USD")
        product_id = params.get("product_id")

        try:
            # Validate price
            price_result = Money.create(price, currency)
            if not price_result.success:
                return FlextResult[Product].fail(f"Invalid price: {price_result.error}")

            # Generate ID if not provided
            if product_id is None:
                product_id = f"PROD-{hash(name) % 100000:05d}"

            product = cls(
                id=product_id,
                name=name.strip(),
                price=price_result.value,
                stock_quantity=max(0, stock_quantity),
                is_available=stock_quantity > 0,
            )

            return FlextResult[Product].ok(product)

        except Exception as e:
            return FlextResult[Product].fail(f"Failed to create product: {e}")

    def update_stock(self, quantity: int) -> FlextResult[None]:
        """Update stock quantity."""
        if self.stock_quantity + quantity < 0:
            return FlextResult[None].fail("Insufficient stock")
        self.stock_quantity += quantity
        self.is_available = self.stock_quantity > 0
        return FlextResult[None].ok(None)


class OrderItem(FlextModels.Value):
    """Order item with product and quantity."""

    product: Product
    quantity: int = Field(..., ge=1)
    total_price: Money

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate order item business rules."""
        if self.quantity <= 0:
            return FlextResult[None].fail("Quantity must be positive")
        product_result = self.product.validate_business_rules()
        if product_result.failure:
            return product_result
        total_result = self.total_price.validate_business_rules()
        if total_result.failure:
            return total_result
        return FlextResult[None].ok(None)

    @classmethod
    def create(cls, product: Product, quantity: int) -> FlextResult[OrderItem]:
        """Create order item with validation."""
        if quantity <= 0:
            return FlextResult[OrderItem].fail("Quantity must be positive")
        if quantity > product.stock_quantity:
            return FlextResult[OrderItem].fail("Insufficient stock")

        # Calculate total price
        price_result = product.price.multiply(quantity)
        if not price_result.success:
            return FlextResult[OrderItem].fail(
                f"Price calculation failed: {price_result.error}",
            )

        item = cls(
            product=product,
            quantity=quantity,
            total_price=price_result.value,
        )
        return FlextResult[OrderItem].ok(item)


class Order(FlextModels.AggregateRoot):
    """Order aggregate with comprehensive business logic."""

    user: User
    items: list[OrderItem]
    status: OrderStatus = OrderStatus.PENDING
    payment_status: PaymentStatus = PaymentStatus.PENDING
    total_amount: Money

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate order business rules."""
        if not self.items:
            return FlextResult[None].fail("Order must have at least one item")

        user_result = self.user.validate_business_rules()
        if user_result.failure:
            return user_result

        for item in self.items:
            item_result = item.validate_business_rules()
            if item_result.failure:
                return item_result

        total_result = self.total_amount.validate_business_rules()
        if total_result.failure:
            return total_result

        return FlextResult[None].ok(None)

    def model_post_init(self, __context: FlextTypes.Core.Dict | None = None, /) -> None:
        """Calculate total after initialization."""
        if not hasattr(self, "total_amount") and self.items:
            self.total_amount = self._calculate_total()

    def _calculate_total(self) -> Money:
        """Calculate order total."""
        if not self.items:
            return Money.create(0).value

        # Get currency from first item
        currency = self.items[0].total_price.currency
        total = Decimal("0.00")

        for item in self.items:
            total += item.total_price.amount

        return Money.create(total, currency).value

    @classmethod
    def create(
        cls,
        user: User,
        items_data: list[FlextTypes.Core.Dict],
        config: ECommerceConfig,
        order_id: str | None = None,
    ) -> FlextResult[Order]:
        """Create order with validation."""
        try:
            # Validate basic constraints
            validation_result = cls._validate_order_constraints(items_data, config)
            if not validation_result.success:
                return FlextResult[Order].fail(
                    validation_result.error or "Validation failed",
                )

            # Create order items
            items_result = cls._create_order_items(items_data)
            if not items_result.success:
                return FlextResult[Order].fail(
                    items_result.error or "Failed to create items",
                )

            items = items_result.value

            # Generate order
            return cls._build_order(user, items, config, order_id)

        except Exception as e:
            return FlextResult[Order].fail(f"Failed to create order: {e}")

    @classmethod
    def _validate_order_constraints(
        cls,
        items_data: list[FlextTypes.Core.Dict],
        config: ECommerceConfig,
    ) -> FlextResult[None]:
        """Validate order constraints."""
        if len(items_data) > config.max_order_items:
            return FlextResult[None].fail(
                f"Too many items: {len(items_data)} > {config.max_order_items}",
            )

        if not items_data:
            return FlextResult[None].fail("Order must have at least one item")

        return FlextResult[None].ok(None)

    @classmethod
    def _create_order_items(
        cls,
        items_data: list[FlextTypes.Core.Dict],
    ) -> FlextResult[list[OrderItem]]:
        """Create order items from data."""
        items: list[OrderItem] = []

        for item_data in items_data:
            if "product" not in item_data or "quantity" not in item_data:
                return FlextResult[list[OrderItem]].fail("Invalid item data")

            product = item_data["product"]
            if not isinstance(product, Product):
                return FlextResult[list[OrderItem]].fail("Invalid product")

            quantity = int(str(item_data["quantity"]))
            item_result = OrderItem.create(product, quantity)
            if not item_result.success:
                return FlextResult[list[OrderItem]].fail(
                    f"Invalid item: {item_result.error}",
                )

            items.append(item_result.value)

        return FlextResult[list[OrderItem]].ok(items)

    @classmethod
    def _build_order(
        cls,
        user: User,
        items: list[OrderItem],
        config: ECommerceConfig,
        order_id: str | None,
    ) -> FlextResult[Order]:
        """Build final order with validation."""
        # Generate ID
        if order_id is None:
            order_id = f"ORDER-{user.id}-{len(items):03d}"

        # Calculate and validate total
        total_result = cls._calculate_and_validate_total(items, config)
        if not total_result.success:
            return FlextResult[Order].fail(
                total_result.error or "Total validation failed",
            )

        total_money = total_result.value

        # Create final order
        order = cls(
            id=order_id,
            user=user,
            items=items,
            total_amount=total_money,
        )

        return FlextResult[Order].ok(order)

    @classmethod
    def _calculate_and_validate_total(
        cls,
        items: list[OrderItem],
        config: ECommerceConfig,
    ) -> FlextResult[Money]:
        """Calculate and validate order total."""
        if not items:
            return FlextResult[Money].fail("No items to calculate total")

        currency = items[0].total_price.currency
        total = Decimal("0.00")
        for item in items:
            total += item.total_price.amount

        total_money_result = Money.create(total, currency)
        if not total_money_result.success:
            return FlextResult[Money].fail(
                f"Total calculation failed: {total_money_result.error}",
            )

        # Validate total amount
        if total < config.min_order_value:
            return FlextResult[Money].fail(f"Order value too low: {total}")

        if total > config.max_order_value:
            return FlextResult[Money].fail(f"Order value too high: {total}")

        return total_money_result

    def confirm(self) -> FlextResult[None]:
        """Confirm order and update stock."""
        if self.status != OrderStatus.PENDING:
            return FlextResult[None].fail(
                f"Cannot confirm order in {self.status} status",
            )

        # Update stock for all items
        for item in self.items:
            stock_result = item.product.update_stock(-item.quantity)
            if not stock_result.success:
                return FlextResult[None].fail(
                    f"Stock update failed: {stock_result.error}",
                )

        self.status = OrderStatus.CONFIRMED
        return FlextResult[None].ok(None)


class PaymentService:
    """Payment processing service."""

    def __init__(self, config: ECommerceConfig) -> None:
        """Initialize payment service."""
        self.config = config

    def process_payment(
        self,
        order: Order,
        payment_method: str = "credit_card",
    ) -> FlextResult[str]:
        """Process payment for order."""
        try:
            if order.payment_status != PaymentStatus.PENDING:
                return FlextResult[str].fail("Payment already processed")

            # Simulate payment processing
            payment_id = f"PAY-{order.id}-{hash(payment_method) % 10000:04d}"

            # In real implementation, this would call external payment gateway
            print(f"Processing payment: {payment_id} for ${order.total_amount.amount}")

            order.payment_status = PaymentStatus.COMPLETED
            return FlextResult[str].ok(payment_id)

        except Exception as e:
            return FlextResult[str].fail(f"Payment processing failed: {e}")


class OrderService:
    """Order management service."""

    def __init__(
        self,
        config: ECommerceConfig,
        payment_service: PaymentService,
    ) -> None:
        """Initialize order service."""
        self.config = config
        self.payment_service = payment_service

    def create_and_process_order(
        self,
        user: User,
        items_data: list[FlextTypes.Core.Dict],
    ) -> FlextResult[Order]:
        """Create and process complete order."""
        try:
            # Create order
            order_result = Order.create(user, items_data, self.config)
            if not order_result.success:
                return FlextResult[Order].fail(
                    f"Order creation failed: {order_result.error}",
                )

            order = order_result.value

            # Confirm order
            confirm_result = order.confirm()
            if not confirm_result.success:
                return FlextResult[Order].fail(
                    f"Order confirmation failed: {confirm_result.error}",
                )

            # Process payment
            payment_result = self.payment_service.process_payment(order)
            if not payment_result.success:
                return FlextResult[Order].fail(
                    f"Payment failed: {payment_result.error}",
                )

            order.status = OrderStatus.PROCESSING
            return FlextResult[Order].ok(order)

        except Exception as e:
            return FlextResult[Order].fail(f"Order processing failed: {e}")


def demonstrate_user_creation() -> FlextResult[User]:
    """Demonstrate user creation with validation."""
    print("Creating user with validation...")

    user_result = User.create(
        email="john.doe@example.com",
        name="John Doe",
        age=30,
    )

    if user_result.success:
        user = user_result.value
        print(f"✅ User created: {user.name} ({user.email.address})")
        return user_result

    print(f"❌ User creation failed: {user_result.error}")
    return user_result


def demonstrate_product_creation() -> FlextResult[list[Product]]:
    """Demonstrate product creation and management."""
    print("Creating products...")

    products: list[Product] = []
    product_data = [
        {"name": "Laptop Pro", "price": "1299.99", "stock": 10},
        {"name": "Wireless Mouse", "price": "29.99", "stock": 50},
        {"name": "USB Cable", "price": "9.99", "stock": 100},
    ]

    for data in product_data:
        product_result = Product.create(
            name=str(data["name"]),
            price=str(data["price"]),
            stock_quantity=int(str(data["stock"])),
        )

        if product_result.success:
            product = product_result.value
            products.append(product)
            print(f"✅ Product created: {product.name} - ${product.price.amount}")
        else:
            print(f"❌ Product creation failed: {product_result.error}")
            return FlextResult[list[Product]].fail(
                f"Product creation failed: {product_result.error}",
            )

    return FlextResult[list[Product]].ok(products)


def demonstrate_order_processing(
    user: User,
    products: list[Product],
) -> FlextResult[Order]:
    """Demonstrate complete order processing."""
    print("Processing order...")

    # Create configuration
    config = ECommerceConfig()

    # Create services
    payment_service = PaymentService(config)
    order_service = OrderService(config, payment_service)

    # Create order items
    items_data: list[FlextTypes.Core.Dict] = [
        {"product": products[0], "quantity": 1},  # Laptop
        {"product": products[1], "quantity": 2},  # Mouse x2
    ]

    # Process order
    order_result = order_service.create_and_process_order(user, items_data)

    if order_result.success:
        order = order_result.value
        print(f"✅ Order processed: {order.id} - ${order.total_amount.amount}")
        print(f"   Status: {order.status}")
        print(f"   Payment: {order.payment_status}")
    else:
        print(f"❌ Order processing failed: {order_result.error}")

    return order_result


def main() -> int:
    """Main demonstration function."""
    print("🚀 Modern FLEXT Patterns Showcase")
    print("=" * 50)

    try:
        # Demonstrate user creation
        user_result = demonstrate_user_creation()
        if not user_result.success:
            return 1

        user = user_result.value

        # Demonstrate product creation
        products_result = demonstrate_product_creation()
        if not products_result.success:
            return 1

        products = products_result.value

        # Demonstrate order processing
        order_result = demonstrate_order_processing(user, products)
        if not order_result.success:
            return 1

        print("\n✅ All demonstrations completed successfully!")
        return 0

    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
