#!/usr/bin/env python3
"""FLEXT Core Entity and Value Object DDD Patterns - Working Example.

This example demonstrates Domain-Driven Design patterns using FLEXT Core:
- Entity base class with identity and events
- Value objects for data modeling
- Domain event handling
- Factory patterns for object creation

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import Field

from flext_core import (
    FlextConstants,
    FlextDomainService,
    FlextModels,
    FlextResult,
    FlextUtilities,
)


class DDDConstants(FlextConstants):
    """Domain-driven design constants."""

    MIN_PRICE = Decimal("0.01")
    MAX_DISCOUNT = Decimal(100)
    COUNTRY_CODE_LENGTH = 2


class Money(FlextModels.Value):
    """Value object for monetary amounts."""

    amount: Decimal = Field(..., description="Monetary amount")
    currency: str = Field(..., min_length=3, max_length=3, description="Currency code")

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate money business rules."""
        if self.amount < Decimal(0):
            return FlextResult[None].fail("Amount cannot be negative")
        return FlextResult[None].ok(None)

    def add(self, other: Money) -> Money:
        """Add two money amounts."""
        if self.currency != other.currency:
            msg = "Cannot add different currencies"
            raise ValueError(msg)
        return Money(amount=self.amount + other.amount, currency=self.currency)

    def multiply(self, factor: Decimal) -> Money:
        """Multiply money by a factor."""
        return Money(amount=self.amount * factor, currency=self.currency)

    def __str__(self) -> str:
        """String representation of money amount."""
        return f"{self.amount} {self.currency}"


class Address(FlextModels.Value):
    """Value object for addresses."""

    street: str = Field(..., min_length=1, description="Street address")
    city: str = Field(..., min_length=1, description="City")
    postal_code: str = Field(..., min_length=5, description="Postal code")
    country: str = Field(..., min_length=2, description="Country code")

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate address business rules."""
        if len(self.country) != DDDConstants.COUNTRY_CODE_LENGTH:
            return FlextResult[None].fail(
                f"Country code must be {DDDConstants.COUNTRY_CODE_LENGTH} characters",
            )
        return FlextResult[None].ok(None)

    def __str__(self) -> str:
        """String representation of address."""
        return f"{self.street}, {self.city} {self.postal_code}, {self.country}"


class Product(FlextModels.Entity):
    """Product entity with business logic."""

    name: str = Field(..., min_length=1, description="Product name")
    price: Money = Field(..., description="Product price")
    active: bool = Field(default=True, description="Product active status")

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate product business rules."""
        if self.price.amount <= Decimal(0):
            return FlextResult[None].fail("Product price must be positive")
        return FlextResult[None].ok(None)

    def activate(self) -> None:
        """Activate the product."""
        if not self.active:
            self.active = True
            self.add_domain_event(
                FlextModels.Event(
                    event_type="ProductActivated",
                    payload={
                        "product_id": self.id,
                        "product_name": self.name,
                    },
                    aggregate_id=self.id,
                ),
            )

    def update_price(self, new_price: Money) -> FlextResult[None]:
        """Update product price with validation."""
        if new_price.amount < DDDConstants.MIN_PRICE:
            return FlextResult[None].fail("Price too low")

        self.price = new_price

        # Domain event simulation (simplified for current API)
        print(f"Domain event: PriceChanged for product {self.id}")

        return FlextResult[None].ok(None)


class Customer(FlextModels.Entity):
    """Customer entity with address management."""

    name: str = Field(..., min_length=1, description="Customer name")
    email: str = Field(..., description="Customer email")
    address: Address = Field(..., description="Customer address")

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate customer business rules."""
        if "@" not in self.email:
            return FlextResult[None].fail("Invalid email address")
        return FlextResult[None].ok(None)

    def update_address(self, new_address: Address) -> None:
        """Update customer address."""
        old_address = self.address
        self.address = new_address

        if old_address != new_address:
            self.add_domain_event(
                FlextModels.Event(
                    event_type="AddressChanged",
                    payload={
                        "customer_id": self.id,
                        "old_address": str(old_address),
                        "new_address": str(new_address),
                    },
                    aggregate_id=self.id,
                ),
            )


class CartItem(FlextModels.Value):
    """Shopping cart item value object."""

    product_id: str = Field(..., description="Product identifier")
    product_name: str = Field(..., description="Product name")
    price: Money = Field(..., description="Unit price")
    quantity: int = Field(..., ge=1, description="Item quantity")

    def total_price(self) -> Money:
        """Calculate total price for this item."""
        return self.price.multiply(Decimal(str(self.quantity)))


class ShoppingCart(FlextModels.Entity):
    """Shopping cart entity with business rules."""

    customer_id: str = Field(..., description="Customer identifier")
    items: list[CartItem] = Field(default_factory=list, description="Cart items")
    discount_percent: Decimal = Field(
        default=Decimal(0),
        description="Discount percentage",
    )

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate shopping cart business rules."""
        if self.discount_percent < Decimal(0) or self.discount_percent > Decimal(100):
            return FlextResult[None].fail("Discount must be between 0 and 100")
        return FlextResult[None].ok(None)

    def add_item(self, product: Product, quantity: int = 1) -> FlextResult[None]:
        """Add item to cart."""
        if quantity < 1:
            return FlextResult[None].fail("Quantity must be positive")

        # Check if item already exists
        for existing_item in self.items:
            if existing_item.product_id == product.id:
                return FlextResult[None].fail("Item already in cart")

        cart_item = CartItem(
            product_id=product.id,
            product_name=product.name,
            price=product.price,
            quantity=quantity,
        )

        self.items.append(cart_item)

        # Domain event simulation (simplified for current API)
        print(f"Domain event: ItemAdded to cart {self.id}")

        return FlextResult[None].ok(None)

    def calculate_total(self) -> Money:
        """Calculate cart total with discount."""
        if not self.items:
            return Money(amount=Decimal(0), currency="USD")

        # Assume all items have same currency as first item
        currency = self.items[0].price.currency
        subtotal = sum((item.total_price().amount for item in self.items), Decimal(0))

        discount_amount = subtotal * (self.discount_percent / Decimal(100))
        total = subtotal - discount_amount

        return Money(amount=total, currency=currency)

    def apply_discount(self, discount_percent: Decimal) -> FlextResult[None]:
        """Apply discount to cart."""
        if (
            discount_percent < Decimal(0)
            or discount_percent > DDDConstants.MAX_DISCOUNT
        ):
            return FlextResult[None].fail("Invalid discount percentage")

        self.discount_percent = discount_percent

        self.add_domain_event(
            FlextModels.Event(
                event_type="DiscountApplied",
                payload={
                    "cart_id": self.id,
                    "discount_percent": float(discount_percent),
                },
                aggregate_id=self.id,
            ),
        )

        return FlextResult[None].ok(None)


class DDDDomainService(FlextDomainService[Product]):
    """Domain service for DDD object creation and management."""

    def __init__(self) -> None:
        """Initialize domain service."""
        super().__init__()

    def create_product(self, name: str, price: Money) -> FlextResult[Product]:
        """Create a new product with validation."""
        if not name.strip():
            return FlextResult[Product].fail("Product name cannot be empty")

        price_validation = price.validate_business_rules()
        if price_validation.is_failure:
            return FlextResult[Product].fail(f"Invalid price: {price_validation.error}")

        product = Product(
            id=FlextUtilities.Generators.generate_id()[:8],
            name=name,
            price=price,
        )

        validation_result = product.validate_business_rules()
        if validation_result.is_failure:
            return FlextResult[Product].fail(
                validation_result.error or "Product validation failed",
            )

        return FlextResult[Product].ok(product)

    def create_customer(
        self,
        name: str,
        email: str,
        address: Address,
    ) -> FlextResult[Customer]:
        """Create a new customer with validation."""
        if not name.strip():
            return FlextResult[Customer].fail("Customer name cannot be empty")

        address_validation = address.validate_business_rules()
        if address_validation.is_failure:
            return FlextResult[Customer].fail(
                f"Invalid address: {address_validation.error}",
            )

        customer = Customer(
            id=FlextUtilities.Generators.generate_id()[:8],
            name=name,
            email=email,
            address=address,
        )

        validation_result = customer.validate_business_rules()
        if validation_result.is_failure:
            return FlextResult[Customer].fail(
                validation_result.error or "Customer validation failed",
            )

        return FlextResult[Customer].ok(customer)

    def create_shopping_cart(self, customer_id: str) -> FlextResult[ShoppingCart]:
        """Create a new shopping cart with validation."""
        if not customer_id.strip():
            return FlextResult[ShoppingCart].fail("Customer ID cannot be empty")

        cart = ShoppingCart(
            id=FlextUtilities.Generators.generate_id()[:8],
            customer_id=customer_id,
        )

        validation_result = cart.validate_business_rules()
        if validation_result.is_failure:
            return FlextResult[ShoppingCart].fail(
                validation_result.error or "Cart validation failed",
            )

        return FlextResult[ShoppingCart].ok(cart)

    def execute(self) -> FlextResult[Product]:
        """Execute demo functionality - required by FlextDomainService."""
        demo_price = Money(amount=Decimal("99.99"), currency="USD")
        return self.create_product("Demo Product", demo_price)


def demonstrate_ddd_patterns() -> None:
    """Unified demonstration of DDD patterns with data-driven approach."""
    print("=== FLEXT Core DDD Patterns Showcase ===")

    service = DDDDomainService()

    # Value object demonstrations
    print("\n1. Value Objects:")
    money_examples = [
        (Decimal("19.99"), "USD"),
        (Decimal("5.00"), "USD"),
    ]

    money_objects = []
    for amount, currency in money_examples:
        money = Money(amount=amount, currency=currency)
        money_objects.append(money)
        print(f"   Money: {money}")

    if len(money_objects) >= 2:
        total = money_objects[0].add(money_objects[1])
        print(f"   Total: {total}")

    # Address value object
    demo_address = Address(
        street="123 Main St",
        city="New York",
        postal_code="10001",
        country="US",
    )
    print(f"   Address: {demo_address}")

    # Entity creation demonstrations
    print("\n2. Entities:")

    # Product creation
    laptop_price = Money(amount=Decimal("999.99"), currency="USD")
    product_result = service.create_product("Gaming Laptop", laptop_price)

    if product_result.is_success:
        product = product_result.unwrap()
        print(f"   Created product: {product.name} - {product.price}")

        # Price update
        new_price = Money(amount=Decimal("899.99"), currency="USD")
        update_result = product.update_price(new_price)
        if update_result.is_success:
            print(f"   Updated price to: {product.price}")

    # Customer creation
    customer_result = service.create_customer(
        "John Doe",
        "john@example.com",
        demo_address,
    )
    if customer_result.is_success:
        customer = customer_result.unwrap()
        print(f"   Created customer: {customer.name}")

    # Aggregate demonstrations
    print("\n3. Aggregates:")

    if customer_result.is_success:
        customer = customer_result.unwrap()
        cart_result = service.create_shopping_cart(customer.id)

        if cart_result.is_success and product_result.is_success:
            cart = cart_result.unwrap()
            product = product_result.unwrap()
            print(f"   Created cart: {cart.id}")

            # Add item and apply discount
            add_result = cart.add_item(product, 1)
            if add_result.is_success:
                print(f"   Added {product.name} to cart")

                discount_result = cart.apply_discount(Decimal(10))
                if discount_result.is_success:
                    total = cart.calculate_total()
                    print(f"   Cart total with discount: {total}")
                    print(f"   Domain events: {len(cart.domain_events)} events")


def main() -> None:
    """Advanced FLEXT Core DDD patterns demonstration."""
    print("🚀 Advanced FLEXT Core DDD Patterns Example")
    print("=" * 50)
    print("Architecture: FlextDomainService • FlextModels • FlextConstants")
    print()

    demonstrate_ddd_patterns()

    print("\n" + "=" * 50)
    print("✅ Advanced DDD patterns demonstrated successfully!")


if __name__ == "__main__":
    main()
