#!/usr/bin/env python3
"""FlextCore Entity and Value Object DDD Patterns - Working Example.

This example demonstrates Domain-Driven Design patterns using FlextCore:
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

from flext_core import FlextModels, FlextResult

# Constants
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
        if len(self.country) != COUNTRY_CODE_LENGTH:
            return FlextResult[None].fail(
                f"Country code must be {COUNTRY_CODE_LENGTH} characters",
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
                {
                    "event_type": "ProductActivated",
                    "product_id": self.id,
                    "product_name": self.name,
                },
            )

    def update_price(self, new_price: Money) -> FlextResult[None]:
        """Update product price with validation."""
        if new_price.amount < MIN_PRICE:
            return FlextResult[None].fail("Price too low")

        old_price = self.price
        self.price = new_price

        self.add_domain_event(
            {
                "event_type": "PriceChanged",
                "product_id": self.id,
                "old_price": str(old_price),
                "new_price": str(new_price),
            },
        )

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
                {
                    "event_type": "AddressChanged",
                    "customer_id": self.id,
                    "old_address": str(old_address),
                    "new_address": str(new_address),
                },
            )


class CartItem(FlextModels.Value):
    """Shopping cart item value object."""

    product_id: str = Field(..., description="Product identifier")
    product_name: str = Field(..., description="Product name")
    price: Money = Field(..., description="Unit price")
    quantity: int = Field(..., ge=1, description="Item quantity")

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate cart item business rules."""
        if self.quantity <= 0:
            return FlextResult[None].fail("Quantity must be positive")
        return FlextResult[None].ok(None)

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

        self.add_domain_event(
            {
                "event_type": "ItemAdded",
                "cart_id": self.id,
                "product_id": product.id,
                "quantity": quantity,
            },
        )

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
        if discount_percent < Decimal(0) or discount_percent > MAX_DISCOUNT:
            return FlextResult[None].fail("Invalid discount percentage")

        self.discount_percent = discount_percent

        self.add_domain_event(
            {
                "event_type": "DiscountApplied",
                "cart_id": self.id,
                "discount_percent": float(discount_percent),
            },
        )

        return FlextResult[None].ok(None)


class DomainObjectFactory:
    """Factory for creating domain objects."""

    @staticmethod
    def create_product(name: str, price: Money) -> FlextResult[Product]:
        """Create a new product."""
        try:
            product = Product(
                id=f"product_{name.lower().replace(' ', '_')}",
                name=name,
                price=price,
            )
            return FlextResult[Product].ok(product)
        except Exception as e:
            return FlextResult[Product].fail(f"Failed to create product: {e}")

    @staticmethod
    def create_customer(
        name: str,
        email: str,
        address: Address,
    ) -> FlextResult[Customer]:
        """Create a new customer."""
        try:
            customer = Customer(
                id=f"customer_{name.lower().replace(' ', '_')}",
                name=name,
                email=email,
                address=address,
            )
            return FlextResult[Customer].ok(customer)
        except Exception as e:
            return FlextResult[Customer].fail(f"Failed to create customer: {e}")

    @staticmethod
    def create_shopping_cart(customer_id: str) -> FlextResult[ShoppingCart]:
        """Create a new shopping cart."""
        try:
            cart = ShoppingCart(id=f"cart_{customer_id}", customer_id=customer_id)
            return FlextResult[ShoppingCart].ok(cart)
        except Exception as e:
            return FlextResult[ShoppingCart].fail(f"Failed to create cart: {e}")


def demonstrate_value_objects() -> None:
    """Demonstrate value object patterns."""
    print("=== Value Objects Demo ===")

    # Money value object
    price1 = Money(amount=Decimal("19.99"), currency="USD")
    price2 = Money(amount=Decimal("5.00"), currency="USD")

    total = price1.add(price2)
    print(f"Price 1: {price1}")
    print(f"Price 2: {price2}")
    print(f"Total: {total}")

    # Address value object
    address = Address(
        street="123 Main St",
        city="New York",
        postal_code="10001",
        country="US",
    )
    print(f"Address: {address}")


def demonstrate_entities() -> None:
    """Demonstrate entity patterns."""
    print("\n=== Entities Demo ===")

    # Create product
    price = Money(amount=Decimal("99.99"), currency="USD")
    product_result = DomainObjectFactory.create_product("Laptop", price)

    if product_result.success:
        product = product_result.value
        print(f"Created product: {product.name} - {product.price}")

        # Update price
        new_price = Money(amount=Decimal("89.99"), currency="USD")
        update_result = product.update_price(new_price)

        if update_result.success:
            print(f"Updated price to: {product.price}")

        # Check domain events
        print(f"Domain events: {len(product.domain_events)} events")

    # Create customer
    customer_address = Address(
        street="456 Oak Ave",
        city="Boston",
        postal_code="02101",
        country="US",
    )

    customer_result = DomainObjectFactory.create_customer(
        "John Doe",
        "john@example.com",
        customer_address,
    )

    if customer_result.success:
        customer = customer_result.value
        print(f"Created customer: {customer.name}")


def demonstrate_aggregates() -> None:
    """Demonstrate aggregate patterns."""
    print("\n=== Aggregates Demo ===")

    # Create shopping cart
    cart_result = DomainObjectFactory.create_shopping_cart("customer_john_doe")

    if cart_result.success:
        cart = cart_result.value
        print(f"Created cart: {cart.id}")

        # Create product
        laptop_price = Money(amount=Decimal("999.99"), currency="USD")
        laptop_result = DomainObjectFactory.create_product(
            "Gaming Laptop",
            laptop_price,
        )

        if laptop_result.success:
            laptop = laptop_result.value

            # Add item to cart
            add_result = cart.add_item(laptop, 1)

            if add_result.success:
                print(f"Added {laptop.name} to cart")

                # Apply discount
                discount_result = cart.apply_discount(Decimal(10))

                if discount_result.success:
                    total = cart.calculate_total()
                    print(f"Cart total with discount: {total}")
                    print(f"Domain events: {len(cart.domain_events)} events")


def main() -> None:
    """Run DDD patterns demonstration."""
    print("FlextCore DDD Patterns Example")
    print("=" * 40)

    demonstrate_value_objects()
    demonstrate_entities()
    demonstrate_aggregates()

    print("\n" + "=" * 40)
    print("✅ All DDD patterns demonstrated successfully!")


if __name__ == "__main__":
    main()
