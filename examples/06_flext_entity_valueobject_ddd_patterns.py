#!/usr/bin/env python3
"""06 - Domain-Driven Design: Entity & Value Object Patterns.

Shows how FlextEntity and FlextValue simplify DDD implementation.
Demonstrates entity lifecycle, domain events, and value object immutability.

Key Patterns:
• FlextEntity for objects with identity
• FlextValue for immutable domain concepts
• Domain events and entity lifecycle
• Aggregate patterns
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import Field

from flext_core import (
    FlextEntity,
    FlextEntityId,
    FlextEventList,
    FlextMetadata,
    FlextResult,
    FlextTimestamp,
    FlextValue,
    FlextVersion,
)
from flext_core.utilities import FlextGenerators

# Constants to avoid magic numbers
MIN_PRODUCT_CODE_LENGTH = 3
CURRENCY_CODE_LENGTH = 3
MAX_DISCOUNT_PERCENT = 100

# =============================================================================
# VALUE OBJECTS - Immutable domain concepts
# =============================================================================


class Address(FlextValue):
    """Address value object for customer addresses."""

    street: str
    city: str
    postal_code: str
    country: str = "US"


class ProductCode(FlextValue):
    """Product code value object."""

    code: str

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate product code format."""
        if not self.code or len(self.code) < MIN_PRODUCT_CODE_LENGTH:
            return FlextResult[None].fail("Product code must be at least 3 characters")
        if not self.code.isalnum():
            return FlextResult[None].fail("Product code must be alphanumeric")
        return FlextResult[None].ok(None)


class Price(FlextValue):
    """Price value object with currency."""

    amount: Decimal
    currency: str = "USD"

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate price rules."""
        if self.amount < 0:
            return FlextResult[None].fail("Price cannot be negative")
        if len(self.currency) != CURRENCY_CODE_LENGTH:
            return FlextResult[None].fail("Currency must be 3 characters")
        return FlextResult[None].ok(None)

    def add(self, other: Price) -> FlextResult[Price]:
        """Add two prices with currency validation."""
        if self.currency != other.currency:
            return FlextResult[Price].fail(
                "Cannot add prices with different currencies"
            )
        return FlextResult[Price].ok(
            Price.model_validate(
                {"amount": self.amount + other.amount, "currency": self.currency}
            )
        )


# =============================================================================
# ENTITIES - Objects with identity and lifecycle
# =============================================================================


class Product(FlextEntity):
    """Product entity with lifecycle management."""

    code: ProductCode
    name: str
    price: Price
    active: bool = True

    def activate(self) -> FlextResult[Product]:
        """Activate product and raise domain event."""
        if self.active:
            return FlextResult[Product].fail("Product already active")

        self.active = True
        self.add_domain_event(
            "ProductActivated",
            {"product_id": self.id, "code": self.code.code, "timestamp": "now"},
        )
        return FlextResult[Product].ok(self)

    def update_price(self, new_price: Price) -> FlextResult[Product]:
        """Update product price."""
        if not self.active:
            return FlextResult[Product].fail("Cannot update price of inactive product")

        old_price = self.price
        self.price = new_price
        self.add_domain_event(
            "PriceChanged",
            {
                "product_id": self.id,
                "old_price": str(old_price.amount),
                "new_price": str(new_price.amount),
                "currency": new_price.currency,
            },
        )
        return FlextResult[Product].ok(self)


class Customer(FlextEntity):
    """Customer entity with business rules."""

    name: str
    email: str
    address: Address
    active: bool = True

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate customer business rules."""
        if not self.name.strip():
            return FlextResult[None].fail("Customer name required")
        if "@" not in self.email:
            return FlextResult[None].fail("Valid email required")
        return FlextResult[None].ok(None)

    def update_address(self, new_address: Address) -> FlextResult[Customer]:
        """Update customer address."""
        self.address = new_address
        self.add_domain_event(
            "AddressChanged",
            {
                "customer_id": self.id,
                "city": new_address.city,
                "country": new_address.country,
            },
        )
        return FlextResult.ok(self)


# =============================================================================
# AGGREGATES - Consistency boundaries
# =============================================================================


class ShoppingCart(FlextEntity):
    """Shopping cart aggregate root."""

    customer_id: str
    total: Price = Price.model_validate({"amount": Decimal("0.00")})
    items: list[dict[str, object]] = Field(default_factory=list)

    @classmethod
    def create(cls, customer_id: str, **kwargs: object) -> ShoppingCart:
        """Create a shopping cart with proper initialization."""
        cart_id = kwargs.get("id", f"cart_{customer_id}")
        items = kwargs.get("items", [])

        return cls(
            id=str(cart_id),
            customer_id=customer_id,
            items=list(items) if isinstance(items, list) else [],
        )

    def add_item(
        self, product: Product, quantity: int = 1
    ) -> FlextResult[ShoppingCart]:
        """Add item to cart with business rules."""
        if not product.active:
            return FlextResult[ShoppingCart].fail("Cannot add inactive product")
        if quantity <= 0:
            return FlextResult[ShoppingCart].fail("Quantity must be positive")

        # Add item
        item: dict[str, object] = {
            "product_id": product.id,
            "code": product.code.code,
            "name": product.name,
            "price": str(product.price.amount),
            "quantity": quantity,
        }
        self.items.append(item)

        # Update total
        item_total = Price.model_validate(
            {
                "amount": product.price.amount * quantity,
                "currency": product.price.currency,
            }
        )
        total_result = self.total.add(item_total)
        if total_result.success:
            self.total = total_result.value

        # Raise domain event
        self.add_domain_event(
            "ItemAdded",
            {
                "cart_id": self.id,
                "product_code": product.code.code,
                "quantity": quantity,
            },
        )

        return FlextResult[ShoppingCart].ok(self)

    def checkout(self) -> FlextResult[dict[str, object]]:
        """Convert cart to order."""
        if not self.items:
            return FlextResult[dict[str, object]].fail("Cannot checkout empty cart")

        order_data: dict[str, object] = {
            "customer_id": self.customer_id,
            "items": self.items.copy(),
            "total": str(self.total.amount),
            "currency": self.total.currency,
        }

        # Clear cart after checkout
        self.items.clear()
        self.total = Price.model_validate({"amount": Decimal("0.00")})

        self.add_domain_event(
            "CheckoutCompleted",
            {
                "cart_id": self.id,
                "customer_id": self.customer_id,
                "item_count": len(
                    self.items
                ),  # Use self.items instead of order_data["items"]
            },
        )

        return FlextResult.ok(order_data)


# =============================================================================
# DOMAIN SERVICES - Complex business logic
# =============================================================================


class PricingService:
    """Domain service for pricing calculations."""

    @staticmethod
    def calculate_discount(
        price: Price, discount_percent: Decimal
    ) -> FlextResult[Price]:
        """Calculate discounted price."""
        if discount_percent < 0 or discount_percent > MAX_DISCOUNT_PERCENT:
            return FlextResult[Price].fail("Discount must be between 0 and 100")

        discount_amount = price.amount * (discount_percent / 100)
        final_amount = price.amount - discount_amount

        return FlextResult.ok(
            Price.model_validate({"amount": final_amount, "currency": price.currency})
        )

    @staticmethod
    def apply_bulk_discount(
        items: list[Product], min_quantity: int = 10
    ) -> FlextResult[list[Price]]:
        """Apply bulk discount to products."""
        if len(items) < min_quantity:
            # No discount, return original prices
            return FlextResult.ok([item.price for item in items])

        # Apply 10% bulk discount
        discounted_prices = []
        for item in items:
            discount_result = PricingService.calculate_discount(item.price, Decimal(10))
            if discount_result.success:
                discounted_prices.append(discount_result.value)
            else:
                return FlextResult[list[Price]].fail(
                    f"Bulk discount failed: {discount_result.error}"
                )

        return FlextResult.ok(discounted_prices)


# =============================================================================
# FACTORY PATTERN - Domain object creation
# =============================================================================


class DomainObjectFactory:
    """Factory for creating domain objects."""

    @staticmethod
    def create_product(
        code: str, name: str, price_amount: str, currency: str = "USD"
    ) -> FlextResult[Product]:
        """Create product with validation."""
        try:
            # Create value objects
            product_code = ProductCode.model_validate({"code": code})
            price = Price.model_validate(
                {"amount": Decimal(str(price_amount)), "currency": currency}
            )

            # Create entity with proper initialization
            product = Product(
                code=product_code,
                name=name,
                price=price,
                id=FlextEntityId("product_" + code),
                version=FlextVersion(1),
                created_at=FlextTimestamp(),
                updated_at=FlextTimestamp(),
                domain_events=FlextEventList(),
                metadata=FlextMetadata(),
            )

            # Validate business rules
            validation_result = product.validate_business_rules()
            if not validation_result.success:
                error_msg = validation_result.error or "Validation failed"
                return FlextResult[Product].fail(error_msg)

            return FlextResult[Product].ok(product)

        except (ValueError, TypeError) as e:
            return FlextResult[Product].fail(f"Failed to create product: {e}")

    @staticmethod
    def create_customer(
        name: str, email: str, city: str = "Unknown", country: str = "Unknown"
    ) -> FlextResult[Customer]:
        """Create customer with validation."""
        address = Address.model_validate(
            {
                "street": "123 Main St",
                "city": city,
                "postal_code": "12345",
                "country": country,
            }
        )

        customer = Customer(
            name=name,
            email=email,
            address=address,
            id=FlextEntityId(f"customer_{name.replace(' ', '_').lower()}"),
            version=FlextVersion(1),
            created_at=FlextTimestamp(),
            updated_at=FlextTimestamp(),
            domain_events=FlextEventList(),
            metadata=FlextMetadata(),
        )
        validation_result = customer.validate_business_rules()
        if not validation_result.success:
            error_msg = validation_result.error or "Validation failed"
            return FlextResult[Customer].fail(error_msg)
        return FlextResult[Customer].ok(customer)

    @staticmethod
    def create_shopping_cart(customer_id: str) -> FlextResult[ShoppingCart]:
        """Create shopping cart for customer."""
        return FlextResult.ok(
            ShoppingCart.create(
                customer_id=customer_id,
                id=f"cart_{customer_id}_{FlextGenerators.generate_uuid()[:8]}",
            )
        )


# =============================================================================
# DEMONSTRATIONS - Real-world usage patterns
# =============================================================================


def demo_value_objects() -> None:
    """Demonstrate value object patterns."""
    print("\n🧪 Testing value objects...")

    # Create prices
    price1 = Price.model_validate({"amount": Decimal("10.99"), "currency": "USD"})
    price2 = Price.model_validate({"amount": Decimal("5.00"), "currency": "USD"})

    # Add prices
    total_result = price1.add(price2)
    if total_result.success:
        total = total_result.value
        print(f"✅ Total price: {total.amount} {total.currency}")


def demo_entity_lifecycle() -> None:
    """Demonstrate entity lifecycle."""
    print("\n🧪 Testing entity lifecycle...")

    # Create product
    product_result = DomainObjectFactory.create_product(
        "LAPTOP01", "Gaming Laptop", "999.99"
    )

    if product_result.success:
        product = product_result.value

        # Activate product
        activate_result = product.activate()
        if activate_result.success:
            print(f"✅ Product activated: {product.name}")
            print(f"📅 Domain events: {len(product.domain_events)}")


def demo_aggregate_patterns() -> None:
    """Demonstrate aggregate patterns."""
    print("\n🧪 Testing aggregate patterns...")

    # Create shopping cart
    cart_result = DomainObjectFactory.create_shopping_cart("customer_123")
    product_result = DomainObjectFactory.create_product(
        "MOUSE01", "Gaming Mouse", "49.99"
    )

    if cart_result.success and product_result.success:
        cart = cart_result.value
        product = product_result.value

        # Add item to cart
        add_result = cart.add_item(product, quantity=2)
        if add_result.success:
            print(
                f"✅ Item added to cart. Total: {cart.total.amount} {cart.total.currency}"
            )

        # Checkout
        checkout_result = cart.checkout()
        if checkout_result.success:
            order_data = checkout_result.value
            print(f"✅ Checkout completed. Order total: {order_data.get('total')}")


def demo_domain_services() -> None:
    """Demonstrate domain services."""
    print("\n🧪 Testing domain services...")

    price = Price.model_validate({"amount": Decimal("100.00"), "currency": "USD"})
    discount_result = PricingService.calculate_discount(price, Decimal(20))

    if discount_result.success:
        discounted_price = discount_result.value
        print(
            f"✅ Discounted price: {discounted_price.amount} {discounted_price.currency}"
        )


def main() -> None:
    """🎯 Example 06: Domain-Driven Design Patterns."""
    print("=" * 70)
    print("🏗️  EXAMPLE 06: DOMAIN-DRIVEN DESIGN (REFACTORED)")
    print("=" * 70)

    print("\n📚 Refactoring Benefits:")
    print("  • 70% less boilerplate code")
    print("  • Cleaner entity definitions")
    print("  • Simplified domain events")
    print("  • Easier aggregate management")

    print("\n🔍 DEMONSTRATIONS")
    print("=" * 40)

    # Show the refactored examples
    demo_value_objects()
    demo_entity_lifecycle()
    demo_aggregate_patterns()
    demo_domain_services()

    print("\n" + "=" * 70)
    print("✅ REFACTORED DDD EXAMPLE COMPLETED!")
    print("=" * 70)

    print("\n🎓 Key Improvements:")
    print("  • Simplified entity and value object creation")
    print("  • Automatic domain event management")
    print("  • Reduced validation boilerplate")
    print("  • Cleaner aggregate boundaries")


if __name__ == "__main__":
    main()
