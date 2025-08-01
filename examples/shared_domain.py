#!/usr/bin/env python3
"""Shared Domain Models for FLEXT Examples.

This module provides reusable domain models, value objects, and utilities
that are commonly used across multiple examples. This reduces code duplication
and demonstrates proper domain-driven design patterns with flext-core.

Features:
- Common value objects (Email, Money, Address, etc.)
- Shared entities (User, Customer, Product, Order, etc.)
- Business rule constants and validation
- Factory patterns for consistent object creation
- Mixins for cross-cutting concerns
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

from flext_core import (
    FlextEntity,
    FlextResult,
    FlextUtilities,
    FlextValueObject,
    TEntityId,
    get_logger,
)

# =============================================================================
# BUSINESS CONSTANTS - Centralized business rules
# =============================================================================

# Age validation
MIN_AGE = 18
MAX_AGE = 120

# Email validation
MAX_EMAIL_LENGTH = 254

# Name validation
MIN_NAME_LENGTH = 2
MAX_NAME_LENGTH = 100

# Address validation
MIN_STREET_LENGTH = 5
MIN_CITY_LENGTH = 2
MIN_POSTAL_CODE_LENGTH = 3
MIN_COUNTRY_LENGTH = 2

# Currency validation
CURRENCY_CODE_LENGTH = 3
SUPPORTED_CURRENCIES = {"USD", "EUR", "GBP", "JPY", "CAD", "BRL"}

# Phone validation
MIN_PHONE_LENGTH = 10
MAX_PHONE_LENGTH = 15

# Product validation
MIN_PRODUCT_NAME_LENGTH = 3
MAX_PRODUCT_PRICE = 100000

# Order validation
MIN_ORDER_ITEMS = 1
MAX_ORDER_ITEMS = 50

# Validation message lengths
MIN_REASON_LENGTH = 10
MIN_DESCRIPTION_LENGTH = 10

# Product item validation
MIN_PRODUCT_ITEM_NAME_LENGTH = 2
MAX_ORDER_ITEM_QUANTITY = 100

# Email format validation
REQUIRED_EMAIL_PARTS = 2

# =============================================================================
# ENUMS - Domain enumerations
# =============================================================================


class UserStatus(StrEnum):
    """User status enumeration."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"


class OrderStatus(StrEnum):
    """Order status enumeration."""

    PENDING = "pending"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class PaymentMethod(StrEnum):
    """Payment method enumeration."""

    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    PAYPAL = "paypal"
    BANK_TRANSFER = "bank_transfer"
    CASH = "cash"


# =============================================================================
# VALUE OBJECTS - Reusable immutable domain concepts
# =============================================================================


class EmailAddress(FlextValueObject):
    """Email address value object with comprehensive validation."""

    email: str

    def validate_domain_rules(self) -> FlextResult[None]:  # noqa: PLR0911
        """Validate email address domain rules."""
        if not self.email or not isinstance(self.email, str):
            return FlextResult.fail("Email must be a non-empty string")

        email = self.email.strip().lower()

        if "@" not in email:
            return FlextResult.fail("Email must contain @ symbol")

        if len(email) > MAX_EMAIL_LENGTH:
            return FlextResult.fail(
                f"Email cannot exceed {MAX_EMAIL_LENGTH} characters",
            )

        parts = email.split("@")
        if len(parts) != REQUIRED_EMAIL_PARTS:
            return FlextResult.fail("Email must have exactly one @ symbol")

        local, domain = parts
        if not local or not domain:
            return FlextResult.fail("Email must have both local and domain parts")

        if "." not in domain:
            return FlextResult.fail("Email domain must contain at least one dot")

        return FlextResult.ok(None)


class Age(FlextValueObject):
    """Age value object with business rule validation."""

    value: int

    def validate_domain_rules(self) -> FlextResult[None]:
        """Validate age business rules."""
        if self.value < MIN_AGE:
            return FlextResult.fail(f"Age must be at least {MIN_AGE}")

        if self.value > MAX_AGE:
            return FlextResult.fail(f"Age cannot exceed {MAX_AGE}")

        return FlextResult.ok(None)


class Money(FlextValueObject):
    """Money value object with currency and amount validation."""

    amount: Decimal
    currency: str

    def validate_domain_rules(self) -> FlextResult[None]:
        """Validate money domain rules."""
        if self.amount < 0:
            return FlextResult.fail("Amount cannot be negative")

        if not self.currency or len(self.currency) != CURRENCY_CODE_LENGTH:
            return FlextResult.fail(
                f"Currency must be {CURRENCY_CODE_LENGTH} characters",
            )

        if self.currency.upper() not in SUPPORTED_CURRENCIES:
            return FlextResult.fail(f"Unsupported currency: {self.currency}")

        return FlextResult.ok(None)

    def add(self, other: Money) -> FlextResult[Money]:
        """Add two money amounts (same currency only)."""
        if self.currency != other.currency:
            return FlextResult.fail(
                f"Cannot add different currencies: {self.currency} + {other.currency}",
            )

        result = Money(
            amount=self.amount + other.amount,
            currency=self.currency,
        )

        validation = result.validate_domain_rules()
        if validation.is_failure:
            return FlextResult.fail(validation.error or "Invalid money calculation")

        return FlextResult.ok(result)

    def multiply(self, factor: Decimal) -> FlextResult[Money]:
        """Multiply money by a factor."""
        if factor < 0:
            return FlextResult.fail("Factor cannot be negative")

        result = Money(
            amount=self.amount * factor,
            currency=self.currency,
        )

        validation = result.validate_domain_rules()
        if validation.is_failure:
            return FlextResult.fail(validation.error or "Invalid money calculation")

        return FlextResult.ok(result)


class Address(FlextValueObject):
    """Address value object with validation."""

    street: str
    city: str
    postal_code: str
    country: str
    state: str | None = None

    def validate_domain_rules(self) -> FlextResult[None]:
        """Validate address domain rules."""
        if not self.street or len(self.street.strip()) < MIN_STREET_LENGTH:
            return FlextResult.fail(
                f"Street must be at least {MIN_STREET_LENGTH} characters",
            )

        if not self.city or len(self.city.strip()) < MIN_CITY_LENGTH:
            return FlextResult.fail(
                f"City must be at least {MIN_CITY_LENGTH} characters",
            )

        if (
            not self.postal_code
            or len(self.postal_code.strip()) < MIN_POSTAL_CODE_LENGTH
        ):
            return FlextResult.fail(
                f"Postal code must be at least {MIN_POSTAL_CODE_LENGTH} characters",
            )

        if not self.country or len(self.country.strip()) < MIN_COUNTRY_LENGTH:
            return FlextResult.fail(
                f"Country must be at least {MIN_COUNTRY_LENGTH} characters",
            )

        return FlextResult.ok(None)

    def is_same_city(self, other: Address) -> bool:
        """Check if two addresses are in the same city."""
        return self.city.strip().lower() == other.city.strip().lower()


class PhoneNumber(FlextValueObject):
    """Phone number value object with validation."""

    number: str
    country_code: str | None = None

    def validate_domain_rules(self) -> FlextResult[None]:
        """Validate phone number domain rules."""
        if not self.number:
            return FlextResult.fail("Phone number cannot be empty")

        # Remove common formatting characters
        clean_number = "".join(c for c in self.number if c.isdigit())

        if len(clean_number) < MIN_PHONE_LENGTH:
            return FlextResult.fail(
                f"Phone number must have at least {MIN_PHONE_LENGTH} digits",
            )

        if len(clean_number) > MAX_PHONE_LENGTH:
            return FlextResult.fail(
                f"Phone number cannot exceed {MAX_PHONE_LENGTH} digits",
            )

        return FlextResult.ok(None)


# =============================================================================
# ENTITIES - Domain entities with identity and behavior
# =============================================================================


class User(FlextEntity):
    """User entity with comprehensive business rules."""

    name: str
    email_address: EmailAddress
    age: Age
    status: UserStatus = UserStatus.PENDING
    phone: PhoneNumber | None = None
    address: Address | None = None

    def validate_domain_rules(self) -> FlextResult[None]:  # noqa: PLR0911
        """Validate user domain rules."""
        if not self.name or len(self.name.strip()) < MIN_NAME_LENGTH:
            return FlextResult.fail(
                f"Name must be at least {MIN_NAME_LENGTH} characters",
            )

        if len(self.name.strip()) > MAX_NAME_LENGTH:
            return FlextResult.fail(f"Name cannot exceed {MAX_NAME_LENGTH} characters")

        # Validate embedded value objects
        email_validation = self.email_address.validate_domain_rules()
        if email_validation.is_failure:
            return FlextResult.fail(
                f"Email validation failed: {email_validation.error}",
            )

        age_validation = self.age.validate_domain_rules()
        if age_validation.is_failure:
            return FlextResult.fail(f"Age validation failed: {age_validation.error}")

        if self.phone:
            phone_validation = self.phone.validate_domain_rules()
            if phone_validation.is_failure:
                return FlextResult.fail(
                    f"Phone validation failed: {phone_validation.error}",
                )

        if self.address:
            address_validation = self.address.validate_domain_rules()
            if address_validation.is_failure:
                return FlextResult.fail(
                    f"Address validation failed: {address_validation.error}",
                )

        return FlextResult.ok(None)

    def activate(self) -> FlextResult[User]:
        """Activate user account."""
        if self.status == UserStatus.ACTIVE:
            return FlextResult.fail("User is already active")

        result = self.copy_with(status=UserStatus.ACTIVE)
        if result.is_success and result.data is not None:
            activated_user = result.data
            # Add domain event
            event_result = activated_user.add_domain_event(
                "UserActivated",
                {
                    "user_id": self.id,
                    "email": self.email_address.email,
                    "activated_at": FlextUtilities.generate_iso_timestamp(),
                },
            )
            if event_result.is_failure:
                return FlextResult.fail(
                    f"Failed to add domain event: {event_result.error}",
                )

        return result

    def suspend(self, reason: str) -> FlextResult[User]:
        """Suspend user account with reason."""
        if self.status == UserStatus.SUSPENDED:
            return FlextResult.fail("User is already suspended")

        if not reason or len(reason.strip()) < MIN_REASON_LENGTH:
            return FlextResult.fail("Suspension reason must be at least 10 characters")

        result = self.copy_with(status=UserStatus.SUSPENDED)
        if result.is_success and result.data is not None:
            suspended_user = result.data
            # Add domain event
            event_result = suspended_user.add_domain_event(
                "UserSuspended",
                {
                    "user_id": self.id,
                    "reason": reason,
                    "suspended_at": FlextUtilities.generate_iso_timestamp(),
                },
            )
            if event_result.is_failure:
                return FlextResult.fail(
                    f"Failed to add domain event: {event_result.error}",
                )

        return result


class Product(FlextEntity):
    """Product entity with business rules."""

    name: str
    description: str
    price: Money
    category: str
    in_stock: bool = True

    def validate_domain_rules(self) -> FlextResult[None]:
        """Validate product domain rules."""
        if not self.name or len(self.name.strip()) < MIN_PRODUCT_NAME_LENGTH:
            return FlextResult.fail(
                f"Product name must be at least {MIN_PRODUCT_NAME_LENGTH} characters",
            )

        if (
            not self.description
            or len(self.description.strip()) < MIN_DESCRIPTION_LENGTH
        ):
            return FlextResult.fail(
                "Product description must be at least 10 characters",
            )

        if not self.category or len(self.category.strip()) < MIN_NAME_LENGTH:
            return FlextResult.fail("Product category must be at least 2 characters")

        # Validate embedded value objects
        price_validation = self.price.validate_domain_rules()
        if price_validation.is_failure:
            return FlextResult.fail(
                f"Price validation failed: {price_validation.error}",
            )

        if self.price.amount > Decimal(str(MAX_PRODUCT_PRICE)):
            return FlextResult.fail(f"Product price cannot exceed {MAX_PRODUCT_PRICE}")

        return FlextResult.ok(None)

    def update_price(self, new_price: Money) -> FlextResult[Product]:
        """Update product price with validation."""
        price_validation = new_price.validate_domain_rules()
        if price_validation.is_failure:
            return FlextResult.fail(f"Invalid price: {price_validation.error}")

        result = self.copy_with(price=new_price)
        if result.is_success and result.data is not None:
            updated_product = result.data
            # Add domain event
            event_result = updated_product.add_domain_event(
                "ProductPriceUpdated",
                {
                    "product_id": self.id,
                    "old_price": {
                        "amount": str(self.price.amount),
                        "currency": self.price.currency,
                    },
                    "new_price": {
                        "amount": str(new_price.amount),
                        "currency": new_price.currency,
                    },
                    "updated_at": FlextUtilities.generate_iso_timestamp(),
                },
            )
            if event_result.is_failure:
                return FlextResult.fail(
                    f"Failed to add domain event: {event_result.error}",
                )

        return result


class OrderItem(FlextValueObject):
    """Order item value object."""

    product_id: TEntityId
    product_name: str
    quantity: int
    unit_price: Money

    def validate_domain_rules(self) -> FlextResult[None]:
        """Validate order item domain rules."""
        if not self.product_id:
            return FlextResult.fail("Product ID cannot be empty")

        if (
            not self.product_name
            or len(self.product_name.strip()) < MIN_PRODUCT_ITEM_NAME_LENGTH
        ):
            return FlextResult.fail("Product name must be at least 2 characters")

        if self.quantity <= 0:
            return FlextResult.fail("Quantity must be positive")

        if self.quantity > MAX_ORDER_ITEM_QUANTITY:
            return FlextResult.fail("Quantity cannot exceed 100")

        # Validate price
        price_validation = self.unit_price.validate_domain_rules()
        if price_validation.is_failure:
            return FlextResult.fail(
                f"Unit price validation failed: {price_validation.error}",
            )

        return FlextResult.ok(None)

    def total_price(self) -> FlextResult[Money]:
        """Calculate total price for this item."""
        return self.unit_price.multiply(Decimal(str(self.quantity)))


class Order(FlextEntity):
    """Order entity with complex business rules."""

    customer_id: TEntityId
    items: list[OrderItem]
    status: OrderStatus = OrderStatus.PENDING
    payment_method: PaymentMethod | None = None
    shipping_address: Address | None = None

    def validate_domain_rules(self) -> FlextResult[None]:
        """Validate order domain rules."""
        if not self.customer_id:
            return FlextResult.fail("Customer ID cannot be empty")

        if not self.items:
            return FlextResult.fail("Order must have at least one item")

        if len(self.items) > MAX_ORDER_ITEMS:
            return FlextResult.fail(
                f"Order cannot have more than {MAX_ORDER_ITEMS} items",
            )

        # Validate all items
        for i, item in enumerate(self.items):
            item_validation = item.validate_domain_rules()
            if item_validation.is_failure:
                return FlextResult.fail(
                    f"Item {i + 1} validation failed: {item_validation.error}",
                )

        # Validate shipping address if provided
        if self.shipping_address:
            address_validation = self.shipping_address.validate_domain_rules()
            if address_validation.is_failure:
                return FlextResult.fail(
                    f"Shipping address validation failed: {address_validation.error}",
                )

        return FlextResult.ok(None)

    def calculate_total(self) -> FlextResult[Money]:
        """Calculate order total."""
        if not self.items:
            return FlextResult.fail("Cannot calculate total for empty order")

        # Use first item's currency as base
        base_currency = self.items[0].unit_price.currency
        total_amount = Decimal(0)

        for item in self.items:
            if item.unit_price.currency != base_currency:
                return FlextResult.fail("All items must have the same currency")

            item_total_result = item.total_price()
            if item_total_result.is_failure:
                return FlextResult.fail(
                    f"Failed to calculate item total: {item_total_result.error}",
                )

            if item_total_result.data is not None:
                total_amount += item_total_result.data.amount

        return FlextResult.ok(Money(amount=total_amount, currency=base_currency))

    def confirm(self) -> FlextResult[Order]:
        """Confirm the order."""
        if self.status != OrderStatus.PENDING:
            return FlextResult.fail("Only pending orders can be confirmed")

        # Validate order can be fulfilled
        total_result = self.calculate_total()
        if total_result.is_failure:
            return FlextResult.fail(f"Cannot confirm order: {total_result.error}")

        result = self.copy_with(status=OrderStatus.CONFIRMED)
        if (
            result.is_success
            and result.data is not None
            and total_result.data is not None
        ):
            confirmed_order = result.data
            # Add domain event
            event_result = confirmed_order.add_domain_event(
                "OrderConfirmed",
                {
                    "order_id": self.id,
                    "customer_id": self.customer_id,
                    "total_amount": str(total_result.data.amount),
                    "currency": total_result.data.currency,
                    "item_count": len(self.items),
                    "confirmed_at": FlextUtilities.generate_iso_timestamp(),
                },
            )
            if event_result.is_failure:
                return FlextResult.fail(
                    f"Failed to add domain event: {event_result.error}",
                )

        return result


# =============================================================================
# FACTORY PATTERNS - Convenient object creation
# =============================================================================


class SharedDomainFactory:
    """Factory for creating shared domain objects with defaults."""

    @staticmethod
    def create_user(
        name: str,
        email: str,
        age: int,
        **kwargs: object,
    ) -> FlextResult[User]:
        """Create user with validation."""
        # Create value objects
        email_result = EmailAddress(email=email).validate_flext()
        if email_result.is_failure:
            return FlextResult.fail(f"Invalid email: {email_result.error}")

        age_result = Age(value=age).validate_flext()
        if age_result.is_failure:
            return FlextResult.fail(f"Invalid age: {age_result.error}")

        if email_result.data is None or age_result.data is None:
            return FlextResult.fail("Failed to create required value objects")

        # Create user entity
        try:
            # Generate ID if not provided
            entity_id = str(kwargs.get("id", FlextUtilities.generate_entity_id()))

            user = User(
                id=entity_id,
                name=name,
                email_address=email_result.data,
                age=age_result.data,
                status=kwargs.get("status", UserStatus.PENDING),  # type: ignore[arg-type]
                phone=kwargs.get("phone"),  # type: ignore[arg-type]
                address=kwargs.get("address"),  # type: ignore[arg-type]
            )

            validation_result = user.validate_domain_rules()
            if validation_result.is_failure:
                return FlextResult.fail(
                    f"User validation failed: {validation_result.error}",
                )

            return FlextResult.ok(user)

        except (TypeError, ValueError) as e:
            return FlextResult.fail(f"Failed to create user: {e}")

    @staticmethod
    def create_product(
        name: str,
        description: str,
        price_amount: str | Decimal,
        currency: str = "USD",
        **kwargs: object,
    ) -> FlextResult[Product]:
        """Create product with validation."""
        # Create money value object
        try:
            if isinstance(price_amount, str):
                price_amount = Decimal(price_amount)

            money_result = Money(
                amount=price_amount,
                currency=currency,
            ).validate_flext()
            if money_result.is_failure:
                return FlextResult.fail(f"Invalid price: {money_result.error}")

            if money_result.data is None:
                return FlextResult.fail("Failed to create price value object")

            # Create product entity
            entity_id = str(kwargs.get("id", FlextUtilities.generate_entity_id()))

            product = Product(
                id=entity_id,
                name=name,
                description=description,
                price=money_result.data,
                category=str(kwargs.get("category", "general")),
                in_stock=bool(kwargs.get("in_stock", True)),
            )

            validation_result = product.validate_domain_rules()
            if validation_result.is_failure:
                return FlextResult.fail(
                    f"Product validation failed: {validation_result.error}",
                )

            return FlextResult.ok(product)

        except (TypeError, ValueError, ArithmeticError) as e:
            return FlextResult.fail(f"Failed to create product: {e}")

    @staticmethod
    def create_order(
        customer_id: TEntityId,
        items: list[dict[str, object]],
        **kwargs: object,
    ) -> FlextResult[Order]:
        """Create order with items validation."""
        try:
            # Create order items
            order_items = []
            for item_data in items:
                money_result = Money(
                    amount=Decimal(str(item_data["unit_price"])),
                    currency=str(item_data.get("currency", "USD")),
                ).validate_flext()
                if money_result.is_failure:
                    return FlextResult.fail(f"Invalid item price: {money_result.error}")

                if money_result.data is None:
                    return FlextResult.fail("Failed to create item price value object")

                order_item = OrderItem(
                    product_id=str(item_data["product_id"]),
                    product_name=str(item_data["product_name"]),
                    quantity=int(str(item_data["quantity"])),
                    unit_price=money_result.data,
                )

                item_validation = order_item.validate_domain_rules()
                if item_validation.is_failure:
                    return FlextResult.fail(
                        f"Invalid order item: {item_validation.error}",
                    )

                order_items.append(order_item)

            # Create order entity
            entity_id = str(kwargs.get("id", FlextUtilities.generate_entity_id()))

            order = Order(
                id=entity_id,
                customer_id=customer_id,
                items=order_items,
                status=kwargs.get("status", OrderStatus.PENDING),
                payment_method=kwargs.get("payment_method"),
                shipping_address=kwargs.get("shipping_address"),
            )

            validation_result = order.validate_domain_rules()
            if validation_result.is_failure:
                return FlextResult.fail(
                    f"Order validation failed: {validation_result.error}",
                )

            return FlextResult.ok(order)

        except (TypeError, ValueError, KeyError, ArithmeticError) as e:
            return FlextResult.fail(f"Failed to create order: {e}")


# =============================================================================
# UTILITIES - Helper functions
# =============================================================================


def log_domain_operation(
    operation: str,
    entity_type: str,
    entity_id: str,
    **context: object,
) -> None:
    """Log domain operation with structured context."""
    logger = get_logger("shared_domain")
    logger.info(
        "Domain operation: %s",
        operation,
        entity_type=entity_type,
        entity_id=entity_id,
        **context,
    )


# Export all public components
__all__ = [
    "CURRENCY_CODE_LENGTH",
    "MAX_AGE",
    "MAX_EMAIL_LENGTH",
    "MAX_NAME_LENGTH",
    "MIN_AGE",
    "MIN_NAME_LENGTH",
    "SUPPORTED_CURRENCIES",
    "Address",
    "Age",
    "ComplexValueObject",
    # Test models
    "ConcreteFlextEntity",
    "ConcreteValueObject",
    "EmailAddress",
    "Money",
    "Order",
    "OrderItem",
    "OrderStatus",
    "PaymentMethod",
    "PhoneNumber",
    "Product",
    "SharedDomainFactory",
    "TestDomainFactory",
    "User",
    "UserStatus",
    "log_domain_operation",
]


# =============================================================================
# TEST DOMAIN MODELS - For comprehensive test coverage
# =============================================================================


class ConcreteFlextEntity(FlextEntity):
    """Concrete entity implementation for comprehensive testing."""

    name: str
    status: str = "active"

    def validate_domain_rules(self) -> FlextResult[None]:
        """Validate test entity domain rules."""
        if not self.name.strip():
            return FlextResult.fail("Entity name cannot be empty")
        return FlextResult.ok(None)


class ConcreteValueObject(FlextValueObject):
    """Concrete value object implementation for comprehensive testing."""

    amount: Decimal
    currency: str = "USD"
    description: str = ""

    def validate_domain_rules(self) -> FlextResult[None]:
        """Validate test value object domain rules."""
        if self.amount < 0:
            return FlextResult.fail("Amount cannot be negative")
        currency_code_length = 3  # ISO 4217 standard
        if len(self.currency) != currency_code_length:
            return FlextResult.fail("Currency must be 3 characters")
        if not self.currency.isupper():
            return FlextResult.fail("Currency must be uppercase")
        return FlextResult.ok(None)


class ComplexValueObject(FlextValueObject):
    """Value object with complex data types for testing."""

    name: str
    tags: list[str]
    metadata: dict[str, object]

    def validate_domain_rules(self) -> FlextResult[None]:
        """Validate complex value object domain rules."""
        if not self.name.strip():
            return FlextResult.fail("Name cannot be empty")
        return FlextResult.ok(None)


class TestDomainFactory:
    """Factory for creating test domain objects with defaults."""

    @staticmethod
    def create_concrete_entity(
        name: str,
        status: str = "active",
        **kwargs: object,
    ) -> FlextResult[ConcreteFlextEntity]:
        """Create a concrete entity for testing."""
        try:
            entity_id = str(kwargs.get("id", FlextUtilities.generate_entity_id()))
            entity = ConcreteFlextEntity(id=entity_id, name=name, status=status)
            validation_result = entity.validate_domain_rules()
            if validation_result.is_failure:
                return FlextResult.fail(
                    f"Entity validation failed: {validation_result.error}",
                )
            return FlextResult.ok(entity)
        # Test factory needs exceptions handling
        except (RuntimeError, ValueError, TypeError) as e:
            return FlextResult.fail(f"Failed to create test entity: {e}")

    @staticmethod
    def create_concrete_value_object(
        amount: Decimal,
        currency: str = "USD",
        **kwargs: object,
    ) -> FlextResult[ConcreteValueObject]:
        """Create a concrete value object for testing."""
        try:
            description = str(kwargs.get("description", ""))
            vo = ConcreteValueObject(
                amount=amount, currency=currency, description=description
            )
            validation_result = vo.validate_domain_rules()
            if validation_result.is_failure:
                return FlextResult.fail(
                    f"Value object validation failed: {validation_result.error}"
                )
            return FlextResult.ok(vo)
        # Test factory needs exceptions handling
        except (RuntimeError, ValueError, TypeError) as e:
            return FlextResult.fail(f"Failed to create test value object: {e}")

    @staticmethod
    def create_complex_value_object(
        name: str,
        tags: list[str],
        metadata: dict[str, object],
    ) -> FlextResult[ComplexValueObject]:
        """Create a complex value object for testing."""
        try:
            vo = ComplexValueObject(name=name, tags=tags, metadata=metadata)
            validation_result = vo.validate_domain_rules()
            if validation_result.is_failure:
                return FlextResult.fail(
                    f"Complex value object validation failed: {validation_result.error}"
                )
            return FlextResult.ok(vo)
        # Test factory needs exceptions handling
        except (RuntimeError, ValueError, TypeError) as e:
            return FlextResult.fail(f"Failed to create complex value object: {e}")


# =============================================================================
# SHARED DEMONSTRATION PATTERNS - DRY PRINCIPLE
# =============================================================================


class SharedDemonstrationPattern:
    """Shared demonstration pattern to eliminate code duplication in examples.
    
    DRY PRINCIPLE: Eliminates 18-line duplication (mass=102) between main functions.
    """

    @staticmethod
    def run_demonstration(
        title: str,
        demonstration_functions: list[callable],
    ) -> None:
        """Run a demonstration with consistent formatting and error handling."""
        print("=" * 80)
        print(f"🚀 {title}")
        print("=" * 80)

        # Run all demonstrations
        for demo_func in demonstration_functions:
            try:
                demo_func()
            except (RuntimeError, ValueError, TypeError) as e:
                print(f"❌ Demonstration function {demo_func.__name__} failed: {e}")

        print("\n" + "=" * 80)
        print(f"🎉 {title.split(' - ')[0]} DEMONSTRATION COMPLETED")
        print("=" * 80)
