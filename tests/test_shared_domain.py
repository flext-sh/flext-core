"""Test-specific shared domain models for FLEXT Core tests.

This module provides reusable test fixtures, domain models, and utilities
that can be shared across all test modules, eliminating code duplication and
promoting consistent testing patterns.

ARCHITECTURE: This replaces the circular import from examples/shared_domain.py
by providing test-specific domain models directly within the tests directory.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from enum import StrEnum

from pydantic import Field, field_validator

from flext_core import FlextEntity, FlextResult, FlextValueObject, TEntityId

# =============================================================================
# TEST DOMAIN CONSTANTS
# =============================================================================

MIN_AGE = 18
MAX_AGE = 120
CURRENCY_CODE_LENGTH = 3
SUPPORTED_CURRENCIES = {"USD", "EUR", "GBP", "JPY", "CAD"}


# =============================================================================
# TEST ENUMS
# =============================================================================


class TestUserStatus(StrEnum):
    """Test user status enumeration."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"


class TestOrderStatus(StrEnum):
    """Test order status enumeration."""

    PENDING = "pending"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


# =============================================================================
# TEST VALUE OBJECTS
# =============================================================================


class TestEmailAddress(FlextValueObject):
    """Test email address value object."""

    email: str

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate email address for testing."""
        if not self.email or not isinstance(self.email, str):
            return FlextResult.fail("Email must be a non-empty string")

        if "@" not in self.email:
            return FlextResult.fail("Email must contain @ symbol")

        return FlextResult.ok(None)


class TestMoney(FlextValueObject):
    """Test money value object."""

    amount: Decimal
    currency: str = "USD"
    description: str = ""  # Default empty string

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate money for testing."""
        if self.amount < 0:
            return FlextResult.fail("Amount cannot be negative")

        if len(self.currency) != CURRENCY_CODE_LENGTH:
            return FlextResult.fail(
                f"Currency must be {CURRENCY_CODE_LENGTH} characters",
            )

        if self.currency != self.currency.upper():
            return FlextResult.fail("Currency must be uppercase")

        return FlextResult.ok(None)


class TestAddress(FlextValueObject):
    """Test address value object."""

    street: str
    city: str
    state: str
    zip_code: str
    country: str = "USA"

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate address for testing."""
        if not all([self.street, self.city, self.state, self.zip_code]):
            return FlextResult.fail("All address fields are required")

        return FlextResult.ok(None)


class TestComplexValueObject(FlextValueObject):
    """Test complex value object with name, tags, and metadata."""

    name: str
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate complex value object for testing."""
        if not self.name or not self.name.strip():
            return FlextResult.fail("Name cannot be empty or whitespace")

        if len(self.name.strip()) < 2:
            return FlextResult.fail("Name must be at least 2 characters")

        return FlextResult.ok(None)


# =============================================================================
# TEST ENTITIES
# =============================================================================


class TestUser(FlextEntity):
    """Test user entity."""

    name: str
    email: str
    age: int = 25
    status: TestUserStatus = TestUserStatus.ACTIVE
    balance: Decimal = Decimal("0.00")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name field at Pydantic level."""
        if not v or not v.strip():
            empty_name_error = "Name cannot be empty"
            raise ValueError(empty_name_error)
        return v

    def validate_domain_rules(self) -> FlextResult[None]:
        """Validate user domain rules for testing."""
        if not self.name:
            return FlextResult.fail("Entity name cannot be empty")

        if len(self.name) < 2:
            return FlextResult.fail("Name must be at least 2 characters")

        if self.age < MIN_AGE:
            return FlextResult.fail(f"Age must be at least {MIN_AGE}")

        if "@" not in self.email:
            return FlextResult.fail("Invalid email format")

        return FlextResult.ok(None)

    def activate(self) -> FlextResult[None]:
        """Activate user account."""
        if self.status == TestUserStatus.ACTIVE:
            return FlextResult.fail("User is already active")

        # In a real entity, this would be done through a proper method
        # For testing purposes, we simulate the state change
        return FlextResult.ok(None)

    def deactivate(self) -> FlextResult[None]:
        """Deactivate user account."""
        if self.status == TestUserStatus.INACTIVE:
            return FlextResult.fail("User is already inactive")

        return FlextResult.ok(None)


class TestOrder(FlextEntity):
    """Test order entity."""

    user_id: TEntityId
    total_amount: Decimal
    status: TestOrderStatus = TestOrderStatus.PENDING
    items: list[str] = Field(default_factory=list)

    def validate_domain_rules(self) -> FlextResult[None]:
        """Validate order domain rules for testing."""
        if not self.user_id:
            return FlextResult.fail("User ID is required")

        if self.total_amount <= 0:
            return FlextResult.fail("Total amount must be positive")

        if not self.items:
            return FlextResult.fail("Order must have at least one item")

        return FlextResult.ok(None)

    def add_item(self, item: str) -> FlextResult[None]:
        """Add item to order."""
        if not item:
            return FlextResult.fail("Item cannot be empty")

        if item in self.items:
            return FlextResult.fail("Item already exists in order")

        # Simulate adding item
        return FlextResult.ok(None)

    def confirm(self) -> FlextResult[None]:
        """Confirm order."""
        if self.status != TestOrderStatus.PENDING:
            return FlextResult.fail("Only pending orders can be confirmed")

        return FlextResult.ok(None)


class TestProduct(FlextEntity):
    """Test product entity."""

    name: str
    price: Decimal
    description: str = ""
    in_stock: bool = True
    category: str = "general"

    def validate_domain_rules(self) -> FlextResult[None]:
        """Validate product domain rules for testing."""
        if not self.name or len(self.name) < 2:
            return FlextResult.fail("Product name must be at least 2 characters")

        if self.price <= 0:
            return FlextResult.fail("Price must be positive")

        return FlextResult.ok(None)

    def update_stock(self, *, _in_stock: bool) -> FlextResult[None]:
        """Update stock status."""
        # Simulate stock update
        return FlextResult.ok(None)


# =============================================================================
# TEST DOMAIN FACTORY
# =============================================================================


class TestDomainFactory:
    """Factory for creating test domain objects."""

    @classmethod
    def create_test_user(
        cls,
        name: str = "Test User",
        email: str = "test@example.com",
        **kwargs: object,
    ) -> FlextResult[TestUser]:
        """Create test user with validation."""
        try:
            age_value = kwargs.get("age", 25)
            age = int(age_value) if isinstance(age_value, (int, str, float)) else 25
            status_str = str(kwargs.get("status", "active"))
            status = (
                TestUserStatus(status_str)
                if status_str in TestUserStatus
                else TestUserStatus.ACTIVE
            )
            balance_value = kwargs.get("balance", "0.00")
            balance = Decimal(str(balance_value))

            user = TestUser(
                id=str(kwargs.get("id", f"user_{len(name)}")),
                name=name,
                email=email,
                age=age,
                status=status,
                balance=balance,
            )

            validation = user.validate_domain_rules()
            if validation.is_failure:
                return FlextResult.fail(f"User validation failed: {validation.error}")

            return FlextResult.ok(user)

        except Exception as e:
            return FlextResult.fail(f"Failed to create test user: {e}")

    @classmethod
    def create_test_order(
        cls,
        user_id: TEntityId,
        total_amount: str | Decimal = "100.00",
        **kwargs: object,
    ) -> FlextResult[TestOrder]:
        """Create test order with validation."""
        try:
            amount = Decimal(str(total_amount))
            status_str = str(kwargs.get("status", "pending"))
            status = (
                TestOrderStatus(status_str)
                if status_str in TestOrderStatus
                else TestOrderStatus.PENDING
            )
            items_value = kwargs.get("items", ["test_item"])
            items = (
                list(items_value)
                if isinstance(items_value, (list, tuple))
                else ["test_item"]
            )

            order = TestOrder(
                id=str(kwargs.get("id", f"order_{user_id}")),
                user_id=user_id,
                total_amount=amount,
                status=status,
                items=items,
            )

            validation = order.validate_domain_rules()
            if validation.is_failure:
                return FlextResult.fail(f"Order validation failed: {validation.error}")

            return FlextResult.ok(order)

        except Exception as e:
            return FlextResult.fail(f"Failed to create test order: {e}")

    @classmethod
    def create_test_product(
        cls,
        name: str = "Test Product",
        price: str | Decimal = "50.00",
        **kwargs: object,
    ) -> FlextResult[TestProduct]:
        """Create test product with validation."""
        try:
            product_price = Decimal(str(price))
            description = str(kwargs.get("description", ""))
            in_stock = bool(kwargs.get("in_stock", True))
            category = str(kwargs.get("category", "general"))

            product = TestProduct(
                id=str(kwargs.get("id", f"product_{len(name)}")),
                name=name,
                price=product_price,
                description=description,
                in_stock=in_stock,
                category=category,
            )

            validation = product.validate_domain_rules()
            if validation.is_failure:
                return FlextResult.fail(
                    f"Product validation failed: {validation.error}",
                )

            return FlextResult.ok(product)

        except Exception as e:
            return FlextResult.fail(f"Failed to create test product: {e}")

    @classmethod
    def create_test_email(
        cls,
        email: str = "test@example.com",
    ) -> FlextResult[TestEmailAddress]:
        """Create test email address with validation."""
        try:
            email_obj = TestEmailAddress(email=email)
            validation = email_obj.validate_business_rules()
            if validation.is_failure:
                return FlextResult.fail(f"Email validation failed: {validation.error}")

            return FlextResult.ok(email_obj)

        except Exception as e:
            return FlextResult.fail(f"Failed to create test email: {e}")

    @classmethod
    def create_test_money(
        cls,
        amount: str | Decimal = "100.00",
        currency: str = "USD",
        description: str = "test money",
    ) -> FlextResult[TestMoney]:
        """Create test money with validation."""
        try:
            money_amount = Decimal(str(amount))
            money_obj = TestMoney(
                amount=money_amount,
                currency=currency,
                description=description,
            )
            validation = money_obj.validate_business_rules()
            if validation.is_failure:
                return FlextResult.fail(f"Money validation failed: {validation.error}")

            return FlextResult.ok(money_obj)

        except Exception as e:
            return FlextResult.fail(f"Failed to create test money: {e}")

    # Add compatibility aliases for test methods
    @classmethod
    def create_concrete_entity(
        cls,
        name: str = "Test User",
        **kwargs: object,
    ) -> FlextResult[TestUser]:
        """Create concrete entity (alias for create_test_user for backward compatibility)."""
        email = str(kwargs.get("email", "test@example.com"))
        return cls.create_test_user(name=name, email=email, **kwargs)

    @classmethod
    def create_concrete_value_object(cls, **kwargs: object) -> FlextResult[TestMoney]:
        """Create concrete value object (alias for create_test_money for backward compatibility)."""
        amount_obj = kwargs.get("amount", "100.00")
        # Coerce to accepted type for create_test_money
        if isinstance(amount_obj, (str, Decimal)):
            normalized_amount: str | Decimal = amount_obj
        else:
            normalized_amount = str(amount_obj)
        currency = str(kwargs.get("currency", "USD"))
        description = str(kwargs.get("description", "test money"))
        return cls.create_test_money(
            amount=normalized_amount,
            currency=currency,
            description=description,
        )

    @classmethod
    def create_complex_value_object(
        cls,
        name: str,
        tags: list[str],
        metadata: dict[str, object],
    ) -> FlextResult[TestComplexValueObject]:
        """Create complex value object for testing."""
        try:
            complex_vo = TestComplexValueObject(name=name, tags=tags, metadata=metadata)

            validation = complex_vo.validate_business_rules()
            if validation.is_failure:
                return FlextResult.fail(
                    f"Complex value object validation failed: {validation.error}",
                )

            return FlextResult.ok(complex_vo)

        except Exception as e:
            return FlextResult.fail(f"Failed to create test complex value object: {e}")


# =============================================================================
# TEST HELPER FUNCTIONS
# =============================================================================


def create_test_user_safe(
    name: str = "Test User",
    email: str = "test@example.com",
    **kwargs: object,
) -> TestUser:
    """Create test user with error handling."""
    result = TestDomainFactory.create_test_user(name, email, **kwargs)
    if result.is_failure:
        raise ValueError(f"Failed to create test user: {result.error}")
    if result.data is None:
        error_msg = "Failed to create test user: result data is None"
        raise ValueError(error_msg)
    return result.data


def create_test_order_safe(
    user_id: TEntityId,
    total_amount: str | Decimal = "100.00",
    **kwargs: object,
) -> TestOrder:
    """Create test order with error handling."""
    result = TestDomainFactory.create_test_order(user_id, total_amount, **kwargs)
    if result.is_failure:
        raise ValueError(f"Failed to create test order: {result.error}")
    if result.data is None:
        error_msg = "Failed to create test order: result data is None"
        raise ValueError(error_msg)
    return result.data


def create_test_product_safe(
    name: str = "Test Product",
    **kwargs: object,
) -> TestProduct:
    """Create test product with error handling."""
    # Normalize optional price for type-compatibility
    price_obj = kwargs.pop("price", "50.00")
    if isinstance(price_obj, (str, Decimal)):
        normalized_price: str | Decimal = price_obj
    else:
        normalized_price = str(price_obj)
    # mypy: kwargs is object; cast to a typed dict for safe ** expansion
    result = TestDomainFactory.create_test_product(
        name=name,
        price=normalized_price,
        **kwargs,
    )
    if result.is_failure:
        raise ValueError(f"Failed to create test product: {result.error}")
    if result.data is None:
        error_msg = "Failed to create test product: result data is None"
        raise ValueError(error_msg)
    return result.data


def log_test_operation(operation: str, entity_type: str, entity_id: str) -> None:
    """Log test domain operations for debugging."""
    # Simple logging for test operations (using debug logging instead of print)

    logger = logging.getLogger("test_domain")
    logger.debug("TEST: %s %s %s", operation, entity_type, entity_id)


# =============================================================================
# EXPORTS
# =============================================================================

__all__: list[str] = [
    "TestAddress",
    "TestComplexValueObject",
    "TestDomainFactory",
    "TestEmailAddress",
    "TestMoney",
    "TestOrder",
    "TestOrderStatus",
    "TestProduct",
    "TestUser",
    "TestUserStatus",
    "create_test_order_safe",
    "create_test_product_safe",
    "create_test_user_safe",
    "log_test_operation",
]
