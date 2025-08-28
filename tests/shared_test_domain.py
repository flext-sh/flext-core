"""Shared Test Domain Module for FLEXT Core Tests.

This module provides reusable test fixtures, domain models, and utilities
that can be shared across all test modules, eliminating code duplication and
promoting consistent testing patterns.

ARCHITECTURE CHANGE: This module now uses test-specific domain models
instead of importing from examples/, eliminating circular import issues.
"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum

from pydantic import Field

from flext_core import FlextEntity, FlextResult, FlextTypes, FlextValue

EntityId = FlextTypes.Domain.EntityId


class TestUserStatus(Enum):
    """Test user status enumeration."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"


class TestUser(FlextEntity):
    """Test user entity for testing."""

    name: str
    email: str
    status: TestUserStatus = TestUserStatus.ACTIVE

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate business rules for test user."""
        min_name_length = 2
        if len(self.name) < min_name_length:
            return FlextResult[None].fail("Name must be at least 2 characters")
        return FlextResult[None].ok(None)


class TestMoney(FlextValue):
    """Test money value object."""

    amount: Decimal
    currency: str = "USD"
    description: str = ""

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate business rules for money."""
        if self.amount < 0:
            return FlextResult[None].fail("Amount cannot be negative")
        currency_length = 3
        if len(self.currency) != currency_length:
            return FlextResult[None].fail("Currency must be 3 characters")
        if self.currency != self.currency.upper():
            return FlextResult[None].fail("Currency must be uppercase")
        return FlextResult[None].ok(None)


class TestComplexValueObject(FlextValue):
    """Test complex value object with multiple fields."""

    name: str
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate business rules for complex value object."""
        if not self.name or not self.name.strip():
            return FlextResult[None].fail("Name cannot be empty")
        return FlextResult[None].ok(None)


class TestDomainFactory:
    """Factory for creating test domain objects."""

    @staticmethod
    def create_user(
        name: str = "Test User", email: str = "test@example.com"
    ) -> TestUser:
        return TestUser(id=EntityId("test-user"), name=name, email=email)

    @staticmethod
    def create_money(amount: float = 100.0, currency: str = "USD") -> TestMoney:
        return TestMoney(amount=Decimal(str(amount)), currency=currency)

    @staticmethod
    def create_concrete_entity(
        name: str = "Test User", status: str = "active"
    ) -> FlextResult[TestUser]:
        """Create a concrete entity with error handling."""
        try:
            user_status = TestUserStatus(status)
            user = TestUser(
                id=EntityId("test-user"),
                name=name,
                email="test@example.com",
                status=user_status,
            )
            return FlextResult[TestUser].ok(user)
        except ValueError as e:
            return FlextResult[TestUser].fail(f"Invalid status: {e}")

    @staticmethod
    def create_concrete_value_object(
        amount: Decimal, currency: str = "USD", description: str = ""
    ) -> FlextResult[TestMoney]:
        """Create a concrete value object with error handling."""
        money = TestMoney(amount=amount, currency=currency, description=description)
        validation_result = money.validate_business_rules()
        if validation_result.is_failure:
            return FlextResult[TestMoney].fail(
                validation_result.error or "Validation failed"
            )
        return FlextResult[TestMoney].ok(money)

    @staticmethod
    def create_complex_value_object(
        name: str, tags: list[str], metadata: dict[str, object]
    ) -> FlextResult[TestComplexValueObject]:
        """Create a complex value object with error handling."""
        try:
            complex_vo = TestComplexValueObject(name=name, tags=tags, metadata=metadata)
            validation_result = complex_vo.validate_business_rules()
            if validation_result.is_failure:
                return FlextResult[TestComplexValueObject].fail(
                    validation_result.error or "Validation failed"
                )
            return FlextResult[TestComplexValueObject].ok(complex_vo)
        except Exception as e:
            return FlextResult[TestComplexValueObject].fail(
                f"Failed to create complex value object: {e}"
            )


# Entity aliases for backward compatibility
ConcreteFlextEntity = TestUser
ConcreteValueObject = TestMoney  # Using TestMoney as a proper value object
ComplexValueObject = TestComplexValueObject  # Using TestComplexValueObject properly


def create_test_entity_safe(name: str, **kwargs: object) -> TestUser:
    """Create test entity with error handling."""
    status = str(kwargs.get("status", "active"))
    result = TestDomainFactory.create_concrete_entity(name=name, status=status)
    if result.is_failure:
        error_msg: str = f"Failed to create test entity: {result.error}"
        raise ValueError(error_msg)
    # if result.value is None:  # Unreachable - FlextResult success implies data is not None
    #     error_msg = "Failed to create test entity: result data is None"
    #     raise ValueError(error_msg)
    return result.value


def create_test_value_object_safe(
    amount: str,
    currency: str = "USD",
    **kwargs: object,
) -> object:
    """Create test value object with error handling."""
    result = TestDomainFactory.create_concrete_value_object(
        amount=Decimal(amount),
        currency=currency,
        description=str(kwargs.get("description", "")),
    )
    if result.is_failure:
        error_msg: str = f"Failed to create test value object: {result.error}"
        raise ValueError(error_msg)
    # if result.value is None:  # Unreachable - FlextResult success implies data is not None
    #     error_msg = "Failed to create test value object: result data is None"
    #     raise ValueError(error_msg)
    return result.value


def create_complex_test_value_object_safe(
    name: str,
    tags: list[str],
    metadata: dict[str, object],
) -> object:
    """Create complex test value object with error handling."""
    result = TestDomainFactory.create_complex_value_object(
        name=name,
        tags=tags,
        metadata=metadata,
    )
    if result.is_failure:
        error_msg: str = f"Failed to create complex test value object: {result.error}"
        raise ValueError(error_msg)
    # if result.value is None:  # Unreachable - FlextResult success implies data is not None
    #     error_msg = "Failed to create complex test value object: result data is None"
    #     raise ValueError(error_msg)
    return result.value


__all__: list[str] = [
    "ComplexValueObject",
    "ConcreteFlextEntity",
    "ConcreteValueObject",
    "TestDomainFactory",
    "TestMoney",
    "TestUser",
    "TestUserStatus",
    "create_complex_test_value_object_safe",
    "create_test_entity_safe",
    "create_test_value_object_safe",
]
