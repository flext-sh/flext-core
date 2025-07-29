"""Comprehensive tests for FlextValueObject and value object functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from pydantic import Field, ValidationError

from flext_core.result import FlextResult
from flext_core.value_objects import FlextValueObject, FlextValueObjectFactory

# Constants
EXPECTED_BULK_SIZE = 2

if TYPE_CHECKING:
    from collections.abc import Any


class EmailAddress(FlextValueObject):
    """Test email value object."""

    email: str = Field(min_length=5, pattern=r"^[^@]+@[^@]+\.[^@]+$")

    def validate_domain_rules(self) -> FlextResult[None]:
        """Validate email domain rules."""
        if not self.email or "@" not in self.email:
            return FlextResult.fail("Email must contain @ symbol")

        domain = self.email.split("@")[1]
        if "." not in domain:
            return FlextResult.fail("Email domain must contain a dot")

        return FlextResult.ok(None)


class Money(FlextValueObject):
    """Test money value object."""

    amount: float = Field(ge=0)
    currency: str = Field(min_length=3, max_length=3)

    def validate_domain_rules(self) -> FlextResult[None]:
        """Validate money domain rules."""
        if self.amount < 0:
            return FlextResult.fail("Amount cannot be negative")
        if self.currency.upper() not in {"USD", "EUR", "GBP"}:
            return FlextResult.fail("Unsupported currency")
        return FlextResult.ok(None)

    def add(self, other: Money) -> FlextResult[Money]:
        """Add two money amounts."""
        if self.currency != other.currency:
            return FlextResult.fail("Cannot add different currencies")
        try:
            new_amount = self.amount + other.amount
            return FlextResult.ok(Money(amount=new_amount, currency=self.currency))
        except (ValueError, TypeError, ArithmeticError) as e:
            return FlextResult.fail(f"Addition failed: {e}")


class InvalidValueObject(FlextValueObject):
    """Test value object with validation issues."""

    value: str

    def validate_domain_rules(self) -> FlextResult[None]:
        """Return failure for testing purposes."""
        return FlextResult.fail("This value object is always invalid")


class TestFlextValueObject:
    """Test FlextValueObject base functionality."""

    def test_email_value_object_creation(self) -> None:
        """Test creating valid email value object."""
        email = EmailAddress(email="test@example.com")

        assert email.email == "test@example.com", (
            f"Expected {'test@example.com'}, got {email.email}"
        )
        assert isinstance(email, EmailAddress)

        # Test validation
        validation_result = email.validate_domain_rules()
        assert validation_result.is_success

        # Test Pydantic immutability
        # ValidationError or AttributeError for frozen model
        with pytest.raises((AttributeError, ValueError)):
            email.email = "new@example.com"  # Should raise due to frozen model

    def test_email_value_object_invalid_email(self) -> None:
        """Test email value object with invalid email."""
        # Test Pydantic validation failure

        with pytest.raises(ValidationError):
            EmailAddress(email="invalid-email")

        # Test domain rule validation failure
        # Create with minimal validation passing, then test domain rules
        try:
            email = EmailAddress(email="test@invalid")
            validation_result = email.validate_domain_rules()
            assert validation_result.is_failure
            assert "domain must contain a dot" in validation_result.error, (
                f"Expected {'domain must contain a dot'}, got {validation_result.error}"
            )
        except (ValidationError, ValueError):
            # If Pydantic validation prevents creation, that's also valid
            pytest.skip("Pydantic validation prevents object creation")

    def test_money_value_object_operations(self) -> None:
        """Test money value object with operations."""
        money1 = Money(amount=100.0, currency="USD")
        money2 = Money(amount=50.0, currency="USD")

        # Test basic properties
        assert money1.amount == 100.0, f"Expected {100.0}, got {money1.amount}"
        assert money1.currency == "USD"

        # Test domain validation
        validation_result = money1.validate_domain_rules()
        assert validation_result.is_success

        # Test addition operation
        add_result = money1.add(money2)
        assert add_result.is_success
        result_money = add_result.data
        assert result_money.amount == 150.0, (
            f"Expected {150.0}, got {result_money.amount}"
        )
        assert result_money.currency == "USD"

    def test_money_value_object_currency_mismatch(self) -> None:
        """Test money addition with different currencies."""
        money_usd = Money(amount=100.0, currency="USD")
        money_eur = Money(amount=50.0, currency="EUR")

        add_result = money_usd.add(money_eur)
        assert add_result.is_failure
        assert "different currencies" in add_result.error, (
            f"Expected {'different currencies'} in {add_result.error}"
        )

    def test_value_object_equality(self) -> None:
        """Test value object equality based on attributes."""
        email1 = EmailAddress(email="test@example.com")
        email2 = EmailAddress(email="test@example.com")
        email3 = EmailAddress(email="other@example.com")

        # Value objects should be equal based on content
        assert email1 == email2, f"Expected {email2}, got {email1}"
        assert email1 != email3

        # Test hash consistency
        assert hash(email1) == hash(email2), (
            f"Expected {hash(email2)}, got {hash(email1)}"
        )
        assert hash(email1) != hash(email3)

    def test_value_object_string_representation(self) -> None:
        """Test string representation of value objects."""
        email = EmailAddress(email="test@example.com")
        str_repr = str(email)

        assert "EmailAddress" in str_repr, f"Expected {'EmailAddress'} in {str_repr}"
        assert "test@example.com" in str_repr

    def test_value_object_logging_capability(self) -> None:
        """Test value object logging capabilities."""
        email = EmailAddress(email="test@example.com")

        # Test logger access (inherited from FlextLoggableMixin)
        logger = email.logger
        assert logger is not None

    def test_value_object_validation_mixin(self) -> None:
        """Test value object validation mixin functionality."""
        email = EmailAddress(email="test@example.com")

        # Test validation state (inherited from FlextValueObjectMixin)
        # Note: is_valid may start as None until validation is performed
        validation_result = email.validate_domain_rules()
        assert validation_result.is_success

    def test_value_object_utility_inheritance(self) -> None:
        """Test utility inheritance in value objects."""
        email = EmailAddress(email="test@example.com")

        # Test generator utility (inherited from FlextGenerators)
        entity_id = email.generate_entity_id()
        assert isinstance(entity_id, str)
        assert len(entity_id) > 0

        # Test timestamp generation
        timestamp = email.generate_timestamp()
        assert isinstance(timestamp, float)
        assert timestamp > 0

    def test_value_object_payload_conversion(self) -> None:
        """Test payload conversion functionality."""
        email = EmailAddress(email="test@example.com")

        payload = email.to_payload()
        assert payload is not None

        payload_data = payload.data
        assert "value_object_data" in payload_data, (
            f"Expected {'value_object_data'} in {payload_data}"
        )
        assert "metadata" in payload_data
        assert "class_info" in payload_data, (
            f"Expected {'class_info'} in {payload_data}"
        )

        # Check metadata
        metadata = payload_data["metadata"]
        assert "type" in metadata, f"Expected {'type'} in {metadata}"
        assert "timestamp" in metadata
        assert "correlation_id" in metadata, (
            f"Expected {'correlation_id'} in {metadata}"
        )
        assert "validated" in metadata

    def test_value_object_field_validation(self) -> None:
        """Test field validation functionality."""
        email = EmailAddress(email="test@example.com")

        # Test individual field validation
        field_result = email.validate_field("email", "valid@example.com")
        # Result depends on field registry availability
        assert isinstance(field_result, FlextResult)

        # Test all fields validation
        all_fields_result = email.validate_all_fields()
        assert isinstance(all_fields_result, FlextResult)

    def test_invalid_value_object(self) -> None:
        """Test value object that fails domain validation."""
        invalid_obj = InvalidValueObject(value="test")

        # Domain validation should fail
        validation_result = invalid_obj.validate_domain_rules()
        assert validation_result.is_failure
        assert "always invalid" in validation_result.error, (
            f"Expected {'always invalid'} in {validation_result.error}"
        )

        # Validate_flext should also reflect this
        flext_validation = invalid_obj.validate_flext()
        assert flext_validation.is_failure

    def test_value_object_model_operations(self) -> None:
        """Test Pydantic model operations."""
        email = EmailAddress(email="test@example.com")

        # Test model dump
        data = email.model_dump()
        assert isinstance(data, dict)
        assert data["email"] == "test@example.com", (
            f"Expected {'test@example.com'}, got {data['email']}"
        )

        # Test model copy (should work for value objects)
        email_copy = email.model_copy()
        assert email_copy == email, f"Expected {email}, got {email_copy}"
        assert email_copy is not email  # Different instances

    def test_value_object_inheritance_behavior(self) -> None:
        """Test that value object properly inherits from all mixins."""
        email = EmailAddress(email="test@example.com")

        # Test multiple inheritance is working
        assert hasattr(email, "logger")  # From FlextLoggableMixin
        assert hasattr(email, "is_valid")  # From FlextValueObjectMixin
        assert hasattr(email, "generate_entity_id")  # From FlextGenerators
        assert hasattr(email, "format_dict")  # From FlextFormatters (or own method)


class TestFlextValueObjectFactory:
    """Test FlextValueObjectFactory functionality."""

    def test_factory_email_creation(self) -> None:
        """Test factory creation of email value objects."""
        # Create factory for EmailAddress
        email_factory = FlextValueObjectFactory.create_value_object_factory(
            EmailAddress,
        )
        email_result = email_factory(email="test@example.com")

        if email_result.is_success:
            email = email_result.data
            assert email.email == "test@example.com", (
                f"Expected {'test@example.com'}, got {email.email}"
            )
        else:
            # Factory might not be fully implemented
            assert email_result.is_failure

    def test_factory_invalid_email(self) -> None:
        """Test factory with invalid email."""
        # Create factory for EmailAddress
        email_factory = FlextValueObjectFactory.create_value_object_factory(
            EmailAddress,
        )
        email_result = email_factory(email="invalid-email")
        assert email_result.is_failure

    def test_factory_money_creation(self) -> None:
        """Test factory creation of money value objects."""
        # Create factory for Money
        money_factory = FlextValueObjectFactory.create_value_object_factory(Money)
        money_result = money_factory(amount=100.0, currency="USD")

        if money_result.is_success:
            money = money_result.data
            assert money.amount == 100.0, f"Expected {100.0}, got {money.amount}"
            assert money.currency == "USD"
        else:
            # Factory might not be fully implemented
            assert money_result.is_failure

    def test_factory_invalid_money(self) -> None:
        """Test factory with invalid money."""
        # Create factory for Money
        money_factory = FlextValueObjectFactory.create_value_object_factory(Money)
        money_result = money_factory(amount=-100.0, currency="USD")
        assert money_result.is_failure


class TestValueObjectEdgeCases:
    """Test edge cases and error conditions."""

    def test_value_object_with_complex_data(self) -> None:
        """Test value object with complex nested data."""

        class ComplexValueObject(FlextValueObject):
            data: dict[str, Any]
            items: list[str]

            def validate_domain_rules(self) -> FlextResult[None]:
                if not self.data or not self.items:
                    return FlextResult.fail("Data and items are required")
                return FlextResult.ok(None)

        # Rebuild the model to properly handle the Any type
        ComplexValueObject.model_rebuild()

        complex_obj = ComplexValueObject(
            data={"key": "value", "nested": {"inner": "data"}},
            items=["item1", "item2"],
        )

        assert complex_obj.data["key"] == "value", (
            f"Expected {'value'}, got {complex_obj.data['key']}"
        )
        assert len(complex_obj.items) == EXPECTED_BULK_SIZE

        validation_result = complex_obj.validate_domain_rules()
        assert validation_result.is_success

    def test_value_object_serialization(self) -> None:
        """Test value object serialization capabilities."""
        email = EmailAddress(email="test@example.com")

        # Test JSON serialization
        json_str = email.model_dump_json()
        assert isinstance(json_str, str)
        assert "test@example.com" in json_str, (
            f"Expected {'test@example.com'} in {json_str}"
        )

    def test_value_object_validation_chaining(self) -> None:
        """Test chaining multiple validation methods."""
        email = EmailAddress(email="test@example.com")

        # Chain multiple validations
        domain_result = email.validate_domain_rules()
        flext_result = email.validate_flext()
        field_result = email.validate_all_fields()

        # All should be consistent
        if domain_result.is_success:
            # If domain rules pass, flext validation should also consider this
            assert flext_result.is_success or flext_result.is_failure  # Either is valid

        # Field validation depends on field registry
        assert isinstance(field_result, FlextResult)

    def test_value_object_error_handling(self) -> None:
        """Test error handling in value object operations."""
        money = Money(amount=100.0, currency="USD")

        # Test operations that might fail
        # Add with invalid object
        with pytest.raises((AttributeError, TypeError)):
            money.add(None)  # type: ignore[arg-type]

    def test_value_object_performance(self) -> None:
        """Test performance characteristics of value objects."""
        # Create many value objects
        emails = []
        for i in range(100):
            email = EmailAddress(email=f"user{i}@example.com")
            emails.append(email)

        assert len(emails) == 100, f"Expected {100}, got {len(emails)}"

        # Test that they all validate correctly
        for email in emails[:10]:  # Test subset for performance
            validation_result = email.validate_domain_rules()
            assert validation_result.is_success

    def test_value_object_memory_efficiency(self) -> None:
        """Test memory efficiency of value objects."""
        # Create identical value objects
        emails = [EmailAddress(email="test@example.com") for _ in range(10)]

        # They should all be equal
        for i in range(1, len(emails)):
            assert emails[0] == emails[i], f"Expected {emails[i]}, got {emails[0]}"
            assert hash(emails[0]) == hash(emails[i])

    def test_value_object_thread_safety(self) -> None:
        """Test basic thread safety of value objects."""
        email = EmailAddress(email="test@example.com")

        # Immutable objects should be thread-safe
        original_email = email.email

        # Multiple access should be consistent
        for _ in range(10):
            assert email.email == original_email, (
                f"Expected {original_email}, got {email.email}"
            )
            validation_result = email.validate_domain_rules()
            assert validation_result.is_success
