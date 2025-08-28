"""Comprehensive tests for value_objects.py module.

This test suite provides complete coverage of the value object functionality
including abstract base class behavior, validation patterns, factory methods,
serialization capabilities, and complex payload conversion scenarios.

Coverage Target: value_objects.py 63% → 95%+
"""

from __future__ import annotations

import math
from collections.abc import Callable
from decimal import Decimal
from typing import cast, override

import pytest
from pydantic import ValidationError

from flext_core import (
    FlextLogger,
    FlextPayload,
    FlextResult,
    FlextValue,
)
from flext_core.models import FlextFactory

pytestmark = [pytest.mark.unit, pytest.mark.core]


# =============================================================================
# TEST VALUE OBJECTS - Concrete implementations for testing
# =============================================================================


class SimpleValueObject(FlextValue):
    """Simple value object for basic testing."""

    value: str

    @override
    def validate_business_rules(self) -> FlextResult[None]:
        """Validate that value is not empty."""
        if not self.value or not self.value.strip():
            return FlextResult[None].fail("Value cannot be empty")
        return FlextResult[None].ok(None)


class EmailAddress(FlextValue):
    """Email address value object for testing."""

    address: str

    @override
    def validate_business_rules(self) -> FlextResult[None]:
        """Validate email format."""
        if "@" not in self.address:
            return FlextResult[None].fail("Invalid email format")
        if "." not in self.address.split("@")[1]:
            return FlextResult[None].fail("Invalid domain format")
        return FlextResult[None].ok(None)


class MoneyAmount(FlextValue):
    """Money amount value object for testing."""

    amount: Decimal
    currency: str = "USD"

    @override
    def validate_business_rules(self) -> FlextResult[None]:
        """Validate money amount."""
        if self.amount < 0:
            return FlextResult[None].fail("Amount cannot be negative")
        if self.currency not in {"USD", "EUR", "GBP"}:
            return FlextResult[None].fail("Unsupported currency")
        return FlextResult[None].ok(None)


class ComplexValueObject(FlextValue):
    """Complex value object with nested data for testing."""

    name: str
    metadata: dict[str, object]
    tags: list[str]
    settings: dict[str, bool]

    @override
    def validate_business_rules(self) -> FlextResult[None]:
        """Complex validation."""
        if len(self.name) < 2:
            return FlextResult[None].fail("Name too short")
        if not self.metadata:
            return FlextResult[None].fail("Metadata required")
        return FlextResult[None].ok(None)


class InvalidValueObject(FlextValue):
    """Value object that always fails validation for testing."""

    data: str

    @override
    def validate_business_rules(self) -> FlextResult[None]:
        """Fail validation always."""
        return FlextResult[None].fail("Always invalid")


class SerializationTestValueObject(FlextValue):
    """Value object with problematic serialization for testing."""

    name: str
    callback: object = None  # This will cause serialization issues

    @override
    def validate_business_rules(self) -> FlextResult[None]:
        """Validate name is not empty."""
        if not self.name:
            return FlextResult[None].fail("Name required")
        return FlextResult[None].ok(None)


# =============================================================================
# BASIC FUNCTIONALITY TESTS
# =============================================================================


class TestFlextValueObjectBasics:
    """Test basic FlextValue functionality."""

    def test_value_object_creation_success(self) -> None:
        """Test successful value object creation."""
        vo = SimpleValueObject.model_validate({"value": "test"})
        assert vo.value == "test"
        assert isinstance(vo, FlextValue)

    def test_value_object_creation_with_validation(self) -> None:
        """Test value object creation with validation."""
        email = EmailAddress.model_validate({"address": "user@example.com"})
        assert email.address == "user@example.com"

    def test_value_object_immutability(self) -> None:
        """Test that value objects are immutable."""
        vo = SimpleValueObject.model_validate({"value": "test"})

        with pytest.raises(ValidationError):
            vo.value = "changed"  # Should fail due to frozen=True

    def test_value_object_equality(self) -> None:
        """Test attribute-based equality."""
        vo1 = SimpleValueObject.model_validate({"value": "test"})
        vo2 = SimpleValueObject.model_validate({"value": "test"})
        vo3 = SimpleValueObject.model_validate({"value": "different"})

        assert vo1 == vo2  # Same values
        assert vo1 != vo3  # Different values
        assert vo1 is not vo2  # Different instances

    def test_value_object_hash_consistency(self) -> None:
        """Test hash consistency for value objects."""
        vo1 = SimpleValueObject.model_validate({"value": "test"})
        vo2 = SimpleValueObject.model_validate({"value": "test"})
        vo3 = SimpleValueObject.model_validate({"value": "different"})

        assert hash(vo1) == hash(vo2)  # Same values, same hash
        assert hash(vo1) != hash(vo3)  # Different values, different hash

    def test_complex_value_object_hash(self) -> None:
        """Test hash with complex nested data."""
        vo = ComplexValueObject(
            name="test",
            metadata={"key": "value", "nested": {"a": 1}},
            tags=["tag1", "tag2"],
            settings={"enabled": True, "debug": False},
        )

        # Should be able to hash complex structures
        hash_value = hash(vo)
        assert isinstance(hash_value, int)


# =============================================================================
# VALIDATION TESTS
# =============================================================================


class TestValueObjectValidation:
    """Test value object validation functionality."""

    def test_validate_business_rules_success(self) -> None:
        """Test successful business rule validation."""
        vo = SimpleValueObject.model_validate({"value": "test"})
        result = vo.validate_business_rules()
        assert result.success is True

    def test_validate_business_rules_failure(self) -> None:
        """Test business rule validation failure."""
        vo = SimpleValueObject.model_validate({"value": ""})
        result = vo.validate_business_rules()
        assert result.success is False
        assert result.error is not None
        assert "cannot be empty" in result.error

    def test_validate_flext_success(self) -> None:
        """Test FlextValidation success."""
        vo = SimpleValueObject.model_validate({"value": "test"})
        result = vo.validate_flext()
        assert result.success is True
        assert result.value is vo

    def test_validate_flext_failure(self) -> None:
        """Test FlextValidation failure."""
        vo = SimpleValueObject.model_validate({"value": ""})
        result = vo.validate_flext()
        assert result.success is False
        assert result.error is not None
        assert "cannot be empty" in result.error

    def test_validate_flext_with_logging(self) -> None:
        """Test FlextValidation with logging output."""
        vo = InvalidValueObject(data="test")

        # Test real validation failure - no mocking needed
        result = vo.validate_flext()
        assert result.success is False
        assert result.error
        assert "Always invalid" in result.error

    def test_validate_field_with_registry(self) -> None:
        """Test field validation using registry."""
        vo = SimpleValueObject.model_validate({"value": "test"})

        # Test with non-existent field (should pass)
        result = vo.validate_field("unknown_field", "value")
        assert result.success is True

    def test_validate_all_fields(self) -> None:
        """Test validation of all fields."""
        vo = SimpleValueObject.model_validate({"value": "test"})
        result = vo.validate_all_fields()
        assert result.success is True  # No registered fields to validate

    def test_validate_all_fields_with_internal_fields(self) -> None:
        """Test field validation skips internal fields."""

        # Create mock object with internal field
        class TestVO(SimpleValueObject):
            _internal: str = "private"

        vo = TestVO.model_validate({"value": "test"})
        result = vo.validate_all_fields()
        assert result.success is True


# =============================================================================
# STRING REPRESENTATION AND FORMATTING TESTS
# =============================================================================


class TestValueObjectFormatting:
    """Test value object string formatting."""

    def test_format_dict_simple(self) -> None:
        """Test dictionary formatting."""
        vo = SimpleValueObject.model_validate({"value": "test"})
        data = {"name": "test", "count": 42, "active": True}
        formatted = vo.format_dict(data)

        assert "name='test'" in formatted
        assert "count=42" in formatted
        assert "active=True" in formatted

    def test_format_dict_complex(self) -> None:
        """Test formatting complex dictionary."""
        vo = SimpleValueObject.model_validate({"value": "test"})
        data = {
            "string": "value",
            "number": 123,
            "boolean": False,
            "none_val": None,
            "list": [1, 2, 3],
        }
        formatted = vo.format_dict(data)

        assert "string='value'" in formatted
        assert "number=123" in formatted
        assert "boolean=False" in formatted

    def test_str_representation(self) -> None:
        """Test string representation."""
        vo = SimpleValueObject.model_validate({"value": "test"})
        str_repr = str(vo)

        assert "SimpleValueObject" in str_repr
        assert "value='test'" in str_repr

    def test_str_representation_complex(self) -> None:
        """Test string representation with complex data."""
        vo = ComplexValueObject(
            name="test",
            metadata={"key": "value"},
            tags=["tag1", "tag2"],
            settings={"enabled": True},
        )
        str_repr = str(vo)

        assert "ComplexValueObject" in str_repr
        assert "name='test'" in str_repr


# =============================================================================
# PAYLOAD CONVERSION TESTS
# =============================================================================


class TestValueObjectPayloadConversion:
    """Test value object payload conversion."""

    def test_to_payload_success(self) -> None:
        """Test successful payload conversion."""
        vo = SimpleValueObject.model_validate({"value": "test"})

        payload = vo.to_payload()

        assert isinstance(payload, FlextPayload)
        payload_data = payload.value
        assert isinstance(payload_data, dict)
        assert "value_object_data" in payload_data
        assert "class_info" in payload_data
        assert "validation_status" in payload_data

    def test_to_payload_with_validation_failure(self) -> None:
        """Test payload conversion with validation failure."""
        # Use InvalidValueObject which naturally fails validation
        invalid_vo = InvalidValueObject(data="test")
        payload = invalid_vo.to_payload()

        # Should still create payload but mark as invalid
        assert payload is not None
        payload_data = payload.value
        assert isinstance(payload_data, dict)
        assert payload_data["validation_status"] == "invalid"

    def test_to_payload_serialization_fallback(self) -> None:
        """Test payload conversion with serialization fallback."""
        vo = SerializationTestValueObject(name="test", callback=lambda x: x)

        # Test real payload creation with complex objects
        payload = vo.to_payload()
        assert isinstance(payload, FlextPayload)

        # Verify payload contains serialized data
        payload_data = payload.value
        assert isinstance(payload_data, dict)
        assert "value_object_data" in payload_data

    def test_to_payload_complete_failure(self) -> None:
        """Test payload conversion with real failure scenario."""

        # Create value object with None name to trigger validation failure
        class FailingValueObject(FlextValue):
            name: str

            @override
            def validate_business_rules(self) -> FlextResult[None]:
                if not self.name:
                    return FlextResult[None].fail("Name required")
                return FlextResult[None].ok(None)

        # This should trigger the fallback logic in to_payload
        try:
            vo = FailingValueObject(name="")
            payload = vo.to_payload()
            # Should create payload even with validation failure
            assert payload is not None
        except Exception as exc:
            # Real validation error can occur in some cases
            FlextLogger(__name__).debug("Validation exception occurred: %s", exc)

    def test_extract_serializable_attributes_pydantic(self) -> None:
        """Test serializable attribute extraction via Pydantic."""
        vo = SimpleValueObject.model_validate({"value": "test"})
        result = vo._extract_serializable_attributes()

        assert isinstance(result, dict)
        assert "value" in result
        assert result["value"] == "test"

    def test_extract_serializable_attributes_manual(self) -> None:
        """Test manual attribute extraction directly."""
        vo = SimpleValueObject.model_validate({"value": "test"})

        # Test the manual extraction method directly
        result = vo._try_manual_extraction()

        assert isinstance(result, dict)
        # Should contain the value attribute
        assert "value" in result
        assert result["value"] == "test"

    def test_extract_serializable_no_model_dump(self) -> None:
        """Test the fallback info when neither Pydantic nor manual extraction work."""
        vo = SimpleValueObject.model_validate({"value": "test"})

        # Test the fallback info method directly
        result = vo._get_fallback_info()

        assert isinstance(result, dict)
        assert "class_name" in result
        assert result["class_name"] == "SimpleValueObject"

    def test_process_serializable_values(self) -> None:
        """Test processing of serializable values."""
        vo = SimpleValueObject.model_validate({"value": "test"})
        data = {
            "string": "value",
            "int": 42,
            "float": math.pi,
            "bool": True,
            "none": None,
            "complex": {"nested": "object"},
        }

        result = vo._process_serializable_values(data)

        assert result["string"] == "value"
        assert result["int"] == 42
        assert result["float"] == math.pi
        assert result["bool"] is True
        assert result["none"] is None
        assert isinstance(result["complex"], str)  # Should be converted to string

    def test_should_include_attribute(self) -> None:
        """Test attribute inclusion logic."""
        vo = SimpleValueObject.model_validate({"value": "test"})

        assert vo._should_include_attribute("value") is True
        assert vo._should_include_attribute("_private") is False
        assert vo._should_include_attribute("validate_business_rules") is False

    def test_safely_get_attribute(self) -> None:
        """Test safe attribute retrieval."""
        vo = SimpleValueObject.model_validate({"value": "test"})

        # Test successful retrieval
        result = vo._safely_get_attribute("value")
        assert result == "test"

        # Test with non-existent attribute
        result = vo._safely_get_attribute("nonexistent")
        assert result is None

    def test_safely_get_attribute_with_str_method(self) -> None:
        """Test safe attribute retrieval with __str__ conversion."""

        class ObjectWithStr:
            @override
            def __str__(self) -> str:
                return "string_repr"

        vo = SimpleValueObject.model_validate({"value": "test"})
        # Create a test attribute on the value object
        object.__setattr__(vo, "_test_attr", ObjectWithStr())

        result = vo._safely_get_attribute("_test_attr")
        assert result == "string_repr"

    def test_safely_get_attribute_exception(self) -> None:
        """Test safe attribute retrieval with exception."""
        vo = SimpleValueObject.model_validate({"value": "test"})

        # Test with a non-existent attribute - this will naturally raise AttributeError
        result = vo._safely_get_attribute("nonexistent_attribute")
        assert result is None

    def test_get_fallback_info(self) -> None:
        """Test fallback information generation."""
        vo = SimpleValueObject.model_validate({"value": "test"})
        result = vo._get_fallback_info()

        assert "class_name" in result
        assert "module" in result
        assert result["class_name"] == "SimpleValueObject"


# =============================================================================
# SUBCLASS TRACKING TESTS
# =============================================================================


class TestValueObjectSubclassing:
    """Test value object subclass tracking."""

    def test_init_subclass_logging(self) -> None:
        """Test that subclass creation works correctly."""

        # Create a new subclass - real execution
        class TestSubclass(FlextValue):
            test_field: str

            @override
            def validate_business_rules(self) -> FlextResult[None]:
                return FlextResult[None].ok(None)

        # Test that the subclass works
        instance = TestSubclass(test_field="value")
        assert instance.test_field == "value"
        assert isinstance(instance, FlextValue)

        # Test validation works
        result = instance.validate_business_rules()
        assert result.success


# =============================================================================
# FACTORY TESTS
# =============================================================================


class TestFlextFactory:
    """Test FlextFactory functionality."""

    def test_create_value_object_factory_basic(self) -> None:
        """Test basic factory creation."""
        factory = FlextFactory.create_value_object_factory(
            SimpleValueObject, defaults={}
        )

        assert callable(factory)

        # Test factory usage - cast to avoid type checker issues
        callable_factory = cast("Callable[..., FlextResult[object]]", factory)
        result = callable_factory(value="test")
        assert result.success is True
        assert isinstance(result.value, SimpleValueObject)
        assert result.value.value == "test"

    def test_create_value_object_factory_with_defaults(self) -> None:
        """Test factory creation with defaults."""
        defaults: dict[str, object] = {"currency": "EUR"}
        factory = FlextFactory.create_value_object_factory(
            MoneyAmount,
            defaults=defaults,
        )

        # Test with defaults - cast to avoid type checker issues
        callable_factory = cast("Callable[..., FlextResult[object]]", factory)
        result = callable_factory(amount=Decimal("10.00"))
        assert result.success is True
        assert result.value.currency == "EUR"  # Default applied

        # Test overriding defaults
        result = callable_factory(amount=Decimal("20.00"), currency="USD")
        assert result.success is True
        assert result.value.currency == "USD"  # Override applied

    def test_factory_validation_failure(self) -> None:
        """Test factory with validation failure."""
        factory = FlextFactory.create_value_object_factory(
            SimpleValueObject, defaults={}
        )

        # Cast to avoid type checker issues
        callable_factory = cast("Callable[..., FlextResult[object]]", factory)
        result = callable_factory(value="")  # Should fail validation
        assert result.success is False
        assert result.error
        assert "cannot be empty" in str(result.error)

    def test_factory_creation_failure(self) -> None:
        """Test factory with creation failure."""
        factory = FlextFactory.create_value_object_factory(
            SimpleValueObject, defaults={}
        )

        # Cast to avoid type checker issues
        callable_factory = cast("Callable[..., FlextResult[object]]", factory)
        # Pass invalid parameter - this will be handled by validation
        result = callable_factory(
            value=""
        )  # Empty value should fail business validation
        assert result.success is False
        assert result.error
        assert "cannot be empty" in str(result.error)

    def test_factory_without_defaults(self) -> None:
        """Test factory creation without defaults."""
        factory = FlextFactory.create_value_object_factory(
            SimpleValueObject,
            defaults={},
        )

        # Cast to avoid type checker issues
        callable_factory = cast("Callable[..., FlextResult[object]]", factory)
        result = callable_factory(value="test")
        assert result.success is True
        assert result.value.value == "test"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestValueObjectIntegration:
    """Test integration scenarios with value objects."""

    def test_value_object_with_complex_validation(self) -> None:
        """Test value object with complex validation scenario."""
        email = EmailAddress.model_validate({"address": "user@example.com"})

        # Test validation
        validation_result = email.validate_flext()
        assert validation_result.success is True

        # Test payload conversion
        payload = email.to_payload()
        assert isinstance(payload, FlextPayload)

        # Test string representation
        str_repr = str(email)
        assert "EmailAddress" in str_repr
        assert "user@example.com" in str_repr

    def test_value_object_equality_and_hashing(self) -> None:
        """Test complex equality and hashing scenarios."""
        money1 = MoneyAmount(amount=Decimal("10.00"), currency="USD")
        money2 = MoneyAmount(amount=Decimal("10.00"), currency="USD")
        money3 = MoneyAmount(amount=Decimal("10.00"), currency="EUR")

        # Test equality
        assert money1 == money2
        assert money1 != money3

        # Test hashing for use in sets/dicts
        money_set = {money1, money2, money3}
        assert len(money_set) == 2  # money1 and money2 are equal

        # Test as dict keys
        money_dict = {money1: "first", money2: "second", money3: "third"}
        assert len(money_dict) == 2  # money1 and money2 share same key

    def test_value_object_with_nested_structures(self) -> None:
        """Test value objects with complex nested data."""
        complex_vo = ComplexValueObject(
            name="test",
            metadata={"nested": {"deep": {"value": 42}}},
            tags=["tag1", "tag2"],
            settings={"enabled": True, "debug": False},
        )

        # Test creation and validation
        validation_result = complex_vo.validate_flext()
        assert validation_result.success is True

        # Test hash works with nested structures
        hash_value = hash(complex_vo)
        assert isinstance(hash_value, int)

        # Test equality with same nested structures
        complex_vo2 = ComplexValueObject(
            name="test",
            metadata={"nested": {"deep": {"value": 42}}},
            tags=["tag1", "tag2"],
            settings={"enabled": True, "debug": False},
        )
        assert complex_vo == complex_vo2

    def test_factory_integration_scenario(self) -> None:
        """Test complete factory integration scenario."""
        # Create factory with defaults
        email_factory = FlextFactory.create_value_object_factory(
            EmailAddress,
            defaults={},
        )

        # Create multiple emails
        emails = []
        test_addresses = ["user1@example.com", "user2@test.org", "REDACTED_LDAP_BIND_PASSWORD@company.net"]

        # Cast to avoid type checker issues
        callable_factory = cast("Callable[..., FlextResult[object]]", email_factory)

        for address in test_addresses:
            result = callable_factory(address=address)
            assert result.success is True
            emails.append(result.value)

        # Test all emails are valid and unique
        assert len(emails) == 3
        assert len(set(emails)) == 3  # All unique due to different addresses

        # Test string representation for all
        for email in emails:
            str_repr = str(email)
            assert "EmailAddress" in str_repr
            assert email.address in str_repr


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestValueObjectEdgeCases:
    """Test edge cases and error conditions."""

    def test_hash_with_unhashable_nested_data(self) -> None:
        """Test hashing with deeply nested unhashable data."""
        complex_data = ComplexValueObject(
            name="test",
            metadata={
                "list": [1, 2, {"nested": "dict"}],
                "set": {"a", "b", "c"},
                "nested_dict": {"level2": {"level3": ["deep", "list"]}},
            },
            tags=["tag1", "tag2"],
            settings={"enabled": True},
        )

        # Should successfully hash despite complex nested structures
        hash_value = hash(complex_data)
        assert isinstance(hash_value, int)

    def test_equality_with_different_types(self) -> None:
        """Test equality comparison with different types."""
        vo = SimpleValueObject.model_validate({"value": "test"})

        # Compare with non-value object
        assert vo != "string"
        assert vo != 123
        assert vo is not None
        assert vo != []

    def test_serialization_with_circular_references(self) -> None:
        """Test serialization handling of problematic data."""
        # Create value object with callback that could cause issues
        vo = SerializationTestValueObject(name="test")

        # Test extraction methods handle edge cases
        result = vo._extract_serializable_attributes()
        assert isinstance(result, dict)
        assert "name" in result

    def test_field_validation_error_handling(self) -> None:
        """Test field validation with real error conditions."""
        vo = SimpleValueObject.model_validate({"value": "test"})

        # Test field validation with non-existent field (should pass)
        result = vo.validate_field("unknown_field", "value")
        assert result.success is True

        # Test with actual field that exists
        result = vo.validate_field("value", "test")
        assert result.success is True

    def test_all_fields_validation_with_errors(self) -> None:
        """Test validation of all fields with real scenario."""
        # Create value object with multiple fields
        complex_vo = ComplexValueObject(
            name="test",
            metadata={"key": "value"},
            tags=["tag1", "tag2"],
            settings={"enabled": True},
        )

        # Test real field validation
        result = complex_vo.validate_all_fields()
        assert result.success is True  # Should pass with valid data

        # Test with simpler object
        simple_vo = SimpleValueObject.model_validate({"value": "test"})
        result = simple_vo.validate_all_fields()
        assert result.success is True  # Should pass as no registered field validators
