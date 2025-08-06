"""Focused tests for value_objects.py module targeting coverage gaps.

This test suite focuses on testing the core value object functionality
without complex payload conversion to achieve high coverage of the
value_objects.py module.

Coverage Target: value_objects.py 63% → 95%+
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Any
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from flext_core.result import FlextResult
from flext_core.value_objects import FlextValueObject, FlextValueObjectFactory

pytestmark = [pytest.mark.unit, pytest.mark.core]


# =============================================================================
# TEST VALUE OBJECTS - Concrete implementations for testing
# =============================================================================


class SimpleValueObject(FlextValueObject):
    """Simple value object for basic testing."""

    value: str

    def validate_business_rules(self) -> FlextResult[None]:
        """Simple validation - value cannot be empty."""
        if not self.value or not self.value.strip():
            return FlextResult.fail("Value cannot be empty")
        return FlextResult.ok(None)


class EmailAddress(FlextValueObject):
    """Email address value object for testing."""

    address: str

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate email format."""
        if "@" not in self.address:
            return FlextResult.fail("Invalid email format")
        if "." not in self.address.split("@")[1]:
            return FlextResult.fail("Invalid domain format")
        return FlextResult.ok(None)


class MoneyAmount(FlextValueObject):
    """Money amount value object for testing."""

    amount: Decimal
    currency: str = "USD"

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate money amount."""
        if self.amount < 0:
            return FlextResult.fail("Amount cannot be negative")
        if self.currency not in {"USD", "EUR", "GBP"}:
            return FlextResult.fail("Unsupported currency")
        return FlextResult.ok(None)


class ComplexValueObject(FlextValueObject):
    """Complex value object with nested data for testing."""

    name: str
    metadata: dict[str, Any]
    tags: list[str]
    settings: dict[str, bool]

    def validate_business_rules(self) -> FlextResult[None]:
        """Complex validation."""
        if len(self.name) < 2:
            return FlextResult.fail("Name too short")
        if not self.metadata:
            return FlextResult.fail("Metadata required")
        return FlextResult.ok(None)


# =============================================================================
# CORE FUNCTIONALITY TESTS
# =============================================================================


class TestValueObjectCore:
    """Test core value object functionality."""

    def test_value_object_creation_and_validation(self) -> None:
        """Test value object creation with validation."""
        vo = SimpleValueObject(value="test")
        assert vo.value == "test"

        result = vo.validate_business_rules()
        assert result.success is True

    def test_value_object_immutability(self) -> None:
        """Test that value objects are frozen/immutable."""
        vo = SimpleValueObject(value="test")

        with pytest.raises(ValidationError):
            vo.value = "changed"  # Should fail due to frozen=True

    def test_value_object_hash_and_equality(self) -> None:
        """Test hash and equality for value objects."""
        vo1 = SimpleValueObject(value="test")
        vo2 = SimpleValueObject(value="test")
        vo3 = SimpleValueObject(value="different")

        # Test equality
        assert vo1 == vo2  # Same values
        assert vo1 != vo3  # Different values
        assert vo1 is not vo2  # Different instances

        # Test hash consistency
        assert hash(vo1) == hash(vo2)  # Same values, same hash
        assert hash(vo1) != hash(vo3)  # Different values, different hash

    def test_complex_nested_hash(self) -> None:
        """Test hash generation with complex nested data structures."""
        vo = ComplexValueObject(
            name="test",
            metadata={"key": "value", "nested": {"a": 1}},
            tags=["tag1", "tag2"],
            settings={"enabled": True, "debug": False},
        )

        # Should be able to hash complex structures
        hash_value = hash(vo)
        assert isinstance(hash_value, int)

        # Test with unhashable types conversion
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

        hash_value = hash(complex_data)
        assert isinstance(hash_value, int)

    def test_equality_with_different_types(self) -> None:
        """Test equality comparison with different types."""
        vo = SimpleValueObject(value="test")

        # Compare with non-value object types
        assert vo != "string"
        assert vo != 123
        assert vo is not None
        assert vo != []
        assert vo != {}

    def test_subclass_initialization_logging(self) -> None:
        """Test that subclass creation triggers logging."""
        with patch(
            "flext_core.loggings.FlextLoggerFactory.get_logger"
        ) as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            # Create a new subclass
            class TestSubclass(FlextValueObject):
                test_field: str

                def validate_business_rules(self) -> FlextResult[None]:
                    return FlextResult.ok(None)

            # Check that logging was called
            mock_get_logger.assert_called()
            mock_logger.debug.assert_called()


# =============================================================================
# VALIDATION TESTS
# =============================================================================


class TestValueObjectValidation:
    """Test value object validation functionality."""

    def test_validate_flext_success(self) -> None:
        """Test successful FlextValidation."""
        vo = SimpleValueObject(value="test")
        result = vo.validate_flext()
        assert result.success is True
        assert result.data is vo

    def test_validate_flext_failure(self) -> None:
        """Test FlextValidation failure."""
        vo = SimpleValueObject(value="")  # Empty value should fail
        result = vo.validate_flext()
        assert result.success is False
        assert "cannot be empty" in result.error

    def test_validate_field_registry_integration(self) -> None:
        """Test field validation using the registry system."""
        vo = SimpleValueObject(value="test")

        # Test with non-existent field (should pass)
        result = vo.validate_field("unknown_field", "value")
        assert result.success is True

        # Test with mock field that fails validation
        with patch("flext_core.fields.FlextFields.get_field_by_name") as mock_get:
            mock_field = Mock()
            mock_field.validate_value.return_value = FlextResult.fail("Field error")
            mock_get.return_value = FlextResult.ok(mock_field)

            result = vo.validate_field("test_field", "value")
            assert result.success is False
            assert "Field error" in result.error

    def test_validate_all_fields(self) -> None:
        """Test validation of all fields in the value object."""
        vo = SimpleValueObject(value="test")

        # Test successful validation
        result = vo.validate_all_fields()
        assert result.success is True

        # Test with multiple field errors by patching class methods
        def mock_validate_field(self, field_name, field_value):
            if field_name == "field1":
                return FlextResult.fail("Field1 error")
            if field_name == "field2":
                return FlextResult.fail("Field2 error")
            return FlextResult.ok(None)

        with (
            patch.object(SimpleValueObject, "validate_field", mock_validate_field),
            patch.object(
                SimpleValueObject,
                "model_dump",
                return_value={"field1": "val1", "field2": "val2"},
            ),
        ):
            result = vo.validate_all_fields()
            assert result.success is False
            assert "Field validation errors" in result.error
            assert "Field1 error" in result.error
            assert "Field2 error" in result.error

    def test_validate_all_fields_skips_internal(self) -> None:
        """Test that field validation skips internal fields."""

        class TestVO(SimpleValueObject):
            _internal: str = "private"

        vo = TestVO(value="test")
        result = vo.validate_all_fields()
        assert result.success is True  # Should skip _internal field


# =============================================================================
# STRING REPRESENTATION AND FORMATTING TESTS
# =============================================================================


class TestValueObjectFormatting:
    """Test value object string formatting."""

    def test_format_dict_simple(self) -> None:
        """Test basic dictionary formatting."""
        vo = SimpleValueObject(value="test")
        data = {"name": "test", "count": 42, "active": True}
        formatted = vo.format_dict(data)

        assert "name='test'" in formatted
        assert "count=42" in formatted
        assert "active=True" in formatted

    def test_format_dict_complex_types(self) -> None:
        """Test formatting dictionary with various types."""
        vo = SimpleValueObject(value="test")
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
        assert "none_val=None" in formatted
        assert "list=[1, 2, 3]" in formatted

    def test_str_representation(self) -> None:
        """Test string representation."""
        vo = SimpleValueObject(value="test")
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
# SERIALIZATION HELPER TESTS
# =============================================================================


class TestValueObjectSerialization:
    """Test value object serialization helpers."""

    def test_extract_serializable_attributes_pydantic(self) -> None:
        """Test serializable attribute extraction via Pydantic."""
        vo = SimpleValueObject(value="test")
        result = vo._extract_serializable_attributes()

        assert isinstance(result, dict)
        assert "value" in result
        assert result["value"] == "test"

    def test_extract_serializable_no_pydantic(self) -> None:
        """Test fallback to manual extraction."""
        vo = SimpleValueObject(value="test")

        # Mock hasattr to simulate no Pydantic model_dump
        with patch(
            "builtins.hasattr", side_effect=lambda obj, attr: attr != "model_dump"
        ):
            result = vo._extract_serializable_attributes()

            assert isinstance(result, dict)
            # Should fall back to manual extraction

    def test_try_pydantic_serialization(self) -> None:
        """Test Pydantic serialization method."""
        vo = SimpleValueObject(value="test")
        result = vo._try_pydantic_serialization()

        assert isinstance(result, dict)
        assert "value" in result

    def test_try_pydantic_serialization_no_model_dump(self) -> None:
        """Test Pydantic serialization without model_dump."""
        vo = SimpleValueObject(value="test")

        # Mock hasattr to return False for model_dump
        with patch("builtins.hasattr", return_value=False):
            result = vo._try_pydantic_serialization()
            assert result is None

    def test_try_pydantic_serialization_exception(self) -> None:
        """Test Pydantic serialization with exception."""
        vo = SimpleValueObject(value="test")

        # Patch the class method instead of instance method
        with patch.object(
            SimpleValueObject,
            "model_dump",
            side_effect=Exception("Serialization failed"),
        ):
            result = vo._try_pydantic_serialization()
            assert result is None

    def test_try_manual_extraction(self) -> None:
        """Test manual attribute extraction."""
        vo = SimpleValueObject(value="test")
        result = vo._try_manual_extraction()

        assert isinstance(result, dict)
        assert "value" in result

    def test_process_serializable_values(self) -> None:
        """Test processing of serializable values."""
        vo = SimpleValueObject(value="test")
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
        vo = SimpleValueObject(value="test")

        assert vo._should_include_attribute("value") is True
        assert vo._should_include_attribute("_private") is False
        assert vo._should_include_attribute("validate_business_rules") is False

    def test_safely_get_attribute(self) -> None:
        """Test safe attribute retrieval."""
        vo = SimpleValueObject(value="test")

        # Test successful retrieval
        result = vo._safely_get_attribute("value")
        assert result == "test"

        # Test with non-existent attribute
        result = vo._safely_get_attribute("nonexistent")
        assert result is None

    def test_safely_get_attribute_with_str_conversion(self) -> None:
        """Test safe attribute retrieval with __str__ conversion."""

        class ObjectWithStr:
            def __str__(self):
                return "string_repr"

        vo = SimpleValueObject(value="test")
        # Use object.__setattr__ to bypass frozen restriction
        object.__setattr__(vo, "_test_attr", ObjectWithStr())

        result = vo._safely_get_attribute("_test_attr")
        assert result == "string_repr"

    def test_safely_get_attribute_exception_handling(self) -> None:
        """Test safe attribute retrieval with exception."""
        vo = SimpleValueObject(value="test")

        # Mock getattr to raise exception
        with patch("builtins.getattr", side_effect=Exception("Error")):
            result = vo._safely_get_attribute("test_attr")
            assert result is None

    def test_get_fallback_info(self) -> None:
        """Test fallback information generation."""
        vo = SimpleValueObject(value="test")
        result = vo._get_fallback_info()

        assert "class_name" in result
        assert "module" in result
        assert result["class_name"] == "SimpleValueObject"


# =============================================================================
# FACTORY TESTS
# =============================================================================


class TestFlextValueObjectFactory:
    """Test FlextValueObjectFactory functionality."""

    def test_create_factory_basic(self) -> None:
        """Test basic factory creation."""
        factory = FlextValueObjectFactory.create_value_object_factory(SimpleValueObject)

        assert callable(factory)

        # Test factory usage
        result = factory(value="test")
        assert result.success is True
        assert isinstance(result.data, SimpleValueObject)
        assert result.data.value == "test"

    def test_create_factory_with_defaults(self) -> None:
        """Test factory creation with defaults."""
        defaults = {"currency": "EUR"}
        factory = FlextValueObjectFactory.create_value_object_factory(
            MoneyAmount, defaults=defaults
        )

        # Test with defaults
        result = factory(amount=Decimal("10.00"))
        assert result.success is True
        assert result.data.currency == "EUR"  # Default applied

        # Test overriding defaults
        result = factory(amount=Decimal("20.00"), currency="USD")
        assert result.success is True
        assert result.data.currency == "USD"  # Override applied

    def test_factory_validation_failure(self) -> None:
        """Test factory with validation failure."""
        factory = FlextValueObjectFactory.create_value_object_factory(SimpleValueObject)

        result = factory(value="")  # Should fail validation
        assert result.success is False
        assert "cannot be empty" in result.error

    def test_factory_creation_failure(self) -> None:
        """Test factory with creation failure."""
        factory = FlextValueObjectFactory.create_value_object_factory(SimpleValueObject)

        # Pass invalid parameter type
        result = factory(value=123)  # Should be string
        assert result.success is False
        assert "Failed to create" in result.error

    def test_factory_without_defaults(self) -> None:
        """Test factory creation without defaults."""
        factory = FlextValueObjectFactory.create_value_object_factory(
            SimpleValueObject, defaults=None
        )

        result = factory(value="test")
        assert result.success is True
        assert result.data.value == "test"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestValueObjectIntegration:
    """Test integration scenarios with value objects."""

    def test_complex_validation_scenario(self) -> None:
        """Test complex validation scenario."""
        email = EmailAddress(address="user@example.com")

        # Test validation
        validation_result = email.validate_flext()
        assert validation_result.success is True

        # Test string representation
        str_repr = str(email)
        assert "EmailAddress" in str_repr
        assert "user@example.com" in str_repr

    def test_equality_and_hashing_integration(self) -> None:
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

    def test_nested_structures_integration(self) -> None:
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
        email_factory = FlextValueObjectFactory.create_value_object_factory(
            EmailAddress, defaults={}
        )

        # Create multiple emails
        emails = []
        test_addresses = ["user1@example.com", "user2@test.org", "REDACTED_LDAP_BIND_PASSWORD@company.net"]

        for address in test_addresses:
            result = email_factory(address=address)
            assert result.success is True
            emails.append(result.data)

        # Test all emails are valid and unique
        assert len(emails) == 3
        assert len(set(emails)) == 3  # All unique due to different addresses

        # Test string representation for all
        for email in emails:
            str_repr = str(email)
            assert "EmailAddress" in str_repr
            assert email.address in str_repr
