"""Comprehensive tests for FlextValueObject domain value object base class.

This module provides complete test coverage for the FlextValueObject class,
focusing on missing test cases to achieve 80%+ coverage.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from flext_core.domain.value_object import FlextValueObject


class ConcreteValueObject(FlextValueObject):
    """Concrete value object implementation for comprehensive testing."""

    amount: Decimal
    currency: str = "USD"
    description: str = ""

    def validate_domain_rules(self) -> None:
        """Validate test value object domain rules."""
        if self.amount < 0:
            msg = "Amount cannot be negative"
            raise ValueError(msg)
        if len(self.currency) != 3:
            msg = "Currency must be 3 characters"
            raise ValueError(msg)
        if not self.currency.isupper():
            msg = "Currency must be uppercase"
            raise ValueError(msg)


class ComplexValueObject(FlextValueObject):
    """Value object with complex data types for testing."""

    name: str
    tags: list[str]
    metadata: dict[str, Any]

    def validate_domain_rules(self) -> None:
        """Validate complex value object domain rules."""
        if not self.name.strip():
            msg = "Name cannot be empty"
            raise ValueError(msg)


class TestFlextValueObjectEquality:
    """Test FlextValueObject equality behavior."""

    def test_equality_same_values(self) -> None:
        """Test value objects with same values are equal."""
        vo1 = ConcreteValueObject(amount=Decimal("10.50"), currency="USD")
        vo2 = ConcreteValueObject(amount=Decimal("10.50"), currency="USD")

        assert vo1 == vo2
        assert vo2 == vo1

    def test_equality_different_values(self) -> None:
        """Test value objects with different values are not equal."""
        vo1 = ConcreteValueObject(amount=Decimal("10.50"), currency="USD")
        vo2 = ConcreteValueObject(amount=Decimal("20.00"), currency="USD")
        vo3 = ConcreteValueObject(amount=Decimal("10.50"), currency="EUR")

        assert vo1 != vo2
        assert vo1 != vo3
        assert vo2 != vo3

    def test_equality_with_different_types(self) -> None:
        """Test value object equality with different types."""
        vo = ConcreteValueObject(amount=Decimal("10.50"), currency="USD")

        assert vo != "not a value object"
        assert vo != 42
        assert vo != {"amount": Decimal("10.50"), "currency": "USD"}
        assert vo is not None

    def test_equality_with_different_value_object_types(self) -> None:
        """Test equality between different value object types."""

        class AnotherValueObject(FlextValueObject):
            amount: Decimal
            currency: str = "USD"

            def validate_domain_rules(self) -> None:
                pass

        vo1 = ConcreteValueObject(amount=Decimal("10.50"), currency="USD")
        vo2 = AnotherValueObject(amount=Decimal("10.50"), currency="USD")

        # Different classes should not be equal
        assert vo1 != vo2

    def test_equality_with_optional_fields(self) -> None:
        """Test equality with optional fields."""
        vo1 = ConcreteValueObject(amount=Decimal("10.50"), currency="USD")
        vo2 = ConcreteValueObject(
            amount=Decimal("10.50"),
            currency="USD",
            description="",
        )
        vo3 = ConcreteValueObject(
            amount=Decimal("10.50"),
            currency="USD",
            description="Test",
        )

        assert vo1 == vo2  # Empty description = default empty description
        assert vo1 != vo3  # Different description values
        assert vo2 != vo3


class TestFlextValueObjectHashing:
    """Test FlextValueObject hashing behavior."""

    def test_hash_consistency_same_values(self) -> None:
        """Test hash consistency for value objects with same values."""
        vo1 = ConcreteValueObject(amount=Decimal("10.50"), currency="USD")
        vo2 = ConcreteValueObject(amount=Decimal("10.50"), currency="USD")

        assert hash(vo1) == hash(vo2)

    def test_hash_different_values(self) -> None:
        """Test hash difference for value objects with different values."""
        vo1 = ConcreteValueObject(amount=Decimal("10.50"), currency="USD")
        vo2 = ConcreteValueObject(amount=Decimal("20.00"), currency="USD")

        # Hashes should likely be different (not guaranteed but very probable)
        assert hash(vo1) != hash(vo2)

    def test_hash_with_complex_types(self) -> None:
        """Test hashing with complex data types."""
        vo1 = ComplexValueObject(
            name="Test",
            tags=["tag1", "tag2"],
            metadata={"key": "value"},
        )
        vo2 = ComplexValueObject(
            name="Test",
            tags=["tag1", "tag2"],
            metadata={"key": "value"},
        )
        vo3 = ComplexValueObject(
            name="Test",
            tags=["tag1", "tag3"],  # Different tag
            metadata={"key": "value"},
        )

        assert hash(vo1) == hash(vo2)
        assert hash(vo1) != hash(vo3)

    def test_hash_stability(self) -> None:
        """Test hash stability across multiple calls."""
        vo = ConcreteValueObject(amount=Decimal("10.50"), currency="USD")

        hash1 = hash(vo)
        hash2 = hash(vo)
        hash3 = hash(vo)

        assert hash1 == hash2 == hash3

    def test_hash_in_collections(self) -> None:
        """Test value objects work correctly in hash-based collections."""
        vo1 = ConcreteValueObject(amount=Decimal("10.50"), currency="USD")
        vo2 = ConcreteValueObject(amount=Decimal("10.50"), currency="USD")
        vo3 = ConcreteValueObject(amount=Decimal("20.00"), currency="EUR")

        # Test in set
        vo_set = {vo1, vo2, vo3}
        # vo1 and vo2 are equal, so only 2 unique items
        assert len(vo_set) == 2

        # Test in dict as keys
        vo_dict = {vo1: "first", vo2: "second", vo3: "third"}
        assert len(vo_dict) == 2
        # vo2 overwrote vo1's value
        assert vo_dict[vo1] == "second"


class TestFlextValueObjectStringRepresentation:
    """Test FlextValueObject string representation."""

    def test_str_representation_simple(self) -> None:
        """Test string representation for simple value object."""
        vo = ConcreteValueObject(amount=Decimal("10.50"), currency="USD")

        str_repr = str(vo)

        assert "ConcreteValueObject" in str_repr
        assert "10.50" in str_repr
        assert "USD" in str_repr

    def test_str_representation_with_many_fields(self) -> None:
        """Test string representation with many fields shows ellipsis."""
        vo = ConcreteValueObject(
            amount=Decimal("10.50"),
            currency="USD",
            description=(
                "A long description that should be shown in the string representation"
            ),
        )

        str_repr = str(vo)

        assert "ConcreteValueObject" in str_repr
        # Should show first 3 fields with ellipsis if there are more
        field_count = str_repr.count("=")
        assert field_count <= 3

    def test_str_representation_complex_types(self) -> None:
        """Test string representation with complex data types."""
        vo = ComplexValueObject(
            name="Test Object",
            tags=["tag1", "tag2", "tag3"],
            metadata={"key1": "value1", "key2": "value2"},
        )

        str_repr = str(vo)

        assert "ComplexValueObject" in str_repr
        assert "Test Object" in str_repr


class TestFlextValueObjectDomainValidation:
    """Test FlextValueObject domain validation integration."""

    def test_domain_rules_validation_called(self) -> None:
        """Test that domain rules validation exists and can be called."""
        vo = ConcreteValueObject(amount=Decimal("10.50"), currency="USD")

        # Should not raise any exception for valid value object
        vo.validate_domain_rules()

    def test_domain_rules_validation_negative_amount(self) -> None:
        """Test domain rules validation with invalid amount."""
        vo = ConcreteValueObject(amount=Decimal("-5.00"), currency="USD")

        with pytest.raises(ValueError, match="Amount cannot be negative"):
            vo.validate_domain_rules()

    def test_domain_rules_validation_invalid_currency_length(self) -> None:
        """Test domain rules validation with invalid currency length."""
        vo = ConcreteValueObject(amount=Decimal("10.50"), currency="INVALID")

        with pytest.raises(ValueError, match="Currency must be 3 characters"):
            vo.validate_domain_rules()

    def test_domain_rules_validation_lowercase_currency(self) -> None:
        """Test domain rules validation with lowercase currency."""
        vo = ConcreteValueObject(amount=Decimal("10.50"), currency="usd")

        with pytest.raises(ValueError, match="Currency must be uppercase"):
            vo.validate_domain_rules()

    def test_domain_rules_validation_empty_name(self) -> None:
        """Test domain rules validation with empty name."""
        vo = ComplexValueObject(name="", tags=[], metadata={})

        with pytest.raises(ValueError, match="Name cannot be empty"):
            vo.validate_domain_rules()

    def test_domain_rules_validation_whitespace_name(self) -> None:
        """Test domain rules validation with whitespace-only name."""
        vo = ComplexValueObject(name="   ", tags=[], metadata={})

        with pytest.raises(ValueError, match="Name cannot be empty"):
            vo.validate_domain_rules()


class TestFlextValueObjectPydanticIntegration:
    """Test FlextValueObject integration with Pydantic features."""

    def test_pydantic_validation_required_fields(self) -> None:
        """Test Pydantic validation for required fields."""
        with pytest.raises(ValidationError):
            ConcreteValueObject()  # Missing required 'amount' field

    def test_pydantic_validation_field_types(self) -> None:
        """Test Pydantic validation for field types."""
        with pytest.raises(ValidationError):
            ConcreteValueObject(amount="not_a_decimal", currency="USD")

    def test_pydantic_model_dump(self) -> None:
        """Test Pydantic model_dump functionality."""
        vo = ConcreteValueObject(
            amount=Decimal("10.50"),
            currency="USD",
            description="Test description",
        )

        data = vo.model_dump()

        assert isinstance(data, dict)
        assert data["amount"] == Decimal("10.50")
        assert data["currency"] == "USD"
        assert data["description"] == "Test description"

    def test_pydantic_immutability(self) -> None:
        """Test that value objects are immutable."""
        vo = ConcreteValueObject(amount=Decimal("10.50"), currency="USD")

        with pytest.raises((ValidationError, AttributeError, TypeError)):
            vo.amount = Decimal("20.00")

        with pytest.raises((ValidationError, AttributeError, TypeError)):
            vo.currency = "EUR"

    def test_pydantic_default_values(self) -> None:
        """Test Pydantic default values work correctly."""
        vo = ConcreteValueObject(amount=Decimal("10.50"))

        assert vo.currency == "USD"  # Default value
        assert vo.description == ""  # Default value

    def test_pydantic_string_stripping(self) -> None:
        """Test Pydantic string stripping configuration."""
        vo = ConcreteValueObject(
            amount=Decimal("10.50"),
            currency="USD",
            description="  Test description  ",
        )

        assert vo.description == "Test description"  # Whitespace stripped

    def test_pydantic_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            ConcreteValueObject(
                amount=Decimal("10.50"),
                currency="USD",
                extra_field="not allowed",
            )


class TestFlextValueObjectEdgeCases:
    """Test FlextValueObject edge cases and boundary conditions."""

    def test_empty_complex_collections(self) -> None:
        """Test value objects with empty complex collections."""
        vo = ComplexValueObject(name="Test", tags=[], metadata={})

        assert vo.name == "Test"
        assert vo.tags == []
        assert vo.metadata == {}

        # Should work in hash-based collections
        vo_set = {vo}
        assert len(vo_set) == 1

    def test_nested_complex_data_equality(self) -> None:
        """Test equality with nested complex data structures."""
        vo1 = ComplexValueObject(
            name="Test",
            tags=["a", "b", "c"],
            metadata={"nested": {"key": "value"}, "list": [1, 2, 3]},
        )
        vo2 = ComplexValueObject(
            name="Test",
            tags=["a", "b", "c"],
            metadata={"nested": {"key": "value"}, "list": [1, 2, 3]},
        )

        assert vo1 == vo2
        assert hash(vo1) == hash(vo2)

    def test_large_data_structures(self) -> None:
        """Test value objects with large data structures."""
        large_tags = [f"tag_{i}" for i in range(100)]
        large_metadata = {f"key_{i}": f"value_{i}" for i in range(50)}

        vo = ComplexValueObject(
            name="Large Object",
            tags=large_tags,
            metadata=large_metadata,
        )

        assert len(vo.tags) == 100
        assert len(vo.metadata) == 50

        # Should still work correctly
        hash_value = hash(vo)
        assert isinstance(hash_value, int)

    def test_special_characters_in_strings(self) -> None:
        """Test value objects with special characters."""
        vo = ComplexValueObject(
            name="Test with üñïçödé chars and symbols: !@#$%^&*()",
            tags=["tag-with-dashes", "tag_with_underscores", "tag.with.dots"],
            metadata={"key with spaces": "value with spaces", "émoji": "🎉"},
        )

        assert "üñïçödé" in vo.name
        assert "tag-with-dashes" in vo.tags
        assert vo.metadata["key with spaces"] == "value with spaces"

    def test_decimal_precision_equality(self) -> None:
        """Test decimal precision in equality comparisons."""
        vo1 = ConcreteValueObject(amount=Decimal("10.50"), currency="USD")
        vo2 = ConcreteValueObject(amount=Decimal("10.500"), currency="USD")

        # Decimal comparison should handle precision correctly
        assert vo1 == vo2

    def test_model_dump_with_complex_types(self) -> None:
        """Test model_dump with complex data types."""
        vo = ComplexValueObject(
            name="Test",
            tags=["tag1", "tag2"],
            metadata={"nested": {"key": "value"}},
        )

        data = vo.model_dump()

        assert isinstance(data["tags"], list)
        assert isinstance(data["metadata"], dict)
        assert data["metadata"]["nested"]["key"] == "value"


class TestFlextValueObjectInheritance:
    """Test FlextValueObject inheritance patterns."""

    def test_single_inheritance(self) -> None:
        """Test single inheritance from FlextValueObject."""

        class SpecialValue(ConcreteValueObject):
            special_field: str = "special"

        vo = SpecialValue(amount=Decimal("10.50"), currency="USD")

        assert isinstance(vo, SpecialValue)
        assert isinstance(vo, ConcreteValueObject)
        assert isinstance(vo, FlextValueObject)
        assert vo.special_field == "special"

    def test_multiple_inheritance_behavior(self) -> None:
        """Test value object behavior with multiple inheritance patterns."""

        class Mixin:
            def get_info(self) -> str:
                return "mixin method"

        class MixedValue(ConcreteValueObject, Mixin):
            pass

        vo = MixedValue(amount=Decimal("10.50"), currency="USD")

        assert vo.get_info() == "mixin method"
        assert isinstance(vo, FlextValueObject)

    def test_polymorphic_equality(self) -> None:
        """Test polymorphic equality behavior."""

        class SpecialValue(ConcreteValueObject):
            special_field: str = "special"

        base_vo = ConcreteValueObject(amount=Decimal("10.50"), currency="USD")
        special_vo = SpecialValue(amount=Decimal("10.50"), currency="USD")

        # Different classes should not be equal, even with same base fields
        assert base_vo != special_vo
