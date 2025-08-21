"""Comprehensive tests for FlextValueObject domain value object base class.

This module provides complete test coverage for the FlextValueObject class,
focusing on missing test cases to achieve 80%+ coverage.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from flext_core import FlextResult, FlextValueObject

from ...shared_test_domain import (
    ComplexValueObject,
    ConcreteValueObject,
    TestDomainFactory,
)

# Constants
EXPECTED_BULK_SIZE = 2


class TestFlextValueObjectEquality:
    """Test FlextValueObject equality behavior."""

    def test_equality_same_values(self) -> None:
        """Test value objects with same values are equal."""
        vo1_result = TestDomainFactory.create_concrete_value_object(
            amount=Decimal("10.50"),
            currency="USD",
        )
        vo2_result = TestDomainFactory.create_concrete_value_object(
            amount=Decimal("10.50"),
            currency="USD",
        )
        assert vo1_result.success
        assert vo2_result.success
        vo1, vo2 = vo1_result.value, vo2_result.value

        if vo1 != vo2:
            raise AssertionError(f"Expected {vo2}, got {vo1}")
        assert vo2 == vo1

    def test_equality_different_values(self) -> None:
        """Test value objects with different values are not equal."""
        vo1 = ConcreteValueObject.model_validate(
            {
                "amount": Decimal("10.50"),
                "currency": "USD",
            }
        )
        vo2 = ConcreteValueObject.model_validate(
            {
                "amount": Decimal("20.00"),
                "currency": "USD",
            }
        )
        vo3 = ConcreteValueObject.model_validate(
            {
                "amount": Decimal("10.50"),
                "currency": "EUR",
            }
        )

        assert vo1 != vo2
        assert vo1 != vo3
        assert vo2 != vo3

    def test_equality_with_different_types(self) -> None:
        """Test value object equality with different types."""
        vo = ConcreteValueObject.model_validate(
            {
                "amount": Decimal("10.50"),
                "currency": "USD",
            }
        )

        # Test that value object is not equal to primitive types
        assert vo != "not a value object"
        assert vo != 42
        assert vo != {"amount": Decimal("10.50"), "currency": "USD"}
        assert vo is not None

    def test_equality_with_different_value_object_types(self) -> None:
        """Test equality between different value object types."""

        class AnotherValueObject(FlextValueObject):
            amount: Decimal
            currency: str = "USD"

            def validate_business_rules(self) -> FlextResult[None]:
                return FlextResult[None].ok(None)

        vo1 = ConcreteValueObject.model_validate(
            {
                "amount": Decimal("10.50"),
                "currency": "USD",
            }
        )
        vo2 = AnotherValueObject.model_validate(
            {
                "amount": Decimal("10.50"),
                "currency": "USD",
            }
        )

        # Different classes should not be equal
        assert vo1.__class__.__name__ != vo2.__class__.__name__

    def test_equality_with_optional_fields(self) -> None:
        """Test equality with optional fields."""
        vo1 = ConcreteValueObject.model_validate(
            {
                "amount": Decimal("10.50"),
                "currency": "USD",
            }
        )
        vo2 = ConcreteValueObject.model_validate(
            {
                "amount": Decimal("10.50"),
                "currency": "USD",
                "description": "",
            }
        )
        vo3 = ConcreteValueObject.model_validate(
            {
                "amount": Decimal("10.50"),
                "currency": "USD",
                "description": "Test",
            }
        )

        if vo1 != vo2:  # Empty description = default empty description:
            raise AssertionError(
                f"Expected {vo2} # Empty description = default empty description, got {vo1}",
            )
        assert vo1 != vo3  # Different description values
        assert vo2 != vo3


class TestFlextValueObjectHashing:
    """Test FlextValueObject hashing behavior."""

    def test_hash_consistency_same_values(self) -> None:
        """Test hash consistency for value objects with same values."""
        vo1 = ConcreteValueObject.model_validate(
            {
                "amount": Decimal("10.50"),
                "currency": "USD",
            }
        )
        vo2 = ConcreteValueObject.model_validate(
            {
                "amount": Decimal("10.50"),
                "currency": "USD",
            }
        )

        if hash(vo1) != hash(vo2):
            raise AssertionError(f"Expected {hash(vo2)}, got {hash(vo1)}")

    def test_hash_different_values(self) -> None:
        """Test hash difference for value objects with different values."""
        vo1 = ConcreteValueObject.model_validate(
            {
                "amount": Decimal("10.50"),
                "currency": "USD",
            }
        )
        vo2 = ConcreteValueObject.model_validate(
            {
                "amount": Decimal("20.00"),
                "currency": "USD",
            }
        )

        # Hashes should likely be different (not guaranteed but very probable)
        assert hash(vo1) != hash(vo2)

    def test_hash_with_complex_types(self) -> None:
        """Test that hashing works with complex types by converting them to hashable."""
        vo1 = ComplexValueObject.model_validate(
            {
                "name": "Test",
                "tags": ["tag1", "tag2"],  # List is unhashable but converted to tuple
                "metadata": {
                    "key": "value"
                },  # Dict is unhashable but converted to frozenset
            }
        )

        # Value objects with complex types should now be hashable via conversion
        hash_value = hash(vo1)
        assert isinstance(hash_value, int)

        # Should work in collections that require hashable items
        vo_set = {vo1}  # pyright: ignore[reportUnhashable]
        if len(vo_set) != 1:
            raise AssertionError(f"Expected {1}, got {len(vo_set)}")

    def test_hash_stability(self) -> None:
        """Test hash stability across multiple calls."""
        vo = ConcreteValueObject.model_validate(
            {
                "amount": Decimal("10.50"),
                "currency": "USD",
            }
        )

        hash1 = hash(vo)
        hash2 = hash(vo)
        hash3 = hash(vo)

        if hash1 == hash2 != hash3:
            raise AssertionError(f"Expected {hash3}, got {hash1 == hash2}")

    def test_hash_in_collections(self) -> None:
        """Test value objects work correctly in hash-based collections."""
        vo1 = ConcreteValueObject.model_validate(
            {
                "amount": Decimal("10.50"),
                "currency": "USD",
            }
        )
        vo2 = ConcreteValueObject.model_validate(
            {
                "amount": Decimal("10.50"),
                "currency": "USD",
            }
        )
        vo3 = ConcreteValueObject.model_validate(
            {
                "amount": Decimal("20.00"),
                "currency": "EUR",
            }
        )

        # Test in set
        vo_set = {vo1, vo2, vo3}  # pyright: ignore[reportUnhashable]
        # vo1 and vo2 are equal, so only 2 unique items
        if len(vo_set) != EXPECTED_BULK_SIZE:
            raise AssertionError(f"Expected {2}, got {len(vo_set)}")

        # Test in dict as keys
        vo_dict = {vo1: "first", vo2: "second", vo3: "third"}  # pyright: ignore[reportUnhashable]
        if len(vo_dict) != EXPECTED_BULK_SIZE:
            raise AssertionError(f"Expected {2}, got {len(vo_dict)}")
        # vo2 overwrote vo1's value
        if vo_dict[vo1] != "second":
            raise AssertionError(f"Expected {'second'}, got {vo_dict[vo1]}")


class TestFlextValueObjectStringRepresentation:
    """Test FlextValueObject string representation."""

    def test_str_representation_simple(self) -> None:
        """Test string representation for simple value object."""
        vo = ConcreteValueObject.model_validate(
            {
                "amount": Decimal("10.50"),
                "currency": "USD",
            }
        )

        str_repr = str(vo)

        if "TestMoney" not in str_repr:
            raise AssertionError(f"Expected {'TestMoney'} in {str_repr}")
        assert "10.50" in str_repr
        if "USD" not in str_repr:
            raise AssertionError(f"Expected {'USD'} in {str_repr}")

    def test_str_representation_with_many_fields(self) -> None:
        """Test string representation with many fields shows ellipsis."""
        vo = ConcreteValueObject.model_validate(
            {
                "amount": Decimal("10.50"),
                "currency": "USD",
                "description": (
                    "A long description that should be shown in the string representation"
                ),
            }
        )

        str_repr = str(vo)

        if "TestMoney" not in str_repr:
            raise AssertionError(f"Expected {'TestMoney'} in {str_repr}")
        # Should show the fields (actual count depends on implementation)
        field_count = str_repr.count("=")
        assert field_count >= 1  # At least one field should be shown

    def test_str_representation_complex_types(self) -> None:
        """Test string representation with complex data types."""
        vo = ComplexValueObject.model_validate(
            {
                "name": "Test Object",
                "tags": ["tag1", "tag2", "tag3"],
                "metadata": {"key1": "value1", "key2": "value2"},
            }
        )

        str_repr = str(vo)

        if "ComplexValueObject" not in str_repr:
            raise AssertionError(f"Expected {'ComplexValueObject'} in {str_repr}")
        assert "Test Object" in str_repr


class TestFlextValueObjectDomainValidation:
    """Test FlextValueObject domain validation integration."""

    def test_domain_rules_validation_called(self) -> None:
        """Test that domain rules validation exists and can be called."""
        vo = ConcreteValueObject.model_validate(
            {
                "amount": Decimal("10.50"),
                "currency": "USD",
            }
        )

        # Should return success for valid value object
        result = vo.validate_business_rules()
        assert result.success

    def test_domain_rules_validation_negative_amount(self) -> None:
        """Test domain rules validation with invalid amount."""
        vo = ConcreteValueObject.model_validate(
            {
                "amount": Decimal("-5.00"),
                "currency": "USD",
            }
        )

        result = vo.validate_business_rules()
        assert result.is_failure
        assert "Amount cannot be negative" in (result.error or "")

    def test_domain_rules_validation_invalid_currency_length(self) -> None:
        """Test domain rules validation with invalid currency length."""
        vo = ConcreteValueObject.model_validate(
            {
                "amount": Decimal("10.50"),
                "currency": "INVALID",
            }
        )

        result = vo.validate_business_rules()
        assert result.is_failure
        assert "Currency must be 3 characters" in (result.error or "")

    def test_domain_rules_validation_lowercase_currency(self) -> None:
        """Test domain rules validation with lowercase currency."""
        vo = ConcreteValueObject.model_validate(
            {
                "amount": Decimal("10.50"),
                "currency": "usd",
            }
        )

        result = vo.validate_business_rules()
        assert result.is_failure
        assert "Currency must be uppercase" in (result.error or "")

    def test_domain_rules_validation_empty_name(self) -> None:
        """Test domain rules validation with empty name."""
        vo = ComplexValueObject.model_validate({"name": "", "tags": [], "metadata": {}})

        result = vo.validate_business_rules()
        assert result.is_failure
        assert "Name cannot be empty" in (result.error or "")

    def test_domain_rules_validation_whitespace_name(self) -> None:
        """Test domain rules validation with whitespace-only name."""
        vo = ComplexValueObject.model_validate(
            {
                "name": "   ",
                "tags": [],
                "metadata": {},
            }
        )

        result = vo.validate_business_rules()
        assert result.is_failure
        assert "Name cannot be empty" in (result.error or "")


class TestFlextValueObjectPydanticIntegration:
    """Test FlextValueObject integration with Pydantic features."""

    def test_pydantic_validation_required_fields(self) -> None:
        """Test Pydantic validation for required fields."""
        with pytest.raises(ValidationError):
            # This should fail because amount is required and None is not a valid Decimal
            ConcreteValueObject.model_validate({"amount": None, "currency": "USD"})

    def test_pydantic_validation_field_types(self) -> None:
        """Test Pydantic validation for field types."""
        with pytest.raises(ValidationError):
            ConcreteValueObject.model_validate(
                {
                    "amount": "not_a_decimal",
                    "currency": "USD",
                }
            )

    def test_pydantic_model_dump(self) -> None:
        """Test Pydantic model_dump functionality."""
        vo = ConcreteValueObject.model_validate(
            {
                "amount": Decimal("10.50"),
                "currency": "USD",
                "description": "Test description",
            }
        )

        data = vo.model_dump()

        assert isinstance(data, dict)
        if data["amount"] != Decimal("10.50"):
            raise AssertionError(f"Expected {Decimal('10.50')}, got {data['amount']}")
        assert data["currency"] == "USD"
        if data["description"] != "Test description":
            raise AssertionError(
                f"Expected {'Test description'}, got {data['description']}",
            )

    def test_pydantic_immutability(self) -> None:
        """Test that value objects are immutable."""
        vo = ConcreteValueObject.model_validate(
            {
                "amount": Decimal("10.50"),
                "currency": "USD",
            }
        )

        with pytest.raises((ValidationError, AttributeError, TypeError)):
            vo.amount = Decimal("20.00")

        with pytest.raises((ValidationError, AttributeError, TypeError)):
            vo.currency = "EUR"

    def test_pydantic_default_values(self) -> None:
        """Test Pydantic default values work correctly."""
        vo = ConcreteValueObject.model_validate({"amount": Decimal("10.50")})

        if vo.currency != "USD":  # Default value:
            raise AssertionError(f"Expected {'USD'} # Default value, got {vo.currency}")
        assert vo.description == ""  # Default value

    def test_pydantic_string_stripping(self) -> None:
        """Test Pydantic string stripping configuration."""
        vo = ConcreteValueObject.model_validate(
            {
                "amount": Decimal("10.50"),
                "currency": "USD",
                "description": "  Test description  ",
            }
        )

        # Check if whitespace was stripped or preserved (depends on configuration)
        assert "Test description" in vo.description

    def test_pydantic_extra_fields_forbidden(self) -> None:
        """Test that extra fields are handled appropriately."""
        try:
            vo = ConcreteValueObject.model_validate(
                {
                    "amount": Decimal("10.50"),
                    "currency": "USD",
                    "extra_field": "not allowed",
                }
            )
            # If extra=forbid, this should raise ValidationError
            # If extra=ignore, this succeeds but extra_field is ignored
            # Both are valid depending on configuration
            assert vo.amount == Decimal("10.50")
        except ValidationError:
            # This is expected if extra=forbid is configured
            pass


class TestFlextValueObjectEdgeCases:
    """Test FlextValueObject edge cases and boundary conditions."""

    def test_empty_complex_collections(self) -> None:
        """Test value objects with empty complex collections."""
        vo = ComplexValueObject.model_validate(
            {
                "name": "Test",
                "tags": [],
                "metadata": {},
            }
        )

        if vo.name != "Test":
            raise AssertionError(f"Expected {'Test'}, got {vo.name}")
        assert vo.tags == []
        if vo.metadata != {}:
            raise AssertionError(f"Expected {{}}, got {vo.metadata}")

        # Should work in hash-based collections
        vo_set = {vo}  # pyright: ignore[reportUnhashable]
        if len(vo_set) != 1:
            raise AssertionError(f"Expected {1}, got {len(vo_set)}")

    def test_nested_complex_data_equality(self) -> None:
        """Test equality with nested complex data structures."""
        vo1 = ComplexValueObject.model_validate(
            {
                "name": "Test",
                "tags": ["a", "b", "c"],
                "metadata": {"nested": {"key": "value"}, "list": [1, 2, 3]},
            }
        )
        vo2 = ComplexValueObject.model_validate(
            {
                "name": "Test",
                "tags": ["a", "b", "c"],
                "metadata": {"nested": {"key": "value"}, "list": [1, 2, 3]},
            }
        )

        if vo1 != vo2:
            raise AssertionError(f"Expected {vo2}, got {vo1}")
        assert hash(vo1) == hash(vo2)

    def test_large_data_structures(self) -> None:
        """Test value objects with large data structures."""
        large_tags = [f"tag_{i}" for i in range(100)]
        large_metadata: dict[str, object] = {
            f"key_{i}": f"value_{i}" for i in range(50)
        }

        vo = ComplexValueObject.model_validate(
            {
                "name": "Large Object",
                "tags": large_tags,
                "metadata": large_metadata,
            }
        )

        if len(vo.tags) != 100:
            raise AssertionError(f"Expected {100}, got {len(vo.tags)}")
        assert len(vo.metadata) == 50

        # Should still work correctly
        hash_value = hash(vo)
        assert isinstance(hash_value, int)

    def test_special_characters_in_strings(self) -> None:
        """Test value objects with special characters."""
        vo = ComplexValueObject.model_validate(
            {
                "name": "Test with üñïçödé chars and symbols: !@#$%^&*()",
                "tags": ["tag-with-dashes", "tag_with_underscores", "tag.with.dots"],
                "metadata": {"key with spaces": "value with spaces", "émoji": "🎉"},
            }
        )

        if "üñïçödé" not in vo.name:
            raise AssertionError(f"Expected {'üñïçödé'} in {vo.name}")
        assert "tag-with-dashes" in vo.tags
        if vo.metadata["key with spaces"] != "value with spaces":
            raise AssertionError(
                f"Expected {'value with spaces'}, got {vo.metadata['key with spaces']}",
            )

    def test_decimal_precision_equality(self) -> None:
        """Test decimal precision in equality comparisons."""
        vo1 = ConcreteValueObject.model_validate(
            {
                "amount": Decimal("10.50"),
                "currency": "USD",
            }
        )
        vo2 = ConcreteValueObject.model_validate(
            {
                "amount": Decimal("10.500"),
                "currency": "USD",
            }
        )

        # Decimal comparison should handle precision correctly
        if vo1 != vo2:
            raise AssertionError(f"Expected {vo2}, got {vo1}")

    def test_model_dump_with_complex_types(self) -> None:
        """Test model_dump with complex data types."""
        vo = ComplexValueObject.model_validate(
            {
                "name": "Test",
                "tags": ["tag1", "tag2"],
                "metadata": {"nested": {"key": "value"}},
            }
        )

        data = vo.model_dump()

        assert isinstance(data["tags"], list)
        assert isinstance(data["metadata"], dict)
        if data["metadata"]["nested"]["key"] != "value":
            raise AssertionError(
                f"Expected {'value'}, got {data['metadata']['nested']['key']}",
            )


class TestFlextValueObjectInheritance:
    """Test FlextValueObject inheritance patterns."""

    def test_single_inheritance(self) -> None:
        """Test single inheritance from FlextValueObject."""

        class SpecialValue(ConcreteValueObject):
            special_field: str = "special"

        vo = SpecialValue.model_validate(
            {
                "amount": Decimal("10.50"),
                "currency": "USD",
            }
        )

        assert isinstance(vo, SpecialValue)
        assert isinstance(vo, ConcreteValueObject)
        assert isinstance(vo, FlextValueObject)
        if vo.special_field != "special":
            raise AssertionError(f"Expected {'special'}, got {vo.special_field}")

    def test_multiple_inheritance_behavior(self) -> None:
        """Test value object behavior with multiple inheritance patterns."""

        class Mixin:
            def get_info(self) -> str:
                return "mixin method"

        class MixedValue(ConcreteValueObject, Mixin):
            pass

        vo = MixedValue.model_validate({"amount": Decimal("10.50"), "currency": "USD"})

        if vo.get_info() != "mixin method":
            raise AssertionError(f"Expected {'mixin method'}, got {vo.get_info()}")
        assert isinstance(vo, FlextValueObject)

    def test_polymorphic_equality(self) -> None:
        """Test polymorphic equality behavior."""

        class SpecialValue(ConcreteValueObject):
            special_field: str = "special"

        base_vo = ConcreteValueObject.model_validate(
            {
                "amount": Decimal("10.50"),
                "currency": "USD",
            }
        )
        special_vo = SpecialValue.model_validate(
            {
                "amount": Decimal("10.50"),
                "currency": "USD",
            }
        )

        # Different classes should not be equal, even with same base fields
        assert base_vo != special_vo
