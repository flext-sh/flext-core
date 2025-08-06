"""Test models.py hash method specifically - target lines 152-167."""

from __future__ import annotations

import pytest
from pydantic import Field

from flext_core.models import FlextValue
from flext_core.result import FlextResult

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestFlextValueHashSpecific:
    """Test specific lines in FlextValue.__hash__ method."""

    def test_hash_make_hashable_dict_conversion(self) -> None:
        """Test line 154-155: dict conversion in make_hashable."""

        class TestValue(FlextValue):
            dict_field: dict[str, object] = Field(default_factory=dict)

            def validate_business_rules(self) -> FlextResult[None]:
                return FlextResult.ok(None)

        # Create model with dict to trigger line 154-155
        model = TestValue(dict_field={"key": "value", "num": 42})

        # This should call __hash__ which calls make_hashable with dict
        hash_value = hash(model)
        assert isinstance(hash_value, int)

    def test_hash_make_hashable_list_conversion(self) -> None:
        """Test line 156-157: list conversion in make_hashable."""

        class TestValue(FlextValue):
            list_field: list[str] = Field(default_factory=list)

            def validate_business_rules(self) -> FlextResult[None]:
                return FlextResult.ok(None)

        # Create model with list to trigger line 156-157
        model = TestValue(list_field=["a", "b", "c"])

        # This should call __hash__ which calls make_hashable with list
        hash_value = hash(model)
        assert isinstance(hash_value, int)

    def test_hash_make_hashable_set_conversion(self) -> None:
        """Test line 158-159: set conversion in make_hashable."""

        class TestValue(FlextValue):
            set_field: set[str] = Field(default_factory=set)

            def validate_business_rules(self) -> FlextResult[None]:
                return FlextResult.ok(None)

        # Create model with set to trigger line 158-159
        model = TestValue(set_field={"x", "y", "z"})

        # This should call __hash__ which calls make_hashable with set
        hash_value = hash(model)
        assert isinstance(hash_value, int)

    def test_hash_make_hashable_other_types(self) -> None:
        """Test line 160: other types in make_hashable."""

        class TestValue(FlextValue):
            str_field: str = "test"
            int_field: int = 42

            def validate_business_rules(self) -> FlextResult[None]:
                return FlextResult.ok(None)

        # Create model with other types to trigger line 160
        model = TestValue(str_field="hello", int_field=123)

        # This should call __hash__ which calls make_hashable with str/int
        hash_value = hash(model)
        assert isinstance(hash_value, int)

    def test_hash_model_dump_call(self) -> None:
        """Test line 162: model_dump() call."""

        class TestValue(FlextValue):
            field1: str = "value1"
            field2: int = 42

            def validate_business_rules(self) -> FlextResult[None]:
                return FlextResult.ok(None)

        model = TestValue()

        # This should call model_dump() on line 162
        hash_value = hash(model)
        assert isinstance(hash_value, int)

    def test_hash_metadata_exclusion(self) -> None:
        """Test line 165: metadata exclusion from hash."""

        class TestValue(FlextValue):
            regular_field: str = "test"

            def validate_business_rules(self) -> FlextResult[None]:
                return FlextResult.ok(None)

        model1 = TestValue(regular_field="same", metadata={"different": "data1"})
        model2 = TestValue(regular_field="same", metadata={"different": "data2"})

        # Both should have same hash since metadata is excluded (line 165)
        hash1 = hash(model1)
        hash2 = hash(model2)

        # If metadata is properly excluded, hashes should be equal
        assert hash1 == hash2

    def test_hash_hashable_items_loop(self) -> None:
        """Test lines 164-166: the main hash loop."""

        class TestValue(FlextValue):
            field_a: str = "a"
            field_z: str = "z"
            field_m: str = "m"

            def validate_business_rules(self) -> FlextResult[None]:
                return FlextResult.ok(None)

        model = TestValue()

        # This exercises the sorted loop on lines 164-166
        hash_value = hash(model)
        assert isinstance(hash_value, int)

    def test_hash_complex_nested_structures(self) -> None:
        """Test hash with complex nested structures hitting multiple lines."""

        class ComplexValue(FlextValue):
            # This will hit dict conversion (154-155)
            dict_data: dict[str, object] = Field(default_factory=dict)
            # This will hit list conversion (156-157)
            list_data: list[object] = Field(default_factory=list)
            # This will hit set conversion (158-159)
            set_data: set[str] = Field(default_factory=set)
            # This will hit other types (160)
            str_data: str = "test"

            def validate_business_rules(self) -> FlextResult[None]:
                return FlextResult.ok(None)

        model = ComplexValue(
            dict_data={"key": "value", "count": 42},
            list_data=[1, "two", "three"],
            set_data={"a", "b", "c"},
            str_data="complex",
            metadata={"should": "be_excluded"},
        )

        # This should exercise all branches of make_hashable
        hash_value = hash(model)
        assert isinstance(hash_value, int)

        # Test that it's repeatable
        hash_value2 = hash(model)
        assert hash_value == hash_value2
