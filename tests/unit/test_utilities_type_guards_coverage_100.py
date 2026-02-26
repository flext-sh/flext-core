"""Real tests to achieve 100% type guards utilities coverage - no mocks.

Module: flext_core._utilities.guards
Scope: FlextUtilitiesGuards - all methods and edge cases

This module provides comprehensive real tests (no mocks, patches, or bypasses)
to cover all remaining lines in _utilities/guards.py.

Uses Python 3.13 patterns, FlextTestsUtilities, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, cast

import pytest
from flext_core import u
from flext_core.typings import t


@dataclass(frozen=True, slots=True)
class TypeGuardScenario:
    """Type guard test scenario."""

    name: str
    value: t.GeneralValueType
    expected_result: bool


@dataclass(frozen=True, slots=True)
class NormalizeScenario:
    """Normalize to metadata value test scenario."""

    name: str
    value: t.GeneralValueType
    expected_type: type
    expected_value: object | None = None


class TypeGuardsScenarios:
    """Centralized type guards test scenarios."""

    IS_STRING_NON_EMPTY: ClassVar[list[TypeGuardScenario]] = [
        TypeGuardScenario(name="non_empty_string", value="hello", expected_result=True),
        TypeGuardScenario(
            name="non_empty_with_spaces",
            value="  hello  ",
            expected_result=True,
        ),
        TypeGuardScenario(name="empty_string", value="", expected_result=False),
        TypeGuardScenario(name="whitespace_only", value="   ", expected_result=False),
        TypeGuardScenario(name="newline_only", value="\n", expected_result=False),
        TypeGuardScenario(name="tab_only", value="\t", expected_result=False),
        TypeGuardScenario(name="int_value", value=123, expected_result=False),
        TypeGuardScenario(name="float_value", value=45.6, expected_result=False),
        TypeGuardScenario(name="bool_true", value=True, expected_result=False),
        TypeGuardScenario(name="bool_false", value=False, expected_result=False),
        TypeGuardScenario(name="none_value", value=None, expected_result=False),
        TypeGuardScenario(name="list_value", value=[1, 2, 3], expected_result=False),
        TypeGuardScenario(
            name="dict_value",
            value={"key": "value"},
            expected_result=False,
        ),
    ]

    IS_DICT_NON_EMPTY: ClassVar[list[TypeGuardScenario]] = [
        TypeGuardScenario(
            name="non_empty_dict",
            value={"key": "value"},
            expected_result=True,
        ),
        TypeGuardScenario(
            name="non_empty_dict_multiple",
            value={"a": 1, "b": 2},
            expected_result=True,
        ),
        TypeGuardScenario(name="empty_dict", value={}, expected_result=False),
        TypeGuardScenario(
            name="string_value",
            value="not_a_dict",
            expected_result=False,
        ),
        TypeGuardScenario(name="list_value", value=[1, 2, 3], expected_result=False),
        TypeGuardScenario(name="int_value", value=123, expected_result=False),
        TypeGuardScenario(name="none_value", value=None, expected_result=False),
    ]

    IS_LIST_NON_EMPTY: ClassVar[list[TypeGuardScenario]] = [
        TypeGuardScenario(name="non_empty_list", value=[1, 2, 3], expected_result=True),
        TypeGuardScenario(
            name="non_empty_list_strings",
            value=["a", "b"],
            expected_result=True,
        ),
        TypeGuardScenario(
            name="non_empty_tuple",
            value=(1, 2),
            expected_result=True,
        ),  # is_list_non_empty uses Sequence check, includes tuples
        TypeGuardScenario(name="empty_list", value=[], expected_result=False),
        TypeGuardScenario(name="empty_tuple", value=(), expected_result=False),
        TypeGuardScenario(
            name="string_value",
            value="not_a_list",
            expected_result=False,
        ),
        TypeGuardScenario(
            name="dict_value",
            value={"key": "value"},
            expected_result=False,
        ),
        TypeGuardScenario(name="int_value", value=123, expected_result=False),
        TypeGuardScenario(name="none_value", value=None, expected_result=False),
    ]

    NORMALIZE_TO_METADATA_VALUE: ClassVar[list[NormalizeScenario]] = [
        # Primitive types
        NormalizeScenario(name="string_value", value="test", expected_type=str),
        NormalizeScenario(name="int_value", value=42, expected_type=str),
        NormalizeScenario(name="float_value", value=math.pi, expected_type=str),
        NormalizeScenario(name="bool_true", value=True, expected_type=str),
        NormalizeScenario(name="bool_false", value=False, expected_type=str),
        NormalizeScenario(name="none_value", value=None, expected_type=type(None)),
        # Dict types
        NormalizeScenario(
            name="simple_dict",
            value={"key": "value"},
            expected_type=str,
            expected_value='{"key": "value"}',  # Dicts become JSON strings
        ),
        NormalizeScenario(
            name="dict_with_primitives",
            value={"a": 1, "b": "test", "c": True, "d": None},
            expected_type=str,
        ),
        NormalizeScenario(
            name="dict_with_nested_dict",
            value={"key": {"nested": "value"}},
            expected_type=str,
        ),  # Dict becomes JSON string
        NormalizeScenario(
            name="dict_with_list_value",
            value={"key": [1, 2, 3]},
            expected_type=str,
        ),  # Dict becomes JSON string
        NormalizeScenario(
            name="dict_with_non_string_key",
            value=cast("t.GeneralValueType", {123: "value"}),
            expected_type=str,
        ),
        # List types
        NormalizeScenario(
            name="simple_list",
            value=[1, 2, 3],
            expected_type=list,
            expected_value=["1", "2", "3"],  # Non-scalar items become strings
        ),
        NormalizeScenario(
            name="list_with_primitives",
            value=["a", 1, True, None],
            expected_type=list,
        ),
        NormalizeScenario(
            name="list_with_nested_list",
            value=[[1, 2], [3, 4]],
            expected_type=list,
            expected_value=["[1, 2]", "[3, 4]"],  # Nested lists converted to strings
        ),
        NormalizeScenario(
            name="list_with_dict",
            value=[{"key": "value"}],
            expected_type=list,
            expected_value=["{'key': 'value'}"],  # Dict converted to string
        ),
        NormalizeScenario(
            name="list_with_complex_object",
            value=cast("t.GeneralValueType", [object()]),
            expected_type=list,
        ),
        # Other types (converted to string)
        NormalizeScenario(
            name="complex_object",
            value=cast("t.GeneralValueType", object()),
            expected_type=str,
        ),
        NormalizeScenario(
            name="set_value",
            value=cast("t.GeneralValueType", cast("object", {1, 2, 3})),
            expected_type=str,
        ),
        NormalizeScenario(
            name="tuple_value",
            value=cast("t.GeneralValueType", (1, 2, 3)),
            expected_type=str,
        ),
    ]


class TestuTypeGuardsIsStringNonEmpty:
    """Test FlextUtilitiesGuards.is_string_non_empty."""

    @pytest.mark.parametrize(
        "scenario",
        TypeGuardsScenarios.IS_STRING_NON_EMPTY,
        ids=lambda s: s.name,
    )
    def test_is_string_non_empty(self, scenario: TypeGuardScenario) -> None:
        """Test is_string_non_empty with various scenarios."""
        result = u.is_type(scenario.value, "string_non_empty")
        assert result == scenario.expected_result


class TestuTypeGuardsIsDictNonEmpty:
    """Test FlextUtilitiesGuards.is_dict_non_empty."""

    @pytest.mark.parametrize(
        "scenario",
        TypeGuardsScenarios.IS_DICT_NON_EMPTY,
        ids=lambda s: s.name,
    )
    def test_is_dict_non_empty(self, scenario: TypeGuardScenario) -> None:
        """Test is_dict_non_empty with various scenarios."""
        result = u.is_type(scenario.value, "dict_non_empty")
        assert result == scenario.expected_result


class TestuTypeGuardsIsListNonEmpty:
    """Test FlextUtilitiesGuards.is_list_non_empty."""

    @pytest.mark.parametrize(
        "scenario",
        TypeGuardsScenarios.IS_LIST_NON_EMPTY,
        ids=lambda s: s.name,
    )
    def test_is_list_non_empty(self, scenario: TypeGuardScenario) -> None:
        """Test is_list_non_empty with various scenarios."""
        result = u.is_type(scenario.value, "list_non_empty")
        assert result == scenario.expected_result


class TestuTypeGuardsNormalizeToMetadataValue:
    """Test FlextUtilitiesGuards.normalize_to_metadata_value."""

    @pytest.mark.parametrize(
        "scenario",
        TypeGuardsScenarios.NORMALIZE_TO_METADATA_VALUE,
        ids=lambda s: s.name,
    )
    def test_normalize_to_metadata_value(self, scenario: NormalizeScenario) -> None:
        """Test normalize_to_metadata_value with various scenarios."""
        result = u.normalize_to_metadata_value(scenario.value)

        # Verify type
        assert isinstance(result, scenario.expected_type), scenario.name

        # Verify specific value if provided
        if scenario.expected_value is not None:
            assert result == scenario.expected_value

    def test_normalize_dict_with_non_string_key(self) -> None:
        """Test normalize_to_metadata_value with dict having non-string keys."""
        value = cast("t.GeneralValueType", {123: "value", "key": "test"})
        result = u.normalize_to_metadata_value(value)

        # Dicts are now serialized to JSON strings
        assert isinstance(result, str)
        import json

        parsed = json.loads(result)
        assert "key" in parsed
        assert parsed["key"] == "test"

    def test_normalize_dict_with_complex_nested_structure(self) -> None:
        """Test normalize_to_metadata_value with complex nested dict.

        Dicts containing non-JSON-serializable values (like object()) cause
        TypeError during json.dumps serialization.
        """
        value = cast(
            "t.GeneralValueType",
            {
                "str": "value",
                "int": 42,
                "nested_dict": {"inner": "value"},
                "nested_list": [1, 2, {"inner": "dict"}],
                "complex": object(),
            },
        )
        # object() is not JSON-serializable, so json.dumps raises TypeError
        with pytest.raises(TypeError, match="not JSON serializable"):
            u.normalize_to_metadata_value(value)

    def test_normalize_list_with_complex_items(self) -> None:
        """Test normalize_to_metadata_value with list containing complex items."""
        value = cast(
            "t.GeneralValueType",
            [
                "string",
                42,
                True,
                None,
                {"dict": "value"},
                [1, 2, 3],
                object(),
            ],
        )
        result = u.normalize_to_metadata_value(value)

        assert isinstance(result, list)
        # Type narrowing: result is list after assert_result_matches_expected
        result_list = cast("list[t.MetadataAttributeValue]", result)
        assert result_list[0] == "string"
        assert result_list[1] == "42"  # int becomes str via _is_scalar
        assert result_list[2] == "True"  # bool becomes str via _is_scalar
        assert result_list[3] is None  # None IS scalar, stays None
        # Complex items should be converted to strings
        assert isinstance(result_list[4], str)
        assert isinstance(result_list[5], str)
        assert isinstance(result_list[6], str)

    def test_normalize_custom_object(self) -> None:
        """Test normalize_to_metadata_value with custom object."""

        class CustomObject:
            def __str__(self) -> str:
                return "custom_object"

        value = cast("t.GeneralValueType", cast("object", CustomObject()))
        result = u.normalize_to_metadata_value(value)

        assert isinstance(result, str)
        assert result == "custom_object"


__all__ = [
    "TestuTypeGuardsIsDictNonEmpty",
    "TestuTypeGuardsIsListNonEmpty",
    "TestuTypeGuardsIsStringNonEmpty",
    "TestuTypeGuardsNormalizeToMetadataValue",
    "TypeGuardsScenarios",
]
