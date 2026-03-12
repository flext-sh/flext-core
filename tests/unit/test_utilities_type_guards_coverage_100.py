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
from typing import ClassVar, override

import pytest
from pydantic import BaseModel, ConfigDict, Field

from flext_core import m, t, u
from flext_core._utilities.guards import FlextUtilitiesGuards


class TypeGuardScenario(BaseModel):
    """Type guard test scenario."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    name: str
    value: t.Scalar = Field(default="")
    expected_result: bool = True


class NormalizeScenario(BaseModel):
    """Normalize test scenario.

    expected_type: the expected type of the normalization result.
    expected_value: optional exact value match.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    name: str
    value: t.Scalar = Field(default="")
    expected_type: type = str
    expected_value: t.Scalar | None = None


class TypeGuardsScenarios:
    """Centralized scenarios for type guard parametrization.

    Uses Pydantic models for strict typing and ClassVar for constants.
    """

    IS_STRING_NON_EMPTY: ClassVar[list[TypeGuardScenario]] = [
        TypeGuardScenario(name="non_empty_string", value="test", expected_result=True),
        TypeGuardScenario(name="empty_string", value="", expected_result=False),
        TypeGuardScenario(name="whitespace_string", value="   ", expected_result=False),
        TypeGuardScenario(name="numeric_string", value="123", expected_result=True),
        TypeGuardScenario(
            name="special_chars", value="!@#$%^&*()", expected_result=True
        ),
        TypeGuardScenario(name="unicode_string", value="日本語", expected_result=True),
        TypeGuardScenario(name="newline_string", value="\n", expected_result=False),
        TypeGuardScenario(name="tab_string", value="\t", expected_result=False),
    ]

    IS_DICT_NON_EMPTY: ClassVar[list[TypeGuardScenario]] = [
        TypeGuardScenario(
            name="non_empty_dict", value="has_items", expected_result=True
        ),
        TypeGuardScenario(name="empty_dict", value="empty", expected_result=False),
    ]

    IS_LIST_NON_EMPTY: ClassVar[list[TypeGuardScenario]] = [
        TypeGuardScenario(
            name="non_empty_list", value="has_items", expected_result=True
        ),
        TypeGuardScenario(name="empty_list", value="empty", expected_result=False),
        TypeGuardScenario(
            name="list_with_empty_string", value="has_empty", expected_result=True
        ),
        TypeGuardScenario(
            name="list_with_none", value="has_none", expected_result=True
        ),
        TypeGuardScenario(name="string_value", value="string", expected_result=False),
        TypeGuardScenario(name="int_value", value=123, expected_result=False),
        TypeGuardScenario(name="none_value", value="", expected_result=False),
    ]

    # Scenarios for normalize_to_metadata_value (deprecated → normalize_to_container):
    # - Scalars: preserved as their original type
    # - None → "" (empty string)
    # - dict → m.Dict (BaseModel RootModel)
    # - list → m.ObjectList (BaseModel RootModel)
    # - unknown → str(val)
    NORMALIZE_TO_METADATA_VALUE: ClassVar[list[NormalizeScenario]] = [
        NormalizeScenario(name="string_value", value="test", expected_type=str),
        NormalizeScenario(
            name="int_value",
            value=42,
            expected_type=int,
            expected_value=42,
        ),
        NormalizeScenario(
            name="float_value",
            value=math.pi,
            expected_type=float,
        ),
        NormalizeScenario(
            name="bool_true",
            value=True,
            expected_type=bool,
            expected_value=True,
        ),
        NormalizeScenario(
            name="bool_false",
            value=False,
            expected_type=bool,
            expected_value=False,
        ),
        # None → "" (empty string)
        NormalizeScenario(
            name="none_value",
            value="",
            expected_type=str,
            expected_value="",
        ),
        # string representations — normalize tests with str inputs
        NormalizeScenario(
            name="complex_object",
            value="complex_object",
            expected_type=str,
        ),
        NormalizeScenario(
            name="set_value",
            value="{1, 2, 3}",
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
        """Test is_string_non_empty with various string scenarios."""
        result = FlextUtilitiesGuards.is_string_non_empty(scenario.value)
        assert result == scenario.expected_result


class TestuTypeGuardsIsDictNonEmpty:
    """Test FlextUtilitiesGuards.is_dict_non_empty."""

    @pytest.mark.parametrize(
        "scenario",
        TypeGuardsScenarios.IS_DICT_NON_EMPTY,
        ids=lambda s: s.name,
    )
    def test_is_dict_non_empty(self, scenario: TypeGuardScenario) -> None:
        """Test is_dict_non_empty with dict-like inputs."""
        if scenario.value == "has_items":
            test_value: dict[str, str] = {"key": "value"}
        else:
            test_value = {}
        result = FlextUtilitiesGuards.is_dict_non_empty(test_value)
        assert result == scenario.expected_result


class TestuTypeGuardsIsListNonEmpty:
    """Test FlextUtilitiesGuards.is_list_non_empty."""

    @pytest.mark.parametrize(
        "scenario",
        TypeGuardsScenarios.IS_LIST_NON_EMPTY,
        ids=lambda s: s.name,
    )
    def test_is_list_non_empty(self, scenario: TypeGuardScenario) -> None:
        """Test is_list_non_empty with various inputs."""
        value: t.Scalar | list[t.Scalar] | None
        if scenario.value == "has_items":
            value = [1, 2, 3]
        elif scenario.value == "empty":
            value = []
        elif scenario.value == "has_empty":
            value = [""]
        elif scenario.value == "has_none":
            value = [None]
        elif scenario.value == "string" or isinstance(scenario.value, int):
            value = scenario.value
        else:
            value = None

        assert FlextUtilitiesGuards.is_general_value_type(value)
        result = u.is_type(value, "list_non_empty")
        assert result == scenario.expected_result


class TestuTypeGuardsNormalizeToMetadataValue:
    """Test normalize_to_metadata_value (deprecated → normalize_to_container).

    normalize_to_container behavior:
    - Scalars: preserved as original type (int→int, float→float, bool→bool)
    - None → "" (empty string)
    - dict → m.Dict (BaseModel RootModel)
    - list → m.ObjectList (BaseModel RootModel)
    - BaseModel → passed through
    - unknown → str(val)
    """

    @pytest.mark.parametrize(
        "scenario",
        TypeGuardsScenarios.NORMALIZE_TO_METADATA_VALUE,
        ids=lambda s: s.name,
    )
    def test_normalize_to_metadata_value(self, scenario: NormalizeScenario) -> None:
        """Test normalize_to_metadata_value with scalar scenarios."""
        value = scenario.value
        assert FlextUtilitiesGuards.is_general_value_type(value)
        result = u.normalize_to_metadata_value(value)
        assert isinstance(result, scenario.expected_type), (
            f"{scenario.name}: expected {scenario.expected_type.__name__}, "
            f"got {type(result).__name__}"
        )
        if scenario.expected_value is not None:
            assert result == scenario.expected_value

    def test_normalize_none_to_empty_string(self) -> None:
        """Test normalize_to_metadata_value: None → empty string."""
        result = u.normalize_to_metadata_value(None)
        assert isinstance(result, str)
        assert result == ""

    def test_normalize_dict_to_pydantic_model(self) -> None:
        """Test normalize_to_metadata_value: dict → m.Dict (BaseModel)."""
        test_dict: m.ConfigMap = m.ConfigMap(root={"key": "value", "num": 42})
        result = u.normalize_to_metadata_value(test_dict)
        assert isinstance(result, BaseModel)

    def test_normalize_list_to_pydantic_model(self) -> None:
        """Test normalize_to_metadata_value: list → m.ObjectList (BaseModel)."""
        test_list = [1, 2, 3]
        result = u.normalize_to_metadata_value(test_list)
        assert isinstance(result, BaseModel)

    def test_normalize_dict_with_primitives(self) -> None:
        """Test normalize_to_metadata_value: dict with primitive values → BaseModel."""
        test_dict: m.ConfigMap = m.ConfigMap(root={"a": 1, "b": "test", "c": True})
        result = u.normalize_to_metadata_value(test_dict)
        assert isinstance(result, BaseModel)

    def test_normalize_dict_with_nested_dict(self) -> None:
        """Test normalize_to_metadata_value: dict with nested dict → BaseModel."""
        inner = m.ConfigMap(root={"nested": "value"})
        outer = m.ConfigMap(root={"key": inner})
        result = u.normalize_to_metadata_value(outer)
        assert isinstance(result, BaseModel)

    def test_normalize_dict_with_list_value(self) -> None:
        """Test normalize_to_metadata_value: dict with list value → BaseModel."""
        test_dict = {"key": [1, 2, 3]}
        result = u.normalize_to_metadata_value(test_dict)
        assert isinstance(result, BaseModel)

    def test_normalize_dict_with_non_string_key(self) -> None:
        """Test normalize_to_metadata_value: dict with non-string key → BaseModel.

        Non-string keys are stringified during normalization.
        """
        test_dict = {123: "value", "key": "test"}
        result = u.normalize_to_metadata_value(test_dict)
        assert isinstance(result, BaseModel)

    def test_normalize_list_with_primitives(self) -> None:
        """Test normalize_to_metadata_value: list with primitives → BaseModel."""
        test_list = ["a", 1, True]
        result = u.normalize_to_metadata_value(test_list)
        assert isinstance(result, BaseModel)

    def test_normalize_list_with_nested_list(self) -> None:
        """Test normalize_to_metadata_value: list with nested list → BaseModel."""
        test_list = [[1, 2], [3, 4]]
        result = u.normalize_to_metadata_value(test_list)
        assert isinstance(result, BaseModel)

    def test_normalize_list_with_dict(self) -> None:
        """Test normalize_to_metadata_value: list with dict → BaseModel."""
        test_list = [{"key": "value"}]
        result = u.normalize_to_metadata_value(test_list)
        assert isinstance(result, BaseModel)

    def test_normalize_list_with_complex_items(self) -> None:
        """Test normalize_to_metadata_value: list with mixed items → BaseModel."""
        test_list = ["string", 42, True, {"dict": "value"}, [1, 2, 3]]
        result = u.normalize_to_metadata_value(test_list)
        assert isinstance(result, BaseModel)

    def test_normalize_tuple_to_pydantic_model(self) -> None:
        """Test normalize_to_metadata_value: tuple → BaseModel (list-like)."""
        test_tuple = (1, 2, 3)
        result = u.normalize_to_metadata_value(test_tuple)
        assert isinstance(result, BaseModel)

    def test_normalize_dict_with_complex_nested_structure(self) -> None:
        """Test normalize_to_metadata_value: complex nested dict → BaseModel.

        Complex nested structures are wrapped in m.Dict Pydantic model.
        Object() values are stringified during normalization.
        """
        test_dict = {
            "str": "value",
            "int": 42,
            "nested_dict": {"inner": "value"},
            "nested_list": [1, 2, {"inner": "dict"}],
            "complex": object(),
        }
        result = u.normalize_to_metadata_value(test_dict)
        assert isinstance(result, BaseModel)

    def test_normalize_custom_object(self) -> None:
        """Test normalize_to_metadata_value: custom object → str(obj)."""

        class CustomObject:
            @override
            def __str__(self) -> str:
                return "custom_object"

        obj = CustomObject()
        result = u.normalize_to_metadata_value(obj)
        assert isinstance(result, str)
        assert result == "custom_object"

    def test_normalize_float_pi(self) -> None:
        """Test normalize_to_metadata_value: float math.pi → float."""
        result = u.normalize_to_metadata_value(math.pi)
        assert isinstance(result, float)
        assert result == math.pi

    def test_normalize_basemodel_passthrough(self) -> None:
        """Test normalize_to_metadata_value: BaseModel → preserved."""

        class SampleModel(BaseModel):
            name: str = "test"

        model = SampleModel()
        result = u.normalize_to_metadata_value(model)
        assert isinstance(result, BaseModel)
        assert isinstance(result, SampleModel)
        assert result.name == "test"


__all__ = [
    "TestuTypeGuardsIsDictNonEmpty",
    "TestuTypeGuardsIsListNonEmpty",
    "TestuTypeGuardsIsStringNonEmpty",
    "TestuTypeGuardsNormalizeToMetadataValue",
    "TypeGuardsScenarios",
]
