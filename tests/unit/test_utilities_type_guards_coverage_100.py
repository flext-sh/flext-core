from __future__ import annotations

import math
from typing import Annotated, ClassVar, cast, override

import pytest
from flext_tests import tm
from pydantic import BaseModel, ConfigDict, Field

from flext_core import FlextRuntime
from flext_core._utilities.guards import FlextUtilitiesGuards
from flext_core.typings import t

type ScalarValue = t.Scalar
type NormalizedValue = t.NormalizedValue


class TestUtilitiesTypeGuardsCoverage100:
    class TypeGuardScenario(BaseModel):
        """Scenario for type guard testing."""

        model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
        name: str
        value: Annotated[ScalarValue, Field(default="")]
        expected_result: bool = True

    class NormalizeScenario(BaseModel):
        """Scenario for normalization testing."""

        model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
        name: str
        value: Annotated[ScalarValue, Field(default="")]
        expected_type: type = str
        expected_value: ScalarValue | None = None

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

    NORMALIZE_TO_METADATA: ClassVar[list[NormalizeScenario]] = [
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
        NormalizeScenario(
            name="none_value",
            value="",
            expected_type=str,
            expected_value="",
        ),
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

    @pytest.mark.parametrize("scenario", IS_STRING_NON_EMPTY, ids=lambda s: s.name)
    def test_is_string_non_empty(self, scenario: TypeGuardScenario) -> None:
        result = FlextUtilitiesGuards.is_string_non_empty(scenario.value)
        tm.that(result, eq=scenario.expected_result)

    @pytest.mark.parametrize("scenario", IS_DICT_NON_EMPTY, ids=lambda s: s.name)
    def test_is_dict_non_empty(self, scenario: TypeGuardScenario) -> None:
        if scenario.value == "has_items":
            test_value: dict[str, str] = {"key": "value"}
        else:
            test_value = {}
        result = FlextUtilitiesGuards.is_dict_non_empty(test_value)
        tm.that(result, eq=scenario.expected_result)

    @pytest.mark.parametrize("scenario", IS_LIST_NON_EMPTY, ids=lambda s: s.name)
    def test_is_list_non_empty(self, scenario: TypeGuardScenario) -> None:
        value: NormalizedValue = None
        if scenario.value == "has_items":
            value = [1, 2, 3]
        elif scenario.value == "empty":
            value = [int()]
        elif scenario.value in {"has_empty", "has_none"}:
            value = [""]
        elif scenario.value == "string" or isinstance(scenario.value, int):
            value = scenario.value
        result = FlextUtilitiesGuards.is_type(value, "list_non_empty")
        tm.that(result, eq=scenario.expected_result)

    @pytest.mark.parametrize("scenario", NORMALIZE_TO_METADATA, ids=lambda s: s.name)
    def test_normalize_to_metadata(self, scenario: NormalizeScenario) -> None:
        value = scenario.value
        assert FlextUtilitiesGuards.is_container(value)
        result = FlextRuntime.normalize_to_metadata(value)
        tm.that(result, is_=scenario.expected_type)
        if scenario.expected_value is not None:
            tm.that(result, eq=scenario.expected_value)

    def test_normalize_none_to_empty_string(self) -> None:
        result = FlextRuntime.normalize_to_metadata(None)
        tm.that(result, is_=str)
        tm.that(result, eq="")

    def test_normalize_dict_to_pydantic_model(self) -> None:
        test_dict = {"key": "value", "num": 42}
        result = FlextRuntime.normalize_to_metadata(cast("t.RuntimeData", test_dict))
        tm.that(result, is_=dict)

    def test_normalize_list_to_pydantic_model(self) -> None:
        test_list = [1, 2, 3]
        result = FlextRuntime.normalize_to_metadata(test_list)
        tm.that(result, is_=list)

    def test_normalize_dict_with_primitives(self) -> None:
        test_dict = {"a": 1, "b": "test", "c": True}
        result = FlextRuntime.normalize_to_metadata(cast("t.RuntimeData", test_dict))
        tm.that(result, is_=dict)

    def test_normalize_dict_with_nested_dict(self) -> None:
        outer = {"key": {"nested": "value"}}
        result = FlextRuntime.normalize_to_metadata(outer)
        tm.that(result, is_=dict)

    def test_normalize_dict_with_list_value(self) -> None:
        test_dict = {"key": [1, 2, 3]}
        result = FlextRuntime.normalize_to_metadata(cast("t.RuntimeData", test_dict))
        tm.that(result, is_=dict)
        assert isinstance(result, dict)
        tm.that(result["key"], is_=list)

    def test_normalize_dict_with_non_string_key(self) -> None:
        test_dict = {"123": "value", "key": "test"}
        result = FlextRuntime.normalize_to_metadata(test_dict)
        tm.that(result, is_=dict)
        assert isinstance(result, dict)
        tm.that("123" in result, eq=True)

    def test_normalize_list_with_primitives(self) -> None:
        test_list = ["a", 1, True]
        result = FlextRuntime.normalize_to_metadata(cast("t.RuntimeData", test_list))
        tm.that(result, is_=list)

    def test_normalize_list_with_nested_list(self) -> None:
        test_list: list[NormalizedValue] = [[1, 2], [3, 4]]
        result = FlextRuntime.normalize_to_metadata(cast("t.RuntimeData", test_list))
        tm.that(result, is_=list)

    def test_normalize_list_with_dict(self) -> None:
        test_list = [{"key": "value"}]
        result = FlextRuntime.normalize_to_metadata(cast("t.RuntimeData", test_list))
        tm.that(result, is_=list)

    def test_normalize_list_with_complex_items(self) -> None:
        test_list: list[NormalizedValue] = [
            "string",
            42,
            True,
            {"dict": "value"},
            [1, 2, 3],
        ]
        result = FlextRuntime.normalize_to_metadata(cast("t.RuntimeData", test_list))
        tm.that(result, is_=list)

    def test_normalize_tuple_to_pydantic_model(self) -> None:
        test_tuple = (1, 2, 3)
        result = FlextRuntime.normalize_to_metadata(test_tuple)
        tm.that(result, is_=list)

    def test_normalize_dict_with_complex_nested_structure(self) -> None:
        test_dict: dict[str, NormalizedValue] = {
            "str": "value",
            "int": 42,
            "nested_dict": {"inner": "value"},
            "nested_list": [1, 2, {"inner": "dict"}],
            "complex": {"a": [1, 2]},
        }
        result = FlextRuntime.normalize_to_metadata(cast("t.RuntimeData", test_dict))
        tm.that(result, is_=dict)

    def test_normalize_custom_object(self) -> None:
        class CustomObject:
            @override
            def __str__(self) -> str:
                return "custom_object"

        obj = CustomObject()
        result = FlextRuntime.normalize_to_metadata(cast("t.RuntimeData", obj))
        tm.that(result, is_=str)
        tm.that(result, eq="custom_object")

    def test_normalize_float_pi(self) -> None:
        result = FlextRuntime.normalize_to_metadata(math.pi)
        tm.that(result, is_=float)
        tm.that(result, eq=math.pi)

    def test_normalize_basemodel_passthrough(self) -> None:
        class SampleModel(BaseModel):
            name: str = "test"

        model = SampleModel()
        result = FlextRuntime.normalize_to_metadata(model)
        tm.that(result, is_=str)
        tm.that(result, has="test")


__all__ = ["TestUtilitiesTypeGuardsCoverage100"]
