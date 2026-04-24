from __future__ import annotations

import math
import pathlib
from collections.abc import (
    Sequence,
)
from typing import Annotated, ClassVar, override

import pytest
from flext_tests import tm

from tests import m, t, u


class TestUtilitiesTypeGuardsCoverage100:
    class TypeGuardScenario(m.BaseModel):
        """Scenario for type guard testing."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(
            frozen=True,
            arbitrary_types_allowed=True,
        )
        name: str
        value: Annotated[t.Scalar, m.Field(default="")]
        expected_result: bool = True

    class NormalizeScenario(m.BaseModel):
        """Scenario for normalization testing."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(
            frozen=True,
            arbitrary_types_allowed=True,
        )
        name: str
        value: Annotated[t.Scalar, m.Field(default="")]
        expected_type: type = str
        expected_value: t.Scalar | None = None

    IS_STRING_NON_EMPTY: ClassVar[Sequence[TypeGuardScenario]] = [
        TypeGuardScenario(name="non_empty_string", value="test", expected_result=True),
        TypeGuardScenario(name="empty_string", value="", expected_result=False),
        TypeGuardScenario(name="whitespace_string", value="   ", expected_result=False),
        TypeGuardScenario(name="numeric_string", value="123", expected_result=True),
        TypeGuardScenario(
            name="special_chars",
            value="!@#$%^&*()",
            expected_result=True,
        ),
        TypeGuardScenario(name="unicode_string", value="日本語", expected_result=True),
        TypeGuardScenario(name="newline_string", value="\n", expected_result=False),
        TypeGuardScenario(name="tab_string", value="\t", expected_result=False),
    ]

    dict_non_empty: ClassVar[Sequence[TypeGuardScenario]] = [
        TypeGuardScenario(
            name="non_empty_dict",
            value="has_items",
            expected_result=True,
        ),
        TypeGuardScenario(name="empty_dict", value="empty", expected_result=False),
    ]

    IS_LIST_NON_EMPTY: ClassVar[Sequence[TypeGuardScenario]] = [
        TypeGuardScenario(
            name="non_empty_list",
            value="has_items",
            expected_result=True,
        ),
        TypeGuardScenario(name="empty_list", value="empty", expected_result=False),
        TypeGuardScenario(
            name="list_with_empty_string",
            value="has_empty",
            expected_result=True,
        ),
        TypeGuardScenario(
            name="list_with_none",
            value="has_none",
            expected_result=True,
        ),
        TypeGuardScenario(name="string_value", value="string", expected_result=False),
        TypeGuardScenario(name="int_value", value=123, expected_result=False),
        TypeGuardScenario(name="none_value", value="", expected_result=False),
    ]

    NORMALIZE_TO_METADATA: ClassVar[Sequence[NormalizeScenario]] = [
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
        result = u.string_non_empty(scenario.value)
        tm.that(result, eq=scenario.expected_result)

    @pytest.mark.parametrize("scenario", dict_non_empty, ids=lambda s: s.name)
    def test_is_dict_non_empty(self, scenario: TypeGuardScenario) -> None:
        if scenario.value == "has_items":
            test_value: t.StrMapping = {"key": "value"}
        else:
            test_value = dict[str, str]()
        result = u.dict_non_empty(test_value)
        tm.that(result, eq=scenario.expected_result)

    @pytest.mark.parametrize("scenario", IS_LIST_NON_EMPTY, ids=lambda s: s.name)
    def test_is_list_non_empty(self, scenario: TypeGuardScenario) -> None:
        if scenario.value == "has_items":
            value = [1, 2, 3]
        elif scenario.value == "empty":
            value = list[t.JsonValue]()
        elif scenario.value in {"has_empty", "has_none"}:
            value = [0]
        else:
            value = scenario.value
        result = u.matches_type(value, "list_non_empty")
        tm.that(result, eq=scenario.expected_result)

    @pytest.mark.parametrize("scenario", NORMALIZE_TO_METADATA, ids=lambda s: s.name)
    def test_normalize_to_metadata(self, scenario: NormalizeScenario) -> None:
        value = scenario.value
        assert u.container(value)
        result = u.normalize_to_metadata(value)
        tm.that(result, is_=scenario.expected_type)
        if scenario.expected_value is not None:
            tm.that(result, eq=scenario.expected_value)

    def test_normalize_none_to_empty_string(self) -> None:
        result = u.normalize_to_metadata("")
        tm.that(result, is_=str)
        tm.that(result, eq="")

    def test_normalize_dict_to_pydantic_model(self) -> None:
        test_dict: t.JsonValue = {"key": "value", "num": 42}
        result = u.normalize_to_metadata(test_dict)
        tm.that(result, is_=dict)

    def test_normalize_list_to_pydantic_model(self) -> None:
        test_list: t.JsonValue = [1, 2, 3]
        result = u.normalize_to_metadata(test_list)
        tm.that(result, is_=list)

    def test_normalize_dict_with_primitives(self) -> None:
        test_dict: t.JsonValue = {"a": 1, "b": "test", "c": True}
        result = u.normalize_to_metadata(test_dict)
        tm.that(result, is_=dict)

    def test_normalize_dict_with_models(self) -> None:
        class InnerModel(m.Value):
            value: str = "nested"

        test_dict = InnerModel()
        result = u.normalize_to_metadata(test_dict)
        tm.that(result, is_=dict)
        assert isinstance(result, dict)
        tm.that(result, has="value")
        tm.that(result["value"], eq="nested")

    def test_normalize_dict_with_non_string_key(self) -> None:
        test_dict = {"123": "value", "key": "test"}
        result = u.normalize_to_metadata(test_dict)
        tm.that(result, is_=dict)
        assert isinstance(result, dict)
        tm.that(result, has="123")

    def test_normalize_list_with_primitives(self) -> None:
        test_list: t.JsonValue = ["a", 1, True]
        result = u.normalize_to_metadata(test_list)
        tm.that(result, is_=list)

    def test_normalize_list_with_complex_items(self) -> None:
        test_list: t.JsonValue = [
            "string",
            42,
            True,
            str(pathlib.Path("/test/path")),
        ]
        result = u.normalize_to_metadata(test_list)
        tm.that(result, is_=list)

    def test_normalize_tuple(self) -> None:
        test_tuple = (1, 2, 3)
        result = u.normalize_to_metadata(test_tuple)
        tm.that(result, is_=list)

    def test_normalize_dict_with_valid_flat_structure(self) -> None:
        test_dict: t.JsonValue = {
            "str": "value",
            "int": 42,
            "bool": True,
        }
        result = u.normalize_to_metadata(test_dict)
        tm.that(result, is_=dict)

    def test_normalize_custom_object(self) -> None:
        class CustomObject:
            @override
            def __str__(self) -> str:
                return "custom_object"

        obj = CustomObject()
        normalize_fn = getattr(u, "normalize_to_metadata")
        result = normalize_fn(obj)
        tm.that(result, is_=str)
        tm.that(result, eq="custom_object")

    def test_normalize_float_pi(self) -> None:
        result = u.normalize_to_metadata(math.pi)
        tm.that(result, is_=float)
        tm.that(result, eq=math.pi)

    def test_normalize_basemodel_passthrough(self) -> None:
        class SampleModel(m.Value):
            name: str = "test"

        model = SampleModel()
        result = u.normalize_to_metadata(model)
        tm.that(result, is_=dict)
        assert isinstance(result, dict)
        tm.that(result["name"], eq="test")


__all__: list[str] = ["TestUtilitiesTypeGuardsCoverage100"]
