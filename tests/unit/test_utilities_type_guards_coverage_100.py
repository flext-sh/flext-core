"""Behavior contract for flext_core.utilities type guards + normalize_to_metadata."""

from __future__ import annotations

import math
from typing import Annotated

import pytest
from flext_tests import tm

from tests import m, t, u


class TestsFlextCoreUtilitiesTypeGuards:
    """Behavior contract for u.string_non_empty / u.dict_non_empty / u.matches_type / u.normalize_to_metadata."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("test", True),
            ("", False),
            ("   ", False),
            ("123", True),
            ("!@#$%^&*()", True),
            ("日本語", True),
            ("\n", False),
            ("\t", False),
        ],
    )
    def test_string_non_empty_detects_meaningful_strings(
        self,
        value: str,
        expected: bool,
    ) -> None:
        tm.that(u.string_non_empty(value), eq=expected)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ({"k": "v"}, True),
            ({}, False),
        ],
    )
    def test_dict_non_empty_detects_populated_dicts(
        self,
        value: t.StrMapping,
        expected: bool,
    ) -> None:
        tm.that(u.dict_non_empty(value), eq=expected)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ([1, 2, 3], True),
            ([], False),
            ([0], True),
            ("not-a-list", False),
            (123, False),
        ],
    )
    def test_matches_type_list_non_empty_rejects_non_list_and_empty(
        self,
        value: t.JsonValue,
        expected: bool,
    ) -> None:
        tm.that(u.matches_type(value, "list_non_empty"), eq=expected)

    @pytest.mark.parametrize(
        ("value", "expected_type"),
        [
            ("test", str),
            (42, int),
            (math.pi, float),
            (True, bool),
            (False, bool),
            ("", str),
        ],
    )
    def test_normalize_to_metadata_preserves_primitive_types(
        self,
        value: t.Scalar,
        expected_type: type,
    ) -> None:
        result = u.normalize_to_metadata(value)
        tm.that(result, is_=expected_type)

    def test_normalize_to_metadata_converts_dict_to_dict(self) -> None:
        result = u.normalize_to_metadata({"key": "value", "num": 42})
        tm.that(result, is_=dict)

    def test_normalize_to_metadata_converts_list_to_list(self) -> None:
        result = u.normalize_to_metadata([1, 2, 3])
        tm.that(result, is_=list)

    def test_normalize_to_metadata_converts_tuple_to_list(self) -> None:
        result = u.normalize_to_metadata((1, 2, 3))
        tm.that(result, is_=list)

    def test_normalize_to_metadata_dumps_basemodel_to_dict(self) -> None:
        class SampleModel(m.Value):
            name: Annotated[str, m.Field(default="test")]

        result = u.normalize_to_metadata(SampleModel())
        tm.that(result, is_=dict)
        assert isinstance(result, dict)
        tm.that(result["name"], eq="test")
