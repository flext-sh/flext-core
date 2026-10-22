"""Behavioral contract tests for the public mapper utilities.

Exercises the observable contract of ``u.extract`` / ``u.transform`` /
``u.agg`` / ``u.deep_eq`` / ``u.prop`` — return values, ``r[T]`` success and
failure outcomes, and edge-case invariants a caller depends on. No private
attributes, collaborators, or internal data structures are inspected.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import operator

import pytest

from flext_core import m, t
from tests import tm, u


class TestsFlextCoreMapper:
    """Public-surface behavior of the FLEXT mapper utilities."""

    # ------------------------------------------------------------------ extract
    @staticmethod
    def test_extract_returns_value_for_simple_key() -> None:
        result = u.extract({"a": 1}, "a")

        tm.that(result.success, truth=True)
        tm.that(result.value, eq=1)

    @staticmethod
    def test_extract_resolves_nested_dot_path() -> None:
        result = u.extract({"a": {"b": {"c": 42}}}, "a.b.c")

        tm.that(result.success, truth=True)
        tm.that(result.value, eq=42)

    @staticmethod
    def test_extract_missing_key_with_default_succeeds_with_default() -> None:
        result = u.extract({"a": 1}, "missing", default=0)

        tm.that(result.success, truth=True)
        tm.that(result.value, eq=0)

    @staticmethod
    def test_extract_missing_key_without_default_fails() -> None:
        result = u.extract({"a": 1}, "missing")

        tm.that(result.failure, truth=True)
        tm.that(result.error, none=False)
        tm.that("missing" in str(result.error), truth=True)

    @staticmethod
    def test_extract_required_missing_key_fails() -> None:
        result = u.extract({"a": 1}, "missing", required=True)

        tm.that(result.failure, truth=True)
        tm.that(result.error, none=False)
        tm.that("not found" in str(result.error), truth=True)

    @staticmethod
    def test_extract_honors_custom_separator() -> None:
        result = u.extract({"a": {"b": 7}}, "a/b", separator="/")

        tm.that(result.success, truth=True)
        tm.that(result.value, eq=7)

    # ---------------------------------------------------------------------- agg
    @staticmethod
    def test_agg_sums_field_by_default() -> None:
        tm.that(u.agg([{"x": 1}, {"x": 2}, {"x": 3}], "x"), eq=6)

    @staticmethod
    def test_agg_applies_custom_reducer() -> None:
        def max_numeric(values: t.SequenceOf[t.Numeric]) -> t.Numeric:
            iterator = iter(values)
            highest = next(iterator)
            for value in iterator:
                highest = max(highest, value)
            return highest

        tm.that(u.agg([{"x": 1}, {"x": 5}, {"x": 3}], "x", fn=max_numeric), eq=5)

    @staticmethod
    def test_agg_empty_sequence_returns_zero() -> None:
        tm.that(u.agg([], "x"), eq=0)

    @staticmethod
    def test_agg_accepts_callable_field_selector() -> None:
        tm.that(u.agg([{"x": 1}, {"x": 2}], operator.itemgetter("x")), eq=3)

    @staticmethod
    def test_agg_ignores_non_numeric_and_absent_field_values() -> None:
        tm.that(u.agg([{"x": 1}, {"x": "skip"}, {"y": 9}], "x"), eq=1)

    # ------------------------------------------------------------------ deep_eq
    @staticmethod
    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            ({"a": {"b": [1, 2]}}, {"a": {"b": [1, 2]}}, True),
            ({"a": 1}, {"a": 2}, False),
            ({"a": 1}, {"a": 1, "b": 2}, False),
            ({"a": [1, 2]}, {"a": [1, 2, 3]}, False),
            ({}, {}, True),
        ],
    )
    def test_deep_eq_reports_structural_equality(
        left: t.JsonDict,
        right: t.JsonDict,
        *,
        expected: bool,
    ) -> None:
        tm.that(u.deep_eq(left, right) is expected, truth=True)

    @staticmethod
    def test_deep_eq_is_reflexive() -> None:
        payload: t.JsonDict = {"a": {"b": [1, {"c": 2}]}}

        tm.that(u.deep_eq(payload, payload) is True, truth=True)

    # --------------------------------------------------------------------- prop
    @staticmethod
    def test_prop_returns_accessor_extracting_named_value() -> None:
        accessor = u.prop("name")

        tm.that(accessor({"name": "zed"}), eq="zed")

    @staticmethod
    def test_prop_accessor_returns_empty_string_for_missing_key() -> None:
        accessor = u.prop("name")

        tm.that(accessor({"other": 1}), truth=False)

    # ---------------------------------------------------------------- transform
    @staticmethod
    def test_transform_without_options_returns_equivalent_mapping() -> None:
        result = u.transform({"a": 1, "b": 2})

        tm.that(result.success, truth=True)
        tm.that(result.value, eq={"a": 1, "b": 2})

    @staticmethod
    def test_transform_strip_none_removes_none_values() -> None:
        result = u.transform(
            {"a": 1, "b": None},
            options=m.TransformOptions(strip_none=True),
        )

        tm.that(result.success, truth=True)
        tm.that(result.value, eq={"a": 1})

    @staticmethod
    def test_transform_strip_empty_removes_empty_values() -> None:
        result = u.transform(
            {"a": "", "b": 2},
            options=m.TransformOptions(strip_empty=True),
        )

        tm.that(result.success, truth=True)
        tm.that(result.value, eq={"b": 2})

    @staticmethod
    def test_transform_map_keys_renames_keys() -> None:
        result = u.transform(
            {"a": 1},
            options=m.TransformOptions(map_keys={"a": "renamed"}),
        )

        tm.that(result.success, truth=True)
        tm.that(result.value, eq={"renamed": 1})

    @staticmethod
    def test_transform_filter_keys_keeps_only_selected() -> None:
        result = u.transform({"a": 1, "b": 2, "c": 3}, filter_keys={"a", "c"})

        tm.that(result.success, truth=True)
        tm.that(result.value, eq={"a": 1, "c": 3})

    @staticmethod
    def test_transform_exclude_keys_drops_selected() -> None:
        result = u.transform({"a": 1, "b": 2}, exclude_keys={"b"})

        tm.that(result.success, truth=True)
        tm.that(result.value, eq={"a": 1})

    @staticmethod
    def test_transform_pipeline_composes_rename_then_exclude() -> None:
        result = u.transform(
            {"alpha": {"beta": 1}, "drop": None},
            options=m.TransformOptions(
                normalize=True,
                map_keys={"alpha": "renamed"},
            ),
            exclude_keys={"drop"},
        )

        tm.that(result.success, truth=True)
        tm.that(result.value, eq={"renamed": {"beta": 1}})
