"""Tests for FlextUtilitiesMapper via the u facade.

Source: flext_core._utilities.mapper (~1479 LOC)
Covers: extract, map_get, map_dict_keys, filter_dict, build DSL,
        transform, take, agg, deep_eq, ensure_str, narrow_to_container, prop.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping
from typing import Annotated, cast

import pytest
from pydantic import BaseModel, Field

from flext_tests import tm
from tests import t, u


# ── Helper models ──────────────────────────────────────────────────
class _Address(BaseModel):
    city: Annotated[str, Field(description="City name")]
    zip_code: Annotated[str, Field(description="Zip code")]


class _User(BaseModel):
    name: Annotated[str, Field(description="User name")]
    age: Annotated[int, Field(description="User age")]
    address: Annotated[_Address, Field(description="User address")]


class _Item(BaseModel):
    label: Annotated[str, Field(description="Item label")]
    value: Annotated[int, Field(description="Item value")]


class _DoubleOp:
    """Callable that doubles numeric values."""

    def __call__(self, value: t.RecursiveContainer) -> t.RecursiveContainer:
        if isinstance(value, (int, float)):
            return value * 2
        return value


class _GtTwoOp:
    """Callable predicate: value > 2."""

    def __call__(self, value: t.RecursiveContainer) -> bool:
        if isinstance(value, (int, float)):
            return value > 2
        return False


class _FailingOp:
    """Callable that always raises."""

    def __call__(self, value: t.RecursiveContainer) -> t.RecursiveContainer:
        msg = "boom"
        raise RuntimeError(msg)


class TestFlextUtilitiesMapper:
    """Tests for flext_core._utilities.mapper via the u facade."""

    # ── extract: basic dict ────────────────────────────────────────

    def test_extract_simple_key(self) -> None:
        data: t.ContainerMapping = {"name": "alice", "score": 42}
        result = u.extract(data, "name")
        tm.ok(result, eq="alice")

    def test_extract_nested_dict(self) -> None:
        data: t.ContainerMapping = {"a": {"b": {"c": "deep"}}}
        tm.ok(u.extract(data, "a.b.c"), eq="deep")

    def test_extract_missing_key_default(self) -> None:
        data: t.ContainerMapping = {"a": 1}
        tm.ok(u.extract(data, "missing", default="fallback"), eq="fallback")

    def test_extract_missing_key_required_fails(self) -> None:
        data: t.ContainerMapping = {"a": 1}
        tm.fail(u.extract(data, "missing", required=True), has="not found")

    # ── extract: array index ───────────────────────────────────────

    def test_extract_array_index(self) -> None:
        data: t.ContainerMapping = {"items": [10, 20, 30]}
        tm.ok(u.extract(data, "items[1]"), eq=20)

    def test_extract_negative_array_index(self) -> None:
        data: t.ContainerMapping = {"items": [10, 20, 30]}
        tm.ok(u.extract(data, "items[-1]"), eq=30)

    def test_extract_array_out_of_range(self) -> None:
        data: t.ContainerMapping = {"items": [1]}
        result = u.extract(data, "items[99]", required=True)
        tm.fail(result)

    def test_extract_nested_array_field(self) -> None:
        data: t.ContainerMapping = {
            "users": [{"name": "alice"}, {"name": "bob"}],
        }
        tm.ok(u.extract(data, "users[1].name"), eq="bob")

    # ── extract: pydantic model ────────────────────────────────────

    def test_extract_from_model(self) -> None:
        user = _User(
            name="alice",
            age=30,
            address=_Address(city="NYC", zip_code="10001"),
        )
        tm.ok(u.extract(user, "name"), eq="alice")

    def test_extract_model_nested_dict(self) -> None:
        """Model with dict attribute - extract nested key."""

        class _WithData(BaseModel):
            data: Annotated[
                t.ContainerMapping,
                Field(description="Data dict"),
            ] = {"key": "val"}

        obj = _WithData()
        tm.ok(u.extract(obj, "data.key"), eq="val")

    # ── extract: None intermediate ─────────────────────────────────

    def test_extract_none_intermediate_with_default(self) -> None:
        data: Mapping[str, None] = {"a": None}
        tm.ok(u.extract(data, "a.b", default="safe"), eq="safe")

    def test_extract_none_intermediate_required_fails(self) -> None:
        data: Mapping[str, None] = {"a": None}
        tm.fail(u.extract(data, "a.b", required=True), has="None")

    # ── extract: custom separator ──────────────────────────────────

    def test_extract_custom_separator(self) -> None:
        data: t.ContainerMapping = {"a": {"b": {"c": "val"}}}
        tm.ok(u.extract(data, "a/b/c", separator="/"), eq="val")

    # ── map_get ────────────────────────────────────────────────────

    def test_map_get_existing_key(self) -> None:
        data: t.ContainerMapping = {"x": 42}
        tm.that(u.map_get(data, "x"), eq=42)

    def test_map_get_missing_key_default(self) -> None:
        data: t.ContainerMapping = {"x": 42}
        tm.that(u.map_get(data, "y", default="nope"), eq="nope")

    def test_map_get_missing_key_no_default(self) -> None:
        data: t.ContainerMapping = {"x": 42}
        result = u.map_get(data, "y")
        tm.that(result, eq="")

    # ── map_dict_keys ──────────────────────────────────────────────

    def test_map_dict_keys_basic(self) -> None:
        source: t.ContainerMapping = {"old": "v1", "foo": "v2"}
        result = u.map_dict_keys(source, {"old": "new", "foo": "bar"})
        mapped = tm.ok(result)
        tm.that(mapped, kv={"new": "v1", "bar": "v2"})

    def test_map_dict_keys_keep_unmapped_true(self) -> None:
        source: t.ContainerMapping = {"old": "v1", "extra": "v2"}
        result = u.map_dict_keys(source, {"old": "new"}, keep_unmapped=True)
        mapped = tm.ok(result)
        tm.that(mapped, keys=["new", "extra"])

    def test_map_dict_keys_keep_unmapped_false(self) -> None:
        source: t.ContainerMapping = {"old": "v1", "extra": "v2"}
        result = u.map_dict_keys(source, {"old": "new"}, keep_unmapped=False)
        mapped = tm.ok(result)
        tm.that(mapped, keys=["new"])

    # ── filter_dict ────────────────────────────────────────────────

    def test_filter_dict_by_value(self) -> None:
        source: t.ContainerMapping = {"a": 1, "b": 2, "c": 3}
        result = u.filter_dict(source, lambda _k, v: isinstance(v, int) and v > 1)
        tm.that(result, eq={"b": 2, "c": 3})

    def test_filter_dict_by_key(self) -> None:
        source: t.ContainerMapping = {"keep": 1, "drop": 2}
        result = u.filter_dict(source, lambda k, _v: k == "keep")
        tm.that(result, eq={"keep": 1})

    def test_filter_dict_empty_source(self) -> None:
        result = u.filter_dict({}, lambda _k, _v: True)
        tm.that(result, eq={})

    # ── deep_eq ────────────────────────────────────────────────────

    def test_deep_eq_identical(self) -> None:
        d: t.ContainerMapping = {"a": 1, "b": [2, 3]}
        tm.that(u.deep_eq(d, d), eq=True)

    def test_deep_eq_equal(self) -> None:
        a: t.ContainerMapping = {"x": {"y": [1, 2]}, "z": "val"}
        b: t.ContainerMapping = {"x": {"y": [1, 2]}, "z": "val"}
        tm.that(u.deep_eq(a, b), eq=True)

    def test_deep_eq_different_values(self) -> None:
        a: t.ContainerMapping = {"x": 1}
        b: t.ContainerMapping = {"x": 2}
        tm.that(u.deep_eq(a, b), eq=False)

    def test_deep_eq_different_keys(self) -> None:
        a: t.ContainerMapping = {"x": 1}
        b: t.ContainerMapping = {"y": 1}
        tm.that(u.deep_eq(a, b), eq=False)

    def test_deep_eq_different_lengths(self) -> None:
        a: t.ContainerMapping = {"x": 1}
        b: t.ContainerMapping = {"x": 1, "y": 2}
        tm.that(u.deep_eq(a, b), eq=False)

    def test_deep_eq_nested_list_mismatch(self) -> None:
        a: t.ContainerMapping = {"items": [1, 2]}
        b: t.ContainerMapping = {"items": [1, 3]}
        tm.that(u.deep_eq(a, b), eq=False)

    def test_deep_eq_none_handling(self) -> None:
        a: t.ContainerMapping = {"x": None}
        b: t.ContainerMapping = {"x": None}
        tm.that(u.deep_eq(a, b), eq=True)

    def test_deep_eq_none_vs_value(self) -> None:
        a: t.ContainerMapping = {"x": None}
        b: t.ContainerMapping = {"x": 1}
        tm.that(u.deep_eq(a, b), eq=False)

    # ── ensure_str ─────────────────────────────────────────────────

    @pytest.mark.parametrize(
        ("value", "default", "expected"),
        [
            ("hello", "", "hello"),
            (42, "", "42"),
            (None, "fallback", "fallback"),
            (None, "", ""),
            (math.pi, "", "3.14"),
            (True, "", "True"),
        ],
        ids=["string", "int", "none-fallback", "none-empty", "float", "bool"],
    )
    def test_ensure_str(
        self,
        value: t.NormalizedValue,
        default: str,
        expected: str,
    ) -> None:
        tm.that(u.ensure_str(value, default), eq=expected)

    # ── narrow_to_container ────────────────────────────────────────

    def test_narrow_none(self) -> None:
        tm.that(u.normalize_to_container(None), none=True)

    def test_narrow_string(self) -> None:
        tm.that(u.normalize_to_container("hello"), eq="hello")

    def test_narrow_int(self) -> None:
        tm.that(u.normalize_to_container(42), eq=42)

    def test_narrow_dict(self) -> None:
        result = u.normalize_to_container({"a": 1})
        tm.that(result, is_=dict)

    def test_narrow_list(self) -> None:
        result = u.normalize_to_container([1, 2, 3])
        tm.that(result, is_=list)

    def test_narrow_pydantic_model(self) -> None:
        item = _Item(label="test", value=5)
        result = u.normalize_to_container(item)
        tm.that(result, is_=dict)
        tm.that(result, kv={"label": "test", "value": 5})

    def test_narrow_ordered_dict(self) -> None:
        od: OrderedDict[str, int] = OrderedDict([("a", 1), ("b", 2)])
        result = u.normalize_to_container(od)
        tm.that(result, is_=dict)

    def test_narrow_tuple_becomes_list(self) -> None:
        result = u.normalize_to_container((1, 2, 3))
        tm.that(result, is_=list)

    # ── prop ───────────────────────────────────────────────────────

    def test_prop_accessor_dict(self) -> None:
        accessor = u.prop("name")
        data = t.ConfigMap({"name": "alice"})
        tm.that(accessor(data), eq="alice")

    def test_prop_accessor_missing_key(self) -> None:
        accessor = u.prop("missing")
        data = t.ConfigMap({"name": "alice"})
        tm.that(accessor(data), eq="")

    # ── take: by key ───────────────────────────────────────────────

    def test_take_by_key(self) -> None:
        data: t.ContainerMapping = {"a": 1, "b": "two"}
        tm.that(u.take(data, "a"), eq=1)

    def test_take_by_key_with_type(self) -> None:
        data: t.ContainerMapping = {"a": 1, "b": "two"}
        tm.that(u.take(data, "a", as_type=int), eq=1)
        tm.that(u.take(data, "b", as_type=int, default=0), eq=0)

    def test_take_by_key_missing(self) -> None:
        data: t.ContainerMapping = {"a": 1}
        tm.that(u.take(data, "z", default="nope"), eq="nope")

    # ── take: N items ──────────────────────────────────────────────

    def test_take_n_from_list(self) -> None:
        items: t.ContainerList = [10, 20, 30, 40, 50]
        tm.that(u.take(items, 2), eq=[10, 20])

    def test_take_n_from_list_end(self) -> None:
        items: t.ContainerList = [10, 20, 30, 40, 50]
        tm.that(u.take(items, 2, from_start=False), eq=[40, 50])

    def test_take_n_from_dict(self) -> None:
        data: t.ContainerMapping = {"a": 1, "b": 2, "c": 3}
        taken = u.take(data, 2)
        tm.that(taken, is_=dict)
        tm.that(len(taken), eq=2)

    def test_take_non_collection_returns_default(self) -> None:
        tm.that(u.take("scalar", 2, default="def"), eq="def")

    # ── agg ────────────────────────────────────────────────────────

    def test_agg_sum_field_name(self) -> None:
        items: list[t.ContainerMapping] = [{"v": 10}, {"v": 20}]
        tm.that(u.agg(items, "v"), eq=30)

    def test_agg_with_custom_fn(self) -> None:
        items: list[t.ContainerMapping] = [{"v": 10}, {"v": 20}, {"v": 5}]
        tm.that(u.agg(items, "v", fn=max), eq=20)

    def test_agg_with_callable_extractor(self) -> None:
        items: list[t.ContainerMapping] = [{"v": 3}, {"v": 7}]

        def get_v(item: t.ContainerMapping) -> t.Numeric:
            val = item.get("v")
            return val if isinstance(val, (int, float)) else 0

        tm.that(u.agg(items, get_v), eq=10)

    def test_agg_empty_list(self) -> None:
        tm.that(u.agg([], "v"), eq=0)

    def test_agg_missing_field(self) -> None:
        items: list[t.ContainerMapping] = [{"x": 10}]
        tm.that(u.agg(items, "v"), eq=0)

    def test_agg_pydantic_model(self) -> None:
        items = [_Item(label="a", value=5), _Item(label="b", value=15)]
        tm.that(u.agg(items, "value"), eq=20)

    # ── build DSL: ensure ──────────────────────────────────────────

    def test_build_no_ops_returns_value(self) -> None:
        result = u.build("hello")
        tm.that(result, eq="hello")

    def test_build_ensure_str(self) -> None:
        result = u.build(42, ops={"ensure": "str"})
        tm.that(result, eq="42")

    def test_build_ensure_str_none(self) -> None:
        result = u.build(None, ops={"ensure": "str"})
        tm.that(result, eq="")

    def test_build_ensure_list(self) -> None:
        result = u.build("single", ops={"ensure": "list"})
        tm.that(result, eq=["single"])

    def test_build_ensure_list_passthrough(self) -> None:
        result = u.build([1, 2], ops={"ensure": "list"})
        tm.that(result, eq=[1, 2])

    def test_build_ensure_str_list(self) -> None:
        result = u.build([1, 2], ops={"ensure": "str_list"})
        tm.that(result, eq=["1", "2"])

    def test_build_ensure_dict(self) -> None:
        result = u.build({"a": 1}, ops={"ensure": "dict"})
        tm.that(result, is_=dict)

    def test_build_ensure_dict_non_mapping(self) -> None:
        result = u.build("not_dict", ops={"ensure": "dict"})
        tm.that(result, is_=dict)

    # ── build DSL: filter ──────────────────────────────────────────

    def test_build_filter_list(self) -> None:
        gt_two = _GtTwoOp()
        result = u.build([1, 2, 3, 4], ops={"filter": gt_two})
        tm.that(result, eq=[3, 4])

    def test_build_filter_scalar_pass(self) -> None:
        gt_two = _GtTwoOp()
        result = u.build(5, ops={"filter": gt_two})
        tm.that(result, eq=5)

    def test_build_filter_scalar_fail(self) -> None:
        gt_two = _GtTwoOp()
        result = u.build(1, ops={"filter": gt_two}, default="gone")
        tm.that(result, eq="gone")

    # ── build DSL: map ─────────────────────────────────────────────

    def test_build_map_list(self) -> None:
        double = _DoubleOp()
        result = u.build([1, 2, 3], ops={"map": double})
        tm.that(result, eq=[2, 4, 6])

    def test_build_map_dict_values(self) -> None:
        double = _DoubleOp()
        result = u.build({"a": 1, "b": 2}, ops={"map": double})
        tm.that(result, kv={"a": 2, "b": 4})

    def test_build_map_scalar(self) -> None:
        double = _DoubleOp()
        result = u.build(5, ops={"map": double})
        tm.that(result, eq=10)

    # ── build DSL: normalize ───────────────────────────────────────

    def test_build_normalize_lower(self) -> None:
        result = u.build("HELLO", ops={"normalize": "lower"})
        tm.that(result, eq="hello")

    def test_build_normalize_upper(self) -> None:
        result = u.build("hello", ops={"normalize": "upper"})
        tm.that(result, eq="HELLO")

    def test_build_normalize_list(self) -> None:
        result = u.build(["Hello", "World"], ops={"normalize": "lower"})
        tm.that(result, eq=["hello", "world"])

    def test_build_normalize_non_string_passthrough(self) -> None:
        result = u.build(42, ops={"normalize": "lower"})
        tm.that(result, eq=42)

    # ── build DSL: convert ─────────────────────────────────────────

    def test_build_convert_to_int(self) -> None:
        result = u.build("42", ops=cast("t.ContainerMapping", {"convert": int}))
        tm.that(result, eq=42)

    def test_build_convert_list(self) -> None:
        result = u.build(
            ["1", "2", "3"], ops=cast("t.ContainerMapping", {"convert": int})
        )
        tm.that(result, eq=[1, 2, 3])

    def test_build_convert_failure_uses_default(self) -> None:
        result = u.build(
            "not_a_number",
            ops=cast("t.ContainerMapping", {"convert": int, "convert_default": -1}),
        )
        tm.that(result, eq=-1)

    def test_build_convert_auto_default_int(self) -> None:
        result = u.build("bad", ops=cast("t.ContainerMapping", {"convert": int}))
        tm.that(result, eq=0)

    # ── build DSL: sort ────────────────────────────────────────────

    def test_build_sort_true(self) -> None:
        result = u.build([3, 1, 2], ops={"sort": True})
        tm.that(result, eq=[1, 2, 3])

    def test_build_sort_by_field_name(self) -> None:
        items: t.ContainerList = [{"name": "bob"}, {"name": "alice"}]
        result = u.build(items, ops={"sort": "name"})
        tm.that(result, is_=list)
        first = result[0] if isinstance(result, list) else None
        tm.that(first, kv={"name": "alice"})

    def test_build_sort_non_list_passthrough(self) -> None:
        result = u.build("string", ops={"sort": True})
        tm.that(result, eq="string")

    # ── build DSL: unique ──────────────────────────────────────────

    def test_build_unique(self) -> None:
        result = u.build([1, 2, 2, 3, 1], ops={"unique": True})
        tm.that(result, eq=[1, 2, 3])

    def test_build_unique_preserves_order(self) -> None:
        result = u.build(["b", "a", "b", "c"], ops={"unique": True})
        tm.that(result, eq=["b", "a", "c"])

    def test_build_unique_false_noop(self) -> None:
        result = u.build([1, 1, 2], ops={"unique": False})
        tm.that(result, eq=[1, 1, 2])

    # ── build DSL: slice ───────────────────────────────────────────

    def test_build_slice(self) -> None:
        result = u.build([10, 20, 30, 40, 50], ops={"slice": (1, 3)})
        tm.that(result, eq=[20, 30])

    def test_build_slice_non_list_passthrough(self) -> None:
        result = u.build("hello", ops={"slice": (0, 2)})
        tm.that(result, eq="hello")

    # ── build DSL: chunk ───────────────────────────────────────────

    def test_build_chunk(self) -> None:
        result = u.build([1, 2, 3, 4, 5], ops={"chunk": 2})
        tm.that(result, eq=[[1, 2], [3, 4], [5]])

    def test_build_chunk_exact(self) -> None:
        result = u.build([1, 2, 3, 4], ops={"chunk": 2})
        tm.that(result, eq=[[1, 2], [3, 4]])

    def test_build_chunk_invalid_size(self) -> None:
        result = u.build([1, 2, 3], ops={"chunk": 0})
        tm.that(result, eq=[1, 2, 3])

    def test_build_chunk_non_list_passthrough(self) -> None:
        result = u.build("hello", ops={"chunk": 2})
        tm.that(result, eq="hello")

    # ── build DSL: group ───────────────────────────────────────────

    def test_build_group_by_field(self) -> None:
        items: t.ContainerList = [
            {"type": "a", "v": 1},
            {"type": "b", "v": 2},
            {"type": "a", "v": 3},
        ]
        result = u.build(items, ops={"group": "type"})
        tm.that(result, is_=dict)
        tm.that(result, keys=["a", "b"])

    def test_build_group_by_callable(self) -> None:
        class _LenGrouper:
            def __call__(self, value: t.RecursiveContainer) -> int:
                return len(value) if isinstance(value, str) else 0

        result = u.build(["hi", "hey", "yo"], ops={"group": _LenGrouper()})
        tm.that(result, is_=dict)

    def test_build_group_non_list_passthrough(self) -> None:
        result = u.build("string", ops={"group": "field"})
        tm.that(result, eq="string")

    # ── build DSL: process ─────────────────────────────────────────

    def test_build_process(self) -> None:
        double = _DoubleOp()
        result = u.build([1, 2, 3], ops={"process": double})
        tm.that(result, eq=[2, 4, 6])

    def test_build_process_error_stop(self) -> None:
        failing = _FailingOp()
        result = u.build(
            [1, 2], ops={"process": failing}, default="safe", on_error="stop"
        )
        tm.that(result, eq="safe")

    def test_build_process_error_continue(self) -> None:
        failing = _FailingOp()
        result = u.build([1, 2], ops={"process": failing}, on_error="continue")
        tm.that(result, eq=[1, 2])

    # ── build DSL: transform (dict ops) ────────────────────────────

    def test_build_transform_strip_none(self) -> None:
        data: t.ContainerMapping = {"a": 1, "b": None, "c": 3}
        result = u.build(data, ops={"transform": {"strip_none": True}})
        tm.that(result, eq={"a": 1, "c": 3})

    def test_build_transform_strip_empty(self) -> None:
        data: t.ContainerMapping = {
            "a": 1,
            "b": "",
            "c": cast("t.ContainerList", []),
            "d": cast("t.ContainerMapping", {}),
            "e": None,
        }
        result = u.build(data, ops={"transform": {"strip_empty": True}})
        tm.that(result, eq={"a": 1})

    def test_build_transform_map_keys(self) -> None:
        data: t.ContainerMapping = {"old": "v1"}
        result = u.build(data, ops={"transform": {"map_keys": {"old": "new"}}})
        tm.that(result, kv={"new": "v1"})

    def test_build_transform_non_mapping_passthrough(self) -> None:
        result = u.build("string", ops={"transform": {"strip_none": True}})
        tm.that(result, eq="string")

    # ── build DSL: pipeline composition ────────────────────────────

    def test_build_pipeline_ensure_map_filter(self) -> None:
        double = _DoubleOp()
        gt_two = _GtTwoOp()
        result = u.build(
            [1, 2, 3, 4],
            ops={"ensure": "list", "map": double, "filter": gt_two},
        )
        tm.that(result, eq=[6, 8])

    def test_build_pipeline_sort_unique_slice(self) -> None:
        result = u.build(
            [3, 1, 2, 3, 1, 4],
            ops={"unique": True, "sort": True, "slice": (0, 3)},
        )
        tm.that(result, eq=[1, 2, 3])

    # ── transform (top-level static) ───────────────────────────────

    def test_transform_strip_none(self) -> None:
        source: t.ContainerMapping = {"a": 1, "b": None}
        result = u.transform(source, strip_none=True)
        mapped = tm.ok(result)
        tm.that(mapped, eq={"a": 1})

    def test_transform_strip_empty(self) -> None:
        source: t.ContainerMapping = {"a": 1, "b": ""}
        result = u.transform(source, strip_empty=True)
        mapped = tm.ok(result)
        tm.that(mapped, eq={"a": 1})

    def test_transform_map_keys(self) -> None:
        source: t.ContainerMapping = {"old": "val"}
        result = u.transform(source, map_keys={"old": "new"})
        mapped = tm.ok(result)
        tm.that(mapped, kv={"new": "val"})

    def test_transform_filter_keys(self) -> None:
        source: t.ContainerMapping = {"a": 1, "b": 2, "c": 3}
        result = u.transform(source, filter_keys={"a", "c"})
        mapped = tm.ok(result)
        tm.that(mapped, kv={"a": 1, "c": 3})

    def test_transform_exclude_keys(self) -> None:
        source: t.ContainerMapping = {"a": 1, "b": 2, "c": 3}
        result = u.transform(source, exclude_keys={"b"})
        mapped = tm.ok(result)
        tm.that(mapped, eq={"a": 1, "c": 3})

    def test_transform_combined(self) -> None:
        source: t.ContainerMapping = {"a": 1, "b": None, "c": ""}
        result = u.transform(source, strip_none=True, strip_empty=True)
        mapped = tm.ok(result)
        tm.that(mapped, eq={"a": 1})

    # ── edge cases ─────────────────────────────────────────────────

    def test_extract_empty_dict(self) -> None:
        result = u.extract({}, "any", default="fallback")
        tm.ok(result, eq="fallback")

    def test_build_empty_list_chunk(self) -> None:
        result = u.build([], ops={"chunk": 3})
        tm.that(result, eq=[])

    def test_build_empty_list_unique(self) -> None:
        result = u.build([], ops={"unique": True})
        tm.that(result, eq=[])

    def test_build_empty_list_sort(self) -> None:
        result = u.build([], ops={"sort": True})
        tm.that(result, eq=[])

    def test_deep_eq_empty_dicts(self) -> None:
        tm.that(u.deep_eq({}, {}), eq=True)

    def test_deep_eq_nested_none_vs_none(self) -> None:
        a: t.ContainerMapping = {"x": cast("t.ContainerMapping", {"y": None})}
        b: t.ContainerMapping = {"x": cast("t.ContainerMapping", {"y": None})}
        tm.that(u.deep_eq(a, b), eq=True)

    @pytest.mark.parametrize(
        ("input_val", "ops", "expected"),
        [
            (None, {"ensure": "str"}, ""),
            (None, {"ensure": "list"}, []),
            (42, None, 42),
            ("hello", {"normalize": "upper"}, "HELLO"),
        ],
        ids=["none-to-str", "none-to-list", "no-ops", "normalize-upper"],
    )
    def test_build_parametrized(
        self,
        input_val: t.NormalizedValue,
        ops: t.ContainerMapping | None,
        expected: t.NormalizedValue,
    ) -> None:
        result = u.build(input_val, ops=ops)
        tm.that(result, eq=expected)

    @pytest.mark.parametrize(
        ("data", "path", "expected"),
        [
            ({"a": 1}, "a", 1),
            ({"a": {"b": 2}}, "a.b", 2),
            ({"items": [10, 20]}, "items[0]", 10),
            ({"items": [10, 20]}, "items[-1]", 20),
        ],
        ids=["simple", "nested", "index-0", "negative-index"],
    )
    def test_extract_parametrized(
        self,
        data: t.ContainerMapping,
        path: str,
        expected: t.NormalizedValue,
    ) -> None:
        tm.ok(u.extract(data, path), eq=expected)

    # ── sort with callable key ─────────────────────────────────────

    def test_build_sort_with_callable(self) -> None:
        class _Negator:
            def __call__(self, value: t.RecursiveContainer) -> t.RecursiveContainer:
                return -value if isinstance(value, (int, float)) else value

        result = u.build([3, 1, 2], ops={"sort": _Negator()})
        tm.that(result, eq=[3, 2, 1])

    # ── build with tuple input (narrowed to list) ──────────────────

    def test_build_tuple_narrowed_to_list(self) -> None:
        """Tuples are narrowed to lists by narrow_to_container."""
        result = u.build((1, 2, 2, 3), ops={"unique": True})
        tm.that(result, is_=list)
        tm.that(result, eq=[1, 2, 3])

    # ── ensure with defaults ───────────────────────────────────────

    def test_build_ensure_list_from_none(self) -> None:
        result = u.build(None, ops={"ensure": "list"})
        tm.that(result, eq=[])

    def test_build_ensure_str_list_from_scalar(self) -> None:
        result = u.build(42, ops={"ensure": "str_list"})
        tm.that(result, eq=["42"])

    def test_build_ensure_dict_returns_default_for_scalar(self) -> None:
        result = u.build(42, ops={"ensure": "dict"})
        tm.that(result, is_=dict)
        tm.that(result, eq={})

    # ── filter dict within build ───────────────────────────────────

    def test_build_filter_dict(self) -> None:
        gt_two = _GtTwoOp()
        data: t.ContainerMapping = {"a": 1, "b": 3, "c": 5}
        result = u.build(data, ops={"filter": gt_two})
        tm.that(result, eq={"b": 3, "c": 5})

    # ── map_get from object with attribute ─────────────────────────

    def test_map_get_from_object(self) -> None:
        item = _Item(label="test", value=99)
        tm.that(u.map_get(item, "label"), eq="test")

    # ── deep_eq: list length mismatch ──────────────────────────────

    def test_deep_eq_list_length_mismatch(self) -> None:
        a: t.ContainerMapping = {"x": [1, 2]}
        b: t.ContainerMapping = {"x": [1, 2, 3]}
        tm.that(u.deep_eq(a, b), eq=False)
