"""Tests for FlextUtilitiesMapper via the u facade.

Source: flext_core
Covers: extract, map_get, map_dict_keys,
        transform, take, agg, deep_eq, normalize_to_container, prop.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from typing import Annotated, cast

import pytest
from pydantic import Field

from flext_tests import tm
from tests import m, t, u


class TestsFlextCoreUtilitiesMapper:
    """Tests for flext_core via the u facade."""

    class Address(m.BaseModel):
        city: Annotated[str, Field(description="City name")]
        zip_code: Annotated[str, Field(description="Zip code")]

    class User(m.BaseModel):
        name: Annotated[str, Field(description="User name")]
        age: Annotated[int, Field(description="User age")]
        address: Annotated[
            TestsFlextCoreUtilitiesMapper.Address,
            Field(description="User address"),
        ]

    class Item(m.BaseModel):
        label: Annotated[str, Field(description="Item label")]
        value: Annotated[int, Field(description="Item value")]

    # ── extract: basic dict ────────────────────────────────────────

    def test_extract_simple_key(self) -> None:
        data: t.RecursiveContainerMapping = {"name": "alice", "score": 42}
        result = u.extract(data, "name")
        tm.ok(result, eq="alice")

    def test_extract_nested_dict(self) -> None:
        data: t.RecursiveContainerMapping = {"a": {"b": {"c": "deep"}}}
        tm.ok(u.extract(data, "a.b.c"), eq="deep")

    def test_extract_missing_key_default(self) -> None:
        data: t.RecursiveContainerMapping = {"a": 1}
        tm.ok(u.extract(data, "missing", default="fallback"), eq="fallback")

    def test_extract_missing_key_required_fails(self) -> None:
        data: t.RecursiveContainerMapping = {"a": 1}
        tm.fail(u.extract(data, "missing", required=True), has="not found")

    # ── extract: array index ───────────────────────────────────────

    def test_extract_array_index(self) -> None:
        data: t.RecursiveContainerMapping = {"items": [10, 20, 30]}
        tm.ok(u.extract(data, "items[1]"), eq=20)

    def test_extract_negative_array_index(self) -> None:
        data: t.RecursiveContainerMapping = {"items": [10, 20, 30]}
        tm.ok(u.extract(data, "items[-1]"), eq=30)

    def test_extract_array_out_of_range(self) -> None:
        data: t.RecursiveContainerMapping = {"items": [1]}
        result = u.extract(data, "items[99]", required=True)
        tm.fail(result)

    def test_extract_nested_array_field(self) -> None:
        data: t.RecursiveContainerMapping = {
            "users": [{"name": "alice"}, {"name": "bob"}],
        }
        tm.ok(u.extract(data, "users[1].name"), eq="bob")

    # ── extract: pydantic model ────────────────────────────────────

    def test_extract_from_model(self) -> None:
        user = self.User(
            name="alice",
            age=30,
            address=self.Address(city="NYC", zip_code="10001"),
        )
        tm.ok(u.extract(user, "name"), eq="alice")

    def test_extract_model_nested_dict(self) -> None:
        """Model with dict attribute - extract nested key."""

        class WithData(m.BaseModel):
            data: Annotated[
                t.RecursiveContainerMapping,
                Field(description="Data dict"),
            ] = {"key": "val"}

        obj = WithData()
        tm.ok(u.extract(obj, "data.key"), eq="val")

    # ── extract: None intermediate ─────────────────────────────────

    def test_extract_none_intermediate_with_default(self) -> None:
        data: Mapping[str, None] = {"a": None}
        tm.ok(u.extract(data, "a.b", default="safe"), eq="safe")

    def test_extract_none_intermediate_required_fails(self) -> None:
        data: Mapping[str, None] = {"a": None}
        tm.fail(u.extract(data, "a.b", required=True), has="not found")

    # ── extract: custom separator ──────────────────────────────────

    def test_extract_custom_separator(self) -> None:
        data: t.RecursiveContainerMapping = {"a": {"b": {"c": "val"}}}
        tm.ok(u.extract(data, "a/b/c", separator="/"), eq="val")

    # ── map_get ────────────────────────────────────────────────────

    def test_map_get_existing_key(self) -> None:
        data: t.RecursiveContainerMapping = {"x": 42}
        tm.that(u.map_get(data, "x"), eq=42)

    def test_map_get_missing_key_default(self) -> None:
        data: t.RecursiveContainerMapping = {"x": 42}
        tm.that(u.map_get(data, "y", default="nope"), eq="nope")

    def test_map_get_missing_key_no_default(self) -> None:
        data: t.RecursiveContainerMapping = {"x": 42}
        result = u.map_get(data, "y")
        tm.that(result, eq="")

    # ── map_dict_keys ──────────────────────────────────────────────

    def test_map_dict_keys_basic(self) -> None:
        source: t.RecursiveContainerMapping = {"old": "v1", "foo": "v2"}
        result = u.map_dict_keys(source, {"old": "new", "foo": "bar"})
        mapped = tm.ok(result)
        tm.that(mapped, kv={"new": "v1", "bar": "v2"})

    def test_map_dict_keys_keep_unmapped_true(self) -> None:
        source: t.RecursiveContainerMapping = {"old": "v1", "extra": "v2"}
        result = u.map_dict_keys(source, {"old": "new"}, keep_unmapped=True)
        mapped = tm.ok(result)
        tm.that(mapped, keys=["new", "extra"])

    def test_map_dict_keys_keep_unmapped_false(self) -> None:
        source: t.RecursiveContainerMapping = {"old": "v1", "extra": "v2"}
        result = u.map_dict_keys(source, {"old": "new"}, keep_unmapped=False)
        mapped = tm.ok(result)
        tm.that(mapped, keys=["new"])

    # ── deep_eq ────────────────────────────────────────────────────

    def test_deep_eq_identical(self) -> None:
        d: t.RecursiveContainerMapping = {"a": 1, "b": [2, 3]}
        tm.that(u.deep_eq(d, d), eq=True)

    def test_deep_eq_equal(self) -> None:
        a: t.RecursiveContainerMapping = {"x": {"y": [1, 2]}, "z": "val"}
        b: t.RecursiveContainerMapping = {"x": {"y": [1, 2]}, "z": "val"}
        tm.that(u.deep_eq(a, b), eq=True)

    def test_deep_eq_different_values(self) -> None:
        a: t.RecursiveContainerMapping = {"x": 1}
        b: t.RecursiveContainerMapping = {"x": 2}
        tm.that(u.deep_eq(a, b), eq=False)

    def test_deep_eq_different_keys(self) -> None:
        a: t.RecursiveContainerMapping = {"x": 1}
        b: t.RecursiveContainerMapping = {"y": 1}
        tm.that(u.deep_eq(a, b), eq=False)

    def test_deep_eq_different_lengths(self) -> None:
        a: t.RecursiveContainerMapping = {"x": 1}
        b: t.RecursiveContainerMapping = {"x": 1, "y": 2}
        tm.that(u.deep_eq(a, b), eq=False)

    def test_deep_eq_nested_list_mismatch(self) -> None:
        a: t.RecursiveContainerMapping = {"items": [1, 2]}
        b: t.RecursiveContainerMapping = {"items": [1, 3]}
        tm.that(u.deep_eq(a, b), eq=False)

    def test_deep_eq_none_handling(self) -> None:
        a: t.RecursiveContainerMapping = {"x": None}
        b: t.RecursiveContainerMapping = {"x": None}
        tm.that(u.deep_eq(a, b), eq=True)

    def test_deep_eq_none_vs_value(self) -> None:
        a: t.RecursiveContainerMapping = {"x": None}
        b: t.RecursiveContainerMapping = {"x": 1}
        tm.that(u.deep_eq(a, b), eq=False)

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
        item = self.Item(label="test", value=5)
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
        data: t.RecursiveContainerMapping = {"a": 1, "b": "two"}
        tm.that(u.take(data, "a"), eq=1)

    def test_take_by_key_with_type(self) -> None:
        data: t.RecursiveContainerMapping = {"a": 1, "b": "two"}
        tm.that(u.take(data, "a", as_type=int), eq=1)
        tm.that(u.take(data, "b", as_type=int, default=0), eq=0)

    def test_take_by_key_missing(self) -> None:
        data: t.RecursiveContainerMapping = {"a": 1}
        tm.that(u.take(data, "z", default="nope"), eq="nope")

    # ── take: N items ──────────────────────────────────────────────

    def test_take_n_from_list(self) -> None:
        items: t.RecursiveContainerList = [10, 20, 30, 40, 50]
        tm.that(u.take(items, 2), eq=[10, 20])

    def test_take_n_from_list_end(self) -> None:
        items: t.RecursiveContainerList = [10, 20, 30, 40, 50]
        tm.that(u.take(items, 2, from_start=False), eq=[40, 50])

    def test_take_n_from_dict(self) -> None:
        data: t.RecursiveContainerMapping = {"a": 1, "b": 2, "c": 3}
        taken = u.take(data, 2)
        tm.that(taken, is_=dict)
        assert isinstance(taken, Mapping)
        tm.that(len(taken), eq=2)

    def test_take_non_collection_returns_default(self) -> None:
        tm.that(u.take("scalar", 2, default="def"), eq="def")

    # ── agg ────────────────────────────────────────────────────────

    def test_agg_sum_field_name(self) -> None:
        items: list[t.RecursiveContainerMapping] = [{"v": 10}, {"v": 20}]
        tm.that(u.agg(items, "v"), eq=30)

    def test_agg_with_custom_fn(self) -> None:
        items: list[t.RecursiveContainerMapping] = [{"v": 10}, {"v": 20}, {"v": 5}]
        tm.that(u.agg(items, "v", fn=max), eq=20)

    def test_agg_with_callable_extractor(self) -> None:
        items: list[t.RecursiveContainerMapping] = [{"v": 3}, {"v": 7}]

        def get_v(item: t.RecursiveContainerMapping) -> t.Numeric:
            val = item.get("v")
            return val if isinstance(val, (int, float)) else 0

        tm.that(u.agg(items, get_v), eq=10)

    def test_agg_empty_list(self) -> None:
        tm.that(u.agg([], "v"), eq=0)

    def test_agg_missing_field(self) -> None:
        items: list[t.RecursiveContainerMapping] = [{"x": 10}]
        tm.that(u.agg(items, "v"), eq=0)

    def test_agg_pydantic_model(self) -> None:
        items = [self.Item(label="a", value=5), self.Item(label="b", value=15)]
        tm.that(u.agg(items, "value"), eq=20)

    # ── transform (top-level static) ───────────────────────────────

    def test_transform_strip_none(self) -> None:
        source: t.RecursiveContainerMapping = {"a": 1, "b": None}
        result = u.transform(source, strip_none=True)
        mapped = tm.ok(result)
        tm.that(mapped, eq={"a": 1})

    def test_transform_strip_empty(self) -> None:
        source: t.RecursiveContainerMapping = {"a": 1, "b": ""}
        result = u.transform(source, strip_empty=True)
        mapped = tm.ok(result)
        tm.that(mapped, eq={"a": 1})

    def test_transform_map_keys(self) -> None:
        source: t.RecursiveContainerMapping = {"old": "val"}
        result = u.transform(source, map_keys={"old": "new"})
        mapped = tm.ok(result)
        tm.that(mapped, kv={"new": "val"})

    def test_transform_filter_keys(self) -> None:
        source: t.RecursiveContainerMapping = {"a": 1, "b": 2, "c": 3}
        result = u.transform(source, filter_keys={"a", "c"})
        mapped = tm.ok(result)
        tm.that(mapped, kv={"a": 1, "c": 3})

    def test_transform_exclude_keys(self) -> None:
        source: t.RecursiveContainerMapping = {"a": 1, "b": 2, "c": 3}
        result = u.transform(source, exclude_keys={"b"})
        mapped = tm.ok(result)
        tm.that(mapped, eq={"a": 1, "c": 3})

    def test_transform_combined(self) -> None:
        source: t.RecursiveContainerMapping = {"a": 1, "b": None, "c": ""}
        result = u.transform(source, strip_none=True, strip_empty=True)
        mapped = tm.ok(result)
        tm.that(mapped, eq={"a": 1})

    # ── edge cases ─────────────────────────────────────────────────

    def test_extract_empty_dict(self) -> None:
        result = u.extract({}, "any", default="fallback")
        tm.ok(result, eq="fallback")

    def test_deep_eq_empty_dicts(self) -> None:
        tm.that(u.deep_eq({}, {}), eq=True)

    def test_deep_eq_nested_none_vs_none(self) -> None:
        a: t.RecursiveContainerMapping = {
            "x": cast("t.RecursiveContainerMapping", {"y": None})
        }
        b: t.RecursiveContainerMapping = {
            "x": cast("t.RecursiveContainerMapping", {"y": None})
        }
        tm.that(u.deep_eq(a, b), eq=True)

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
        data: t.RecursiveContainerMapping,
        path: str,
        expected: t.RecursiveContainer,
    ) -> None:
        tm.ok(u.extract(data, path), eq=expected)

    # ── map_get from object with attribute ─────────────────────────

    def test_map_get_from_object(self) -> None:
        item = self.Item(label="test", value=99)
        tm.that(u.map_get(item, "label"), eq="test")

    # ── deep_eq: list length mismatch ──────────────────────────────

    def test_deep_eq_list_length_mismatch(self) -> None:
        a: t.RecursiveContainerMapping = {"x": [1, 2]}
        b: t.RecursiveContainerMapping = {"x": [1, 2, 3]}
        tm.that(u.deep_eq(a, b), eq=False)
