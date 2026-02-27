"""Real tests for u.Mapper coverage - extract and accessors.

Module: flext_core._utilities.mapper
Scope: FlextUtilitiesMapper - extract, get, at, take, pick, as_, or_, flat, agg, etc.

This module provides comprehensive real tests to achieve 100% coverage for
FlextUtilitiesMapper, focusing on complex nested extraction and accessor patterns.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any, cast

import pytest
from flext_core import t, u
from flext_core.models import m
from pydantic import BaseModel

from tests.test_utils import assertion_helpers


@dataclass
class SimpleObj:
    """Simple test object."""

    name: str
    value: int


class ComplexModel(BaseModel):
    """Complex test model."""

    id: int
    data: dict[str, t.GeneralValueType]
    items: list[str]


class TestuMapperExtract:
    """Tests for u.mapper().extract."""

    def test_extract_dict_simple(self) -> None:
        """Test simple dict extraction."""
        data = {"a": 1, "b": 2}
        result = u.mapper().extract(data, "a")
        assertion_helpers.assert_flext_result_success(result)
        assert result.value == 1

    def test_extract_dict_nested(self) -> None:
        """Test nested dict extraction."""
        data = {"a": {"b": {"c": 3}}}
        result = u.mapper().extract(data, "a.b.c")
        assertion_helpers.assert_flext_result_success(result)
        assert result.value == 3

    def test_extract_object(self) -> None:
        """Test object attribute extraction."""
        obj = SimpleObj(name="test", value=42)
        result = u.mapper().extract(cast("Any", obj), "name")
        assertion_helpers.assert_flext_result_success(result)
        assert result.value == "test"

    def test_extract_model(self) -> None:
        """Test Pydantic model extraction."""
        model = ComplexModel(id=1, data={"key": "val"}, items=["a", "b"])
        result = u.mapper().extract(model, "data.key")
        assertion_helpers.assert_flext_result_success(result)
        assert result.value == "val"

    def test_extract_array_index(self) -> None:
        """Test array indexing."""
        data = {"items": [1, 2, 3]}
        result = u.mapper().extract(data, "items[1]")
        assertion_helpers.assert_flext_result_success(result)
        assert result.value == 2

    def test_extract_array_index_nested(self) -> None:
        """Test nested array indexing."""
        data = {"users": [{"name": "alice"}, {"name": "bob"}]}
        result = u.mapper().extract(data, "users[1].name")
        assertion_helpers.assert_flext_result_success(result)
        assert result.value == "bob"

    def test_extract_missing_default(self) -> None:
        """Test missing key with default."""
        data = {"a": 1}
        result = u.mapper().extract(data, "b", default=10)
        assertion_helpers.assert_flext_result_success(result)
        assert result.value == 10

    def test_extract_missing_required(self) -> None:
        """Test missing key required."""
        data = {"a": 1}
        result = u.mapper().extract(data, "b", required=True)
        assertion_helpers.assert_flext_result_failure(result)
        assert "not found" in str(result.error)

    def test_extract_array_index_error(self) -> None:
        """Test invalid array index."""
        data = {"items": [1]}
        result = u.mapper().extract(data, "items[5]", required=True)
        assertion_helpers.assert_flext_result_failure(result)
        # Check for 'out of range' or 'Invalid index' or 'not found' depending on impl
        msg = str(result.error)
        assert any(x in msg for x in ["out of range", "Invalid index", "not found"])

    def test_extract_none_path(self) -> None:
        """Test extraction when intermediate path is None."""
        data = {"a": None}
        result = u.mapper().extract(data, "a.b", default="defs")
        assertion_helpers.assert_flext_result_success(result)
        assert result.value == "defs"

        result_req = u.mapper().extract(data, "a.b", required=True)
        assert result_req.is_failure
        assert "is None" in str(result_req.error)


class TestuMapperAccessors:
    """Tests for u.Mapper accessors (get, at, take, pick)."""

    def test_get_simple(self) -> None:
        """Test get."""
        data = {"a": 1}
        assert u.mapper().get(data, "a") == 1
        assert u.mapper().get(data, "b", default=2) == 2

    def test_at_list(self) -> None:
        """Test at list."""
        items = [10, 20, 30]
        assert u.mapper().at(items, 1) == 20
        assert u.mapper().at(items, 5) is None
        assert u.mapper().at(items, 5, default=0) == 0

    def test_at_dict(self) -> None:
        """Test at dict."""
        items = {"a": 10}
        assert u.mapper().at(items, "a") == 10
        assert u.mapper().at(items, "b") is None

    def test_take_extraction(self) -> None:
        """Test take value extraction."""
        data = {"a": 1, "b": "str"}
        assert u.mapper().take(cast("Any", data), "a", as_type=int) == 1
        assert (
            u.mapper().take(cast("Any", data), "b", as_type=int, default=0) == 0
        )  # Type mismatch

    def test_take_slice(self) -> None:
        """Test take slicing."""
        items = [1, 2, 3, 4, 5]
        assert u.mapper().take(cast("Any", items), 2) == [1, 2]
        assert u.mapper().take(cast("Any", items), 2, from_start=False) == [4, 5]

        d = {"a": 1, "b": 2, "c": 3}
        # Dict order preserved in recent python
        taken = u.mapper().take(d, 2)
        assert len(taken) == 2
        assert "a" in taken and "b" in taken

    def test_pick_dict(self) -> None:
        """Test pick as dict."""
        data = {"a": 1, "b": 2, "c": 3}
        picked = u.mapper().pick(data, "a", "c")
        assert picked == {"a": 1, "c": 3}

    def test_pick_list(self) -> None:
        """Test pick as list."""
        data = {"a": 1, "b": 2, "c": 3}
        picked = u.mapper().pick(data, "a", "c", as_dict=False)
        assert picked == [1, 3]


class TestuMapperUtils:
    """Tests for u.Mapper utils (as_, or_, flat, agg)."""

    def test_as_conversion(self) -> None:
        """Test as_ type conversion."""
        assert u.mapper().as_("123", int) == 123
        assert u.mapper().as_("12.3", float) == pytest.approx(12.3)
        assert u.mapper().as_("true", bool) is True
        assert u.mapper().as_("invalid", int, default=0) == 0

    def test_as_strict(self) -> None:
        """Test as_ strict mode."""
        assert u.mapper().as_("123", int, strict=True, default=0) == 0
        assert u.mapper().as_(123, int, strict=True) == 123

    def test_or_fallback(self) -> None:
        """Test or_ fallback."""
        assert u.mapper().or_(None, 1, 2) == 1
        assert u.mapper().or_(None, None, default=3) == 3

    def test_flat(self) -> None:
        """Test flat."""
        items = [[1, 2], [3], []]
        assert u.mapper().flat(items) == [1, 2, 3]

    def test_agg(self) -> None:
        """Test agg."""
        items = [{"v": 10}, {"v": 20}]
        assert u.mapper().agg(items, "v") == 30
        assert u.mapper().agg(items, operator.itemgetter("v"), fn=max) == 20


class TestuMapperConversions:
    """Tests for u.Mapper conversions (ensure, convert)."""

    def test_ensure_str(self) -> None:
        """Test ensure_str."""
        assert u.mapper().ensure_str("s") == "s"
        assert u.mapper().ensure_str(1) == "1"
        assert u.mapper().ensure_str(None, "def") == "def"

    def test_ensure_list(self) -> None:
        """Test ensure."""
        assert u.mapper().ensure(["a"]) == ["a"]
        assert u.mapper().ensure("a") == ["a"]
        assert u.mapper().ensure([1, 2]) == ["1", "2"]
        assert u.mapper().ensure(None) == []

    def test_ensure_str_or_none(self) -> None:
        """Test ensure_str_or_none."""
        assert u.mapper().ensure_str_or_none("s") == "s"
        assert u.mapper().ensure_str_or_none(1) is None
        assert u.mapper().ensure_str_or_none(None) is None

    def test_convert_to_json_value(self) -> None:
        """Test convert_to_json_value."""
        obj = SimpleObj("test", 1)
        # Pass dict directly - convert_to_json_value handles any dict
        # Purpose is to CONVERT arbitrary objects to JSON-safe format
        res = u.mapper().convert_to_json_value(cast("Any", {"obj": obj}))
        # Should convert obj to string representation
        assert isinstance(res, dict)
        assert "obj" in res
        assert "SimpleObj" in str(res["obj"])

    def test_convert_dict_to_json(self) -> None:
        """Test convert_dict_to_json - use convert_to_json_value for arbitrary objects."""
        d = {"a": SimpleObj("test", 1)}
        # Use convert_to_json_value which handles any dict
        res = u.mapper().convert_to_json_value(cast("Any", d))
        # Result should be a dict
        if isinstance(res, dict):
            assert isinstance(res["a"], str)
        else:
            msg = "Expected dict result"
            raise AssertionError(msg)

    def test_convert_list_to_json(self) -> None:
        """Test convert_list_to_json - use convert_to_json_value for arbitrary lists."""
        test_list = [{"a": SimpleObj("test", 1)}]
        # Use convert_to_json_value which handles any sequence
        res = u.mapper().convert_to_json_value(cast("Any", test_list))
        # Result should be a list
        if isinstance(res, list) and isinstance(res[0], dict):
            assert isinstance(res[0]["a"], str)
        else:
            msg = "Expected list of dicts result"
            raise AssertionError(msg)


class TestuMapperBuild:
    """Tests for u.Mapper build/construct/fields."""

    def test_build_pipeline(self) -> None:
        """Test build pipeline."""
        ops = {
            "ensure": "list",
            "map": lambda x: x * 2,
            "filter": lambda x: x > 2,  # Filter is applied BEFORE map
        }
        # Input: [1, 2, 3, 4]
        # Ensure: [1, 2, 3, 4]
        # Filter (x>2): [3, 4]
        # Map (x*2): [6, 8]
        res = u.mapper().build(
            [1, 2, 3, 4],
            ops=cast("dict[str, t.GeneralValueType] | None", ops),
        )
        assert res == [6, 8]

    def test_build_all_ops(self) -> None:
        """Test all build operations."""
        input_data = [1, 2, 1, 3, 4]

        ops = {
            "ensure": "list",
            "filter": lambda x: x > 1,  # [2, 3, 4] (removed 1s)
            "map": lambda x: x * 10,  # [20, 30, 40]
            "process": lambda x: x + 5,  # [25, 35, 45]
            "sort": True,  # [25, 35, 45]
            "unique": True,  # [25, 35, 45]
            "slice": (0, 2),  # [25, 35]
        }
        res = u.mapper().build(
            input_data,
            ops=cast("dict[str, t.GeneralValueType] | None", ops),
        )
        assert res == [25, 35]

    def test_build_normalize(self) -> None:
        """Test build normalize."""
        res = u.mapper().build(
            ["A", "b"],
            ops=cast("dict[str, t.GeneralValueType] | None", {"normalize": "lower"}),
        )
        assert res == ["a", "b"]

    def test_build_group(self) -> None:
        """Test build group - keys are converted to strings for ConfigurationDict."""
        res = u.mapper().build(
            ["cat", "dog", "ant"],
            ops=cast("dict[str, t.GeneralValueType] | None", {"group": len}),
        )
        # Keys are converted to strings because result is ConfigurationDict
        assert res == {"3": ["cat", "dog", "ant"]}

    def test_build_chunk(self) -> None:
        """Test build chunk."""
        res = u.mapper().build(
            [1, 2, 3, 4],
            ops=cast("dict[str, t.GeneralValueType] | None", {"chunk": 2}),
        )
        assert res == [[1, 2], [3, 4]]

    def test_fields_single(self) -> None:
        """Test fields single extraction."""
        data = {"a": 1}
        assert u.mapper().field(data, "a") == 1

    def test_fields_multi(self) -> None:
        """Test fields multi extraction."""
        data = {"a": 1, "b": 2}
        spec = {"a": None, "b": None}
        res = u.mapper().fields_multi(data, cast("dict[str, t.GeneralValueType]", spec))
        assert res == {"a": 1, "b": 2}

    def test_construct(self) -> None:
        """Test construct."""
        source = {"user_name": "john", "user_age": 30}
        spec = {
            "name": "user_name",
            "age": "user_age",
            "role": {"value": "REDACTED_LDAP_BIND_PASSWORD"},
        }
        res = u.mapper().construct(
            cast("dict[str, t.ConfigMapValue]", spec),
            m.ConfigMap(root=cast("dict[str, t.ConfigMapValue]", source)),
        )
        assert res == {"name": "john", "age": 30, "role": "REDACTED_LDAP_BIND_PASSWORD"}


class TestuMapperAdvanced:
    """Advanced tests for u.Mapper to reach 100% coverage."""

    def test_model_dump_extraction(self) -> None:
        """Test extraction via model_dump."""

        class Dumpable(BaseModel):
            a: int = 1

        obj = Dumpable()
        assert u.mapper().extract(cast("Any", obj), "a").value == 1
        assert u.mapper().extract(cast("Any", obj), "b", default=2).value == 2

    def test_convert_exception(self) -> None:
        """Test build convert exception handling."""
        # Convert fails -> returns default (which is convert_type() -> int() -> 0)
        res = u.mapper().build(
            "invalid",
            ops=cast("dict[str, t.GeneralValueType] | None", {"convert": int}),
        )
        assert res == 0

        # With custom default
        res = u.mapper().build(
            "invalid",
            ops=cast(
                "dict[str, t.GeneralValueType] | None",
                {"convert": int, "convert_default": 10},
            ),
        )
        assert res == 10

    def test_transform_options(self) -> None:
        """Test build transform options."""
        # Normalize, strip_none, etc.
        data = {"a": "UPPER", "b": None, "c": ""}
        ops = {
            "transform": {
                "normalize": True,
                "strip_none": True,
                "strip_empty": True,
            },
        }
        res = u.mapper().build(
            data, ops=cast("dict[str, t.GeneralValueType] | None", ops),
        )
        # c stripped (empty), b stripped (None). 'a' preserved (cache normalization doesn't lowercase values)
        assert res == {"a": "UPPER"}

    def test_build_sort_complex(self) -> None:
        """Test build sort with callable and string."""
        data = [{"a": 2}, {"a": 1}]
        res = u.mapper().build(
            data,
            ops=cast("dict[str, t.GeneralValueType] | None", {"sort": "a"}),
        )
        assert cast("list[dict[str, int]]", res)[0]["a"] == 1

        res = u.mapper().build(
            data,
            ops=cast(
                "dict[str, t.GeneralValueType] | None",
                {"sort": operator.itemgetter("a")},
            ),
        )
        assert cast("list[dict[str, int]]", res)[0]["a"] == 1

    def test_build_unique(self) -> None:
        """Test build unique."""
        data = [1, 2, 1, 3]
        res = u.mapper().build(data, ops={"unique": True})
        assert res == [1, 2, 3]

    def test_agg_branches(self) -> None:
        """Test agg branches."""
        data = [{"v": 1}, {"v": "str"}]  # "str" ignored
        assert u.mapper().agg(data, "v") == 1
        assert u.mapper().agg([], "v") == 0
