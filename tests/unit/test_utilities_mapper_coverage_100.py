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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

from pydantic import BaseModel

from flext_core import t, u


@dataclass
class SimpleObj:
    """Simple test object."""

    name: str
    value: int


class ComplexModel(BaseModel):
    """Complex test model."""

    id: int
    data: dict[str, Any]
    items: list[str]


class TestuMapperExtract:
    """Tests for u.Mapper.extract."""

    def test_extract_dict_simple(self) -> None:
        """Test simple dict extraction."""
        data = {"a": 1, "b": 2}
        result = u.Mapper.extract(data, "a")
        assert result.is_success
        assert result.value == 1

    def test_extract_dict_nested(self) -> None:
        """Test nested dict extraction."""
        data = {"a": {"b": {"c": 3}}}
        result = u.Mapper.extract(data, "a.b.c")
        assert result.is_success
        assert result.value == 3

    def test_extract_object(self) -> None:
        """Test object attribute extraction."""
        obj = SimpleObj(name="test", value=42)
        result = u.Mapper.extract(obj, "name")
        assert result.is_success
        assert result.value == "test"

    def test_extract_model(self) -> None:
        """Test Pydantic model extraction."""
        model = ComplexModel(id=1, data={"key": "val"}, items=["a", "b"])
        result = u.Mapper.extract(model, "data.key")
        assert result.is_success
        assert result.value == "val"

    def test_extract_array_index(self) -> None:
        """Test array indexing."""
        data = {"items": [1, 2, 3]}
        result = u.Mapper.extract(data, "items[1]")
        assert result.is_success
        assert result.value == 2

    def test_extract_array_index_nested(self) -> None:
        """Test nested array indexing."""
        data = {"users": [{"name": "alice"}, {"name": "bob"}]}
        result = u.Mapper.extract(data, "users[1].name")
        assert result.is_success
        assert result.value == "bob"

    def test_extract_missing_default(self) -> None:
        """Test missing key with default."""
        data = {"a": 1}
        result = u.Mapper.extract(data, "b", default=10)
        assert result.is_success
        assert result.value == 10

    def test_extract_missing_required(self) -> None:
        """Test missing key required."""
        data = {"a": 1}
        result = u.Mapper.extract(data, "b", required=True)
        assert result.is_failure
        assert "not found" in str(result.error)

    def test_extract_array_index_error(self) -> None:
        """Test invalid array index."""
        data = {"items": [1]}
        result = u.Mapper.extract(data, "items[5]", required=True)
        assert result.is_failure
        # Check for 'out of range' or 'Invalid index' or 'not found' depending on impl
        msg = str(result.error)
        assert any(x in msg for x in ["out of range", "Invalid index", "not found"])

    def test_extract_none_path(self) -> None:
        """Test extraction when intermediate path is None."""
        data = {"a": None}
        result = u.Mapper.extract(data, "a.b", default="defs")
        assert result.is_success
        assert result.value == "defs"

        result_req = u.Mapper.extract(data, "a.b", required=True)
        assert result_req.is_failure
        assert "is None" in str(result_req.error)


class TestuMapperAccessors:
    """Tests for u.Mapper accessors (get, at, take, pick)."""

    def test_get_simple(self) -> None:
        """Test get."""
        data = {"a": 1}
        assert u.Mapper.get(data, "a") == 1
        assert u.Mapper.get(data, "b", default=2) == 2

    def test_at_list(self) -> None:
        """Test at list."""
        items = [10, 20, 30]
        assert u.Mapper.at(items, 1) == 20
        assert u.Mapper.at(items, 5) is None
        assert u.Mapper.at(items, 5, default=0) == 0

    def test_at_dict(self) -> None:
        """Test at dict."""
        items = {"a": 10}
        assert u.Mapper.at(items, "a") == 10
        assert u.Mapper.at(items, "b") is None

    def test_take_extraction(self) -> None:
        """Test take value extraction."""
        data = {"a": 1, "b": "str"}
        assert u.Mapper.take(data, "a", as_type=int) == 1
        assert u.Mapper.take(data, "b", as_type=int, default=0) == 0  # Type mismatch

    def test_take_slice(self) -> None:
        """Test take slicing."""
        items = [1, 2, 3, 4, 5]
        assert u.Mapper.take(items, 2) == [1, 2]
        assert u.Mapper.take(items, 2, from_start=False) == [4, 5]

        d = {"a": 1, "b": 2, "c": 3}
        # Dict order preserved in recent python
        taken = u.Mapper.take(d, 2)
        assert len(taken) == 2
        assert "a" in taken and "b" in taken

    def test_pick_dict(self) -> None:
        """Test pick as dict."""
        data = {"a": 1, "b": 2, "c": 3}
        picked = u.Mapper.pick(data, "a", "c")
        assert picked == {"a": 1, "c": 3}

    def test_pick_list(self) -> None:
        """Test pick as list."""
        data = {"a": 1, "b": 2, "c": 3}
        picked = u.Mapper.pick(data, "a", "c", as_dict=False)
        assert picked == [1, 3]


class TestuMapperUtils:
    """Tests for u.Mapper utils (as_, or_, flat, agg)."""

    def test_as_conversion(self) -> None:
        """Test as_ type conversion."""
        assert u.Mapper.as_("123", int) == 123
        assert u.Mapper.as_("12.3", float) == 12.3
        assert u.Mapper.as_("true", bool) is True
        assert u.Mapper.as_("invalid", int, default=0) == 0

    def test_as_strict(self) -> None:
        """Test as_ strict mode."""
        assert u.Mapper.as_("123", int, strict=True, default=0) == 0
        assert u.Mapper.as_(123, int, strict=True) == 123

    def test_or_fallback(self) -> None:
        """Test or_ fallback."""
        assert u.Mapper.or_(None, 1, 2) == 1
        assert u.Mapper.or_(None, None, default=3) == 3

    def test_flat(self) -> None:
        """Test flat."""
        items = [[1, 2], [3], []]
        assert u.Mapper.flat(items) == [1, 2, 3]

    def test_agg(self) -> None:
        """Test agg."""
        items = [{"v": 10}, {"v": 20}]
        assert u.Mapper.agg(items, "v") == 30
        assert u.Mapper.agg(items, operator.itemgetter("v"), fn=max) == 20


class TestuMapperConversions:
    """Tests for u.Mapper conversions (ensure, convert)."""

    def test_ensure_str(self) -> None:
        """Test ensure_str."""
        assert u.Mapper.ensure_str("s") == "s"
        assert u.Mapper.ensure_str(1) == "1"
        assert u.Mapper.ensure_str(None, "def") == "def"

    def test_ensure_list(self) -> None:
        """Test ensure."""
        assert u.Mapper.ensure(["a"]) == ["a"]
        assert u.Mapper.ensure("a") == ["a"]
        assert u.Mapper.ensure([1, 2]) == ["1", "2"]
        assert u.Mapper.ensure(None) == []

    def test_ensure_str_or_none(self) -> None:
        """Test ensure_str_or_none."""
        assert u.Mapper.ensure_str_or_none("s") == "s"
        assert u.Mapper.ensure_str_or_none(1) is None
        assert u.Mapper.ensure_str_or_none(None) is None

    def test_convert_to_json_value(self) -> None:
        """Test convert_to_json_value."""
        obj = SimpleObj("test", 1)
        res = u.Mapper.convert_to_json_value(
            cast("t.Types.ConfigurationDict", {"obj": obj})
        )
        # Should convert obj to string
        assert isinstance(res, dict)
        assert "obj" in res
        assert "SimpleObj" in str(res["obj"])

    def test_convert_dict_to_json(self) -> None:
        """Test convert_dict_to_json."""
        d = {"a": SimpleObj("test", 1)}
        res = u.Mapper.convert_dict_to_json(cast("t.Types.ConfigurationDict", d))
        assert isinstance(res["a"], str)

    def test_convert_list_to_json(self) -> None:
        """Test convert_list_to_json."""
        test_list = [{"a": SimpleObj("test", 1)}]
        res = u.Mapper.convert_list_to_json(
            cast("Sequence[t.GeneralValueType]", test_list)
        )
        assert isinstance(res[0]["a"], str)


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
        res = u.Mapper.build(
            [1, 2, 3, 4], ops=cast("t.Types.ConfigurationDict | None", ops)
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
        res = u.Mapper.build(
            input_data, ops=cast("t.Types.ConfigurationDict | None", ops)
        )
        assert res == [25, 35]

    def test_build_normalize(self) -> None:
        """Test build normalize."""
        res = u.Mapper.build(
            ["A", "b"],
            ops=cast("t.Types.ConfigurationDict | None", {"normalize": "lower"}),
        )
        assert res == ["a", "b"]

    def test_build_group(self) -> None:
        """Test build group."""
        res = u.Mapper.build(
            ["cat", "dog", "ant"],
            ops=cast("t.Types.ConfigurationDict | None", {"group": len}),
        )
        assert res == {3: ["cat", "dog", "ant"]}

    def test_build_chunk(self) -> None:
        """Test build chunk."""
        res = u.Mapper.build(
            [1, 2, 3, 4], ops=cast("t.Types.ConfigurationDict | None", {"chunk": 2})
        )
        assert res == [[1, 2], [3, 4]]

    def test_fields_single(self) -> None:
        """Test fields single extraction."""
        data = {"a": 1}
        assert u.Mapper.field(data, "a") == 1

    def test_fields_multi(self) -> None:
        """Test fields multi extraction."""
        data = {"a": 1, "b": 2}
        spec = {"a": None, "b": None}
        res = u.Mapper.fields_multi(data, cast("dict[str, t.GeneralValueType]", spec))
        assert res == {"a": 1, "b": 2}

    def test_construct(self) -> None:
        """Test construct."""
        source = {"user_name": "john", "user_age": 30}
        spec = {
            "name": "user_name",
            "age": "user_age",
            "role": {"value": "REDACTED_LDAP_BIND_PASSWORD"},
        }
        res = u.Mapper.construct(
            cast("t.Types.ConfigurationDict", spec),
            cast("t.Types.ConfigurationDict", source),
        )
        assert res == {"name": "john", "age": 30, "role": "REDACTED_LDAP_BIND_PASSWORD"}


class TestuMapperAdvanced:
    """Advanced tests for u.Mapper to reach 100% coverage."""

    def test_model_dump_extraction(self) -> None:
        """Test extraction via model_dump."""

        class Dumpable:
            def model_dump(self) -> dict[str, int]:
                return {"a": 1}

        obj = Dumpable()
        assert u.Mapper.extract(obj, "a").value == 1
        assert u.Mapper.extract(obj, "b", default=2).value == 2

    def test_convert_exception(self) -> None:
        """Test build convert exception handling."""
        # Convert fails -> returns default (which is convert_type() -> int() -> 0)
        res = u.Mapper.build(
            "invalid", ops=cast("t.Types.ConfigurationDict | None", {"convert": int})
        )
        assert res == 0

        # With custom default
        res = u.Mapper.build(
            "invalid",
            ops=cast(
                "t.Types.ConfigurationDict | None",
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
        res = u.Mapper.build(data, ops=cast("t.Types.ConfigurationDict | None", ops))
        # c stripped (empty), b stripped (None). 'a' preserved (cache normalization doesn't lowercase values)
        assert res == {"a": "UPPER"}

    def test_build_sort_complex(self) -> None:
        """Test build sort with callable and string."""
        data = [{"a": 2}, {"a": 1}]
        res = u.Mapper.build(
            data, ops=cast("t.Types.ConfigurationDict | None", {"sort": "a"})
        )
        assert cast("list[dict[str, int]]", res)[0]["a"] == 1

        res = u.Mapper.build(
            data,
            ops=cast(
                "t.Types.ConfigurationDict | None", {"sort": operator.itemgetter("a")}
            ),
        )
        assert cast("list[dict[str, int]]", res)[0]["a"] == 1

    def test_build_unique(self) -> None:
        """Test build unique."""
        data = [1, 2, 1, 3]
        res = u.Mapper.build(data, ops={"unique": True})
        assert res == [1, 2, 3]

    def test_agg_branches(self) -> None:
        """Test agg branches."""
        data = [{"v": 1}, {"v": "str"}]  # "str" ignored
        assert u.Mapper.agg(data, "v") == 1
        assert u.Mapper.agg([], "v") == 0
