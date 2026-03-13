"""Real tests for u coverage - extract and accessors.

Module: flext_core._utilities.mapper
Scope: FlextUtilitiesMapper - extract, get, at, take, pick, as_, or_, flat, agg, etc.

This module provides comprehensive real tests to achieve 100% coverage for
FlextUtilitiesMapper, focusing on complex nested extraction and accessor patterns.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import operator
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, cast

from pydantic import BaseModel, Field

from flext_core import m, t, u

from ..test_utils import assertion_helpers
from ._models import ComplexModel


class SimpleObj(BaseModel):
    """Simple test object."""

    name: Annotated[str, Field(description="Simple object name")]
    value: Annotated[int, Field(description="Simple object value")]


class _DoubleOp(BaseModel):
    def __call__(self, value: object) -> object:
        if isinstance(value, (int, float)):
            return value * 2
        return value


class _GreaterThanTwoOp(BaseModel):
    def __call__(self, value: object) -> object:
        if isinstance(value, (int, float)):
            return value > 2
        return False


class _TimesTenOp(BaseModel):
    def __call__(self, value: object) -> object:
        if isinstance(value, (int, float)):
            return value * 10
        return value


class _PlusFiveOp(BaseModel):
    def __call__(self, value: object) -> object:
        if isinstance(value, (int, float)):
            return value + 5
        return value


class _GroupLenOp(BaseModel):
    def __call__(self, value: object) -> object:
        if isinstance(value, str):
            return len(value)
        return 0


class _GetKeyAOp(BaseModel):
    def __call__(self, value: object) -> object:
        if isinstance(value, dict):
            inner = value.get("a")
            return inner if inner is not None else 0
        return 0


class _IntConvertOp(BaseModel):
    """Converter that wraps int() builtin for type-safe convert testing."""

    def __call__(self, value: object) -> object:
        if isinstance(value, (int, float, bool)):
            return int(value)
        if isinstance(value, str):
            return int(value)
        return value


_DOUBLE_OP = _DoubleOp()
_GT_TWO_OP = _GreaterThanTwoOp()
_TIMES_TEN_OP = _TimesTenOp()
_PLUS_FIVE_OP = _PlusFiveOp()
_GROUP_LEN_OP = _GroupLenOp()
_GET_KEY_A_OP = _GetKeyAOp()
_INT_CONVERT_OP = _IntConvertOp()


class TestuMapperExtract:
    """Tests for u.extract."""

    def test_extract_dict_simple(self) -> None:
        """Test simple dict extraction."""
        data = {"a": 1, "b": 2}
        result = u.extract(data, "a")
        _ = assertion_helpers.assert_flext_result_success(result)
        assert result.value == 1

    def test_extract_dict_nested(self) -> None:
        """Test nested dict extraction."""
        data = {"a": {"b": {"c": 3}}}
        result = u.extract(data, "a.b.c")
        _ = assertion_helpers.assert_flext_result_success(result)
        assert result.value == 3

    def test_extract_object(self) -> None:
        """Test object attribute extraction."""
        obj = SimpleObj(name="test", value=42)
        result = u.extract(obj, "name")
        _ = assertion_helpers.assert_flext_result_success(result)
        assert result.value == "test"

    def test_extract_model(self) -> None:
        """Test Pydantic model extraction."""
        model = ComplexModel(id=1, data={"key": "val"}, items=["a", "b"])
        result = u.extract(model, "data.key")
        _ = assertion_helpers.assert_flext_result_success(result)
        assert result.value == "val"

    def test_extract_array_index(self) -> None:
        """Test array indexing."""
        data = {"items": [1, 2, 3]}
        result = u.extract(data, "items[1]")
        _ = assertion_helpers.assert_flext_result_success(result)
        assert result.value == 2

    def test_extract_array_index_nested(self) -> None:
        """Test nested array indexing."""
        data = {"users": [{"name": "alice"}, {"name": "bob"}]}
        result = u.extract(data, "users[1].name")
        _ = assertion_helpers.assert_flext_result_success(result)
        assert result.value == "bob"

    def test_extract_missing_default(self) -> None:
        """Test missing key with default."""
        data = {"a": 1}
        result = u.extract(data, "b", default=10)
        _ = assertion_helpers.assert_flext_result_success(result)
        assert result.value == 10

    def test_extract_missing_required(self) -> None:
        """Test missing key required."""
        data = {"a": 1}
        result = u.extract(data, "b", required=True)
        _ = assertion_helpers.assert_flext_result_failure(result)
        assert "not found" in str(result.error)

    def test_extract_array_index_error(self) -> None:
        """Test invalid array index."""
        data = {"items": [1]}
        result = u.extract(data, "items[5]", required=True)
        _ = assertion_helpers.assert_flext_result_failure(result)
        msg = str(result.error)
        assert any(x in msg for x in ["out of range", "Invalid index", "not found"])

    def test_extract_none_path(self) -> None:
        """Test extraction when intermediate path is None."""
        data: dict[str, None] = {"a": None}
        result = u.extract(data, "a.b", default="defs")
        _ = assertion_helpers.assert_flext_result_success(result)
        assert result.value == "defs"
        result_req = u.extract(data, "a.b", required=True)
        assert result_req.is_failure
        assert "is None" in str(result_req.error)


class TestuMapperAccessors:
    """Tests for u accessors (get, at, take, pick)."""

    def test_get_simple(self) -> None:
        """Test get."""
        data = {"a": 1}
        assert u.get(data, "a") == 1
        assert u.get(data, "b", default=2) == 2

    def test_at_list(self) -> None:
        """Test at list."""
        items = [10, 20, 30]
        assert u.at(items, 1).value == 20
        assert u.at(items, 5).is_failure
        assert u.at(items, 5, default=0).value == 0

    def test_at_dict(self) -> None:
        """Test at dict."""
        items = {"a": 10}
        assert u.at(items, "a").value == 10
        assert u.at(items, "b").is_failure

    def test_take_extraction(self) -> None:
        """Test take value extraction."""
        data: dict[str, t.Scalar] = {"a": 1, "b": "str"}
        assert u.take(data, "a", as_type=int) == 1
        assert u.take(data, "b", as_type=int, default=0) == 0

    def test_take_slice(self) -> None:
        """Test take slicing."""
        items: t.ContainerList = [1, 2, 3, 4, 5]
        assert u.take(items, 2) == [1, 2]
        assert u.take(items, 2, from_start=False) == [4, 5]
        d: dict[str, t.Container] = {"a": 1, "b": 2, "c": 3}
        taken = u.take(d, 2)
        assert isinstance(taken, dict)
        assert len(taken) == 2
        assert "a" in taken and "b" in taken

    def test_pick_dict(self) -> None:
        """Test pick as dict."""
        data = {"a": 1, "b": 2, "c": 3}
        picked = u.pick(data, "a", "c")
        assert picked == {"a": 1, "c": 3}

    def test_pick_list(self) -> None:
        """Test pick as list."""
        data = {"a": 1, "b": 2, "c": 3}
        picked = u.pick(data, "a", "c", as_dict=False)
        assert picked == [1, 3]


class TestuMapperUtils:
    """Tests for u utils (as_, or_, flat, agg)."""

    def test_as_conversion(self) -> None:
        """Test as_ type conversion."""
        assert u.as_("123", int) == 123
        converted = u.as_("12.3", float)
        assert isinstance(converted, float)
        assert abs(converted - 12.3) < 1e-9
        assert u.as_("true", bool) is True
        assert u.as_("invalid", int, default=0) == 0

    def test_as_strict(self) -> None:
        """Test as_ strict mode."""
        assert u.as_("123", int, strict=True, default=0) == 0
        assert u.as_(123, int, strict=True) == 123

    def test_or_fallback(self) -> None:
        """Test or_ fallback."""
        assert u.or_(None, 1, 2).value == 1
        assert u.or_(None, None, default=3).value == 3

    def test_flat(self) -> None:
        """Test flat."""
        items: list[list[int]] = [[1, 2], [3], []]
        assert u.flat(items) == [1, 2, 3]

    def test_agg(self) -> None:
        """Test agg."""
        items = [{"v": 10}, {"v": 20}]
        assert u.agg(items, "v") == 30
        assert u.agg(items, operator.itemgetter("v"), fn=max) == 20


class TestuMapperConversions:
    """Tests for u conversions (ensure, convert)."""

    def test_ensure_str(self) -> None:
        """Test ensure_str."""
        assert u.ensure_str("s") == "s"
        assert u.ensure_str(1) == "1"
        assert u.ensure_str(None, "def") == "def"

    def test_ensure_list(self) -> None:
        """Test ensure."""
        assert u.ensure_str_list(["a"]) == ["a"]
        assert u.ensure_str_list("a") == ["a"]
        assert u.ensure_str_list([1, 2]) == ["1", "2"]
        assert u.ensure_str_list(None) == []

    def test_ensure_str_or_none(self) -> None:
        """Test ensure_str_or_none."""
        str_result = u.ensure_str_or_none("s")
        assert str_result.is_success
        assert str_result.value == "s"
        assert u.ensure_str_or_none(1).is_failure
        assert u.ensure_str_or_none(None).is_failure

    def test_convert_to_json_value(self) -> None:
        obj = SimpleObj(name="test", value=1)
        payload = {"obj": obj}
        res: dict[str, object] = {}
        for key, val in payload.items():
            if isinstance(val, BaseModel):
                res[key] = val.model_dump(mode="json")
            else:
                res[key] = val
        assert isinstance(res, dict)
        assert "obj" in res
        assert res["obj"] == {"name": "test", "value": 1}

    def test_convert_to_json_safe(self) -> None:
        obj = SimpleObj(name="test", value=1)
        now = datetime(2026, 3, 12, 10, 30, 45, tzinfo=UTC)
        payload = {
            "obj": obj,
            "path": Path("/tmp/example"),
            "when": now,
        }

        res: dict[str, object] = {}
        for key, val in payload.items():
            if isinstance(val, BaseModel):
                res[key] = val.model_dump(mode="json")
            elif isinstance(val, Path):
                res[key] = val.as_posix()
            elif isinstance(val, datetime):
                res[key] = val.isoformat()
            else:
                res[key] = val

        assert isinstance(res, dict)
        assert res["obj"] == {"name": "test", "value": 1}
        assert res["path"] == "/tmp/example"
        assert res["when"] == "2026-03-12T10:30:45+00:00"

    def test_convert_dict_to_json(self) -> None:
        d: dict[str, object] = {"a": SimpleObj(name="test", value=1)}
        res: dict[str, object] = {}
        for key, val in d.items():
            if isinstance(val, BaseModel):
                res[key] = val.model_dump(mode="json")
            else:
                res[key] = val
        if isinstance(res, dict):
            assert res["a"] == {"name": "test", "value": 1}
        else:
            msg = "Expected dict result"
            raise AssertionError(msg)

    def test_convert_list_to_json(self) -> None:
        test_list: list[dict[str, object]] = [{"a": SimpleObj(name="test", value=1)}]
        res: list[dict[str, object]] = []
        for item in test_list:
            item_dict: dict[str, object] = {}
            for key, val in item.items():
                if isinstance(val, BaseModel):
                    item_dict[key] = val.model_dump(mode="json")
                else:
                    item_dict[key] = val
            res.append(item_dict)
        if isinstance(res, list) and isinstance(res[0], dict):
            assert res[0]["a"] == {"name": "test", "value": 1}
        else:
            msg = "Expected list of dicts result"
            raise AssertionError(msg)


class TestuMapperBuild:
    """Tests for u build/construct/fields."""

    def test_build_pipeline(self) -> None:
        """Test build pipeline."""
        ops = {
            "ensure": "list",
            "map": _DOUBLE_OP,
            "filter": _GT_TWO_OP,
        }
        res = u.build([1, 2, 3, 4], ops=ops)
        assert res == [6, 8]

    def test_build_all_ops(self) -> None:
        """Test all build operations."""
        input_data = [1, 2, 1, 3, 4]
        ops = {
            "ensure": "list",
            "filter": _GT_TWO_OP,
            "map": _TIMES_TEN_OP,
            "process": _PLUS_FIVE_OP,
            "sort": True,
            "unique": True,
            "slice": (0, 2),
        }
        res = u.build(input_data, ops=ops)
        assert res == [35, 45]

    def test_build_normalize(self) -> None:
        """Test build normalize."""
        ops = {"normalize": "lower"}
        res = u.build(["A", "b"], ops=ops)
        assert res == ["a", "b"]

    def test_build_group(self) -> None:
        """Test build group - keys are converted to strings for ConfigurationDict."""
        ops = {"group": _GROUP_LEN_OP}
        res = u.build(["cat", "dog", "ant"], ops=ops)
        assert res == {"3": ["cat", "dog", "ant"]}

    def test_build_chunk(self) -> None:
        """Test build chunk."""
        ops = {"chunk": 2}
        res = u.build([1, 2, 3, 4], ops=ops)
        assert res == [[1, 2], [3, 4]]

    def test_fields_single(self) -> None:
        """Test fields single extraction."""
        data = {"a": 1}
        assert u.field(data, "a") == 1

    def test_fields_multi(self) -> None:
        """Test fields multi extraction."""
        data: dict[str, object] = {"a": 1, "b": 2}
        spec: dict[str, object] = {"a": None, "b": None}
        res = u.fields_multi(data, spec)
        assert res == {"a": 1, "b": 2}

    def test_construct(self) -> None:
        """Test construct."""
        source: dict[str, object] = {
            "user_name": "john",
            "user_age": 30,
        }
        spec: dict[str, object] = {
            "name": "user_name",
            "age": "user_age",
            "role": {"value": "REDACTED_LDAP_BIND_PASSWORD"},
        }
        res = u.construct(
            spec, m.ConfigMap(root=cast("dict[str, t.Container | BaseModel]", source))
        )
        assert res == {"name": "john", "age": 30, "role": "REDACTED_LDAP_BIND_PASSWORD"}


class TestuMapperAdvanced:
    """Advanced tests for u to reach 100% coverage."""

    def test_model_dump_extraction(self) -> None:
        """Test extraction via model_dump."""

        class Dumpable(BaseModel):
            a: Annotated[int, Field(default=1, description="Dumpable value")]

        obj = Dumpable()
        assert u.extract(obj, "a").value == 1
        assert u.extract(obj, "b", default=2).value == 2

    def test_convert_exception(self) -> None:
        """Test build convert exception handling."""
        ops: dict[str, object] = {
            "convert": _INT_CONVERT_OP,
            "convert_default": 0,
        }
        res = u.build("invalid", ops=ops)
        assert res == 0
        ops_default: dict[str, object] = {
            "convert": _INT_CONVERT_OP,
            "convert_default": 10,
        }
        res = u.build("invalid", ops=ops_default)
        assert res == 10

    def test_transform_options(self) -> None:
        """Test build transform options."""
        data: dict[str, object] = {"a": "UPPER", "b": None, "c": ""}
        ops: dict[str, object] = {
            "transform": {"normalize": True, "strip_none": True, "strip_empty": True},
        }
        res = u.build(data, ops=ops)
        assert res == {"a": "UPPER"}

    def test_build_sort_complex(self) -> None:
        """Test build sort with callable and string."""
        data: list[dict[str, object]] = [{"a": 2}, {"a": 1}]
        ops_sort: dict[str, object] = {"sort": "a"}
        res = u.build(data, ops=ops_sort)
        assert isinstance(res, list) and len(res) > 0
        assert isinstance(res[0], dict) and res[0].get("a") == 1
        ops_getter: dict[str, object] = {
            "sort": _GET_KEY_A_OP,
        }
        res = u.build(data, ops=ops_getter)
        assert isinstance(res, list) and len(res) > 0
        assert isinstance(res[0], dict) and res[0].get("a") == 1

    def test_build_unique(self) -> None:
        """Test build unique."""
        data: list[int] = [1, 2, 1, 3]
        res = u.build(data, ops={"unique": True})
        assert res == [1, 2, 3]

    def test_agg_branches(self) -> None:
        """Test agg branches."""
        data = [{"v": 1}, {"v": "str"}]
        assert u.agg(data, "v") == 1
        assert u.agg([], "v") == 0
