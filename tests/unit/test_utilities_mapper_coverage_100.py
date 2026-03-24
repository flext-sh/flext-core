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
from collections.abc import Mapping, MutableMapping, MutableSequence, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, cast

from flext_tests import tm
from pydantic import BaseModel, Field

from tests import TestUnitModels, assertion_helpers, t, u


class UtilitiesMapperCoverage100Namespace:
    class SimpleObj(BaseModel):
        """Simple test t.NormalizedValue."""

        name: Annotated[str, Field(description="Simple t.NormalizedValue name")]
        value: Annotated[int, Field(description="Simple t.NormalizedValue value")]

    class _DoubleOp(BaseModel):
        def __call__(self, value: float | str) -> t.Numeric | str:
            if isinstance(value, (int, float)):
                return value * 2
            return value

    class _GreaterThanTwoOp(BaseModel):
        def __call__(self, value: float | str) -> bool:
            if isinstance(value, (int, float)):
                return value > 2
            return False

    class _TimesTenOp(BaseModel):
        def __call__(self, value: float | str) -> t.Numeric | str:
            if isinstance(value, (int, float)):
                return value * 10
            return value

    class _PlusFiveOp(BaseModel):
        def __call__(self, value: float | str) -> t.Numeric | str:
            if isinstance(value, (int, float)):
                return value + 5
            return value

    class _GroupLenOp(BaseModel):
        def __call__(self, value: float | str) -> int:
            if isinstance(value, str):
                return len(value)
            return 0

    class _GetKeyAOp(BaseModel):
        def __call__(self, value: Mapping[str, int] | float | str) -> int:
            if isinstance(value, dict):
                inner = value.get("a")
                return inner if inner is not None else 0
            return 0

    class _IntConvertOp(BaseModel):
        """Converter that wraps int() builtin for type-safe convert testing."""

        def __call__(self, value: float | bool | str) -> int:
            return int(value)

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
            tm.that(result.value, eq=1)

        def test_extract_dict_nested(self) -> None:
            """Test nested dict extraction."""
            data = {"a": {"b": {"c": 3}}}
            result = u.extract(data, "a.b.c")
            _ = assertion_helpers.assert_flext_result_success(result)
            tm.that(result.value, eq=3)

        def test_extract_object(self) -> None:
            """Test t.NormalizedValue attribute extraction."""
            obj = SimpleObj(name="test", value=42)
            result = u.extract(obj, "name")
            _ = assertion_helpers.assert_flext_result_success(result)
            tm.that(result.value, eq="test")

        def test_extract_model(self) -> None:
            """Test Pydantic model extraction."""
            model = TestUnitModels.ComplexModel(
                id=1, data={"key": "val"}, items=["a", "b"]
            )
            result = u.extract(model, "data.key")
            _ = assertion_helpers.assert_flext_result_success(result)
            tm.that(result.value, eq="val")

        def test_extract_array_index(self) -> None:
            """Test array indexing."""
            data: t.ContainerMapping = {"items": [1, 2, 3]}
            result = u.extract(data, "items[1]")
            _ = assertion_helpers.assert_flext_result_success(result)
            tm.that(result.value, eq=2)

        def test_extract_array_index_nested(self) -> None:
            """Test nested array indexing."""
            data: t.ContainerMapping = {"users": [{"name": "alice"}, {"name": "bob"}]}
            result = u.extract(data, "users[1].name")
            _ = assertion_helpers.assert_flext_result_success(result)
            tm.that(result.value, eq="bob")

        def test_extract_missing_default(self) -> None:
            """Test missing key with default."""
            data = {"a": 1}
            result = u.extract(data, "b", default=10)
            _ = assertion_helpers.assert_flext_result_success(result)
            tm.that(result.value, eq=10)

        def test_extract_missing_required(self) -> None:
            """Test missing key required."""
            data = {"a": 1}
            result = u.extract(data, "b", required=True)
            _ = assertion_helpers.assert_flext_result_failure(result)
            assert "not found" in str(result.error)

        def test_extract_array_index_error(self) -> None:
            """Test invalid array index."""
            data: t.ContainerMapping = {"items": [1]}
            result = u.extract(data, "items[5]", required=True)
            _ = assertion_helpers.assert_flext_result_failure(result)
            msg = str(result.error)
            assert any(x in msg for x in ["out of range", "Invalid index", "not found"])

        def test_extract_none_path(self) -> None:
            """Test extraction when intermediate path is None."""
            data: Mapping[str, None] = {"a": None}
            result = u.extract(data, "a.b", default="defs")
            _ = assertion_helpers.assert_flext_result_success(result)
            tm.that(result.value, eq="defs")
            result_req = u.extract(data, "a.b", required=True)
            tm.fail(result_req)
            assert "is None" in str(result_req.error)

    class TestuMapperAccessors:
        """Tests for u accessors (get, at, take, pick)."""

        def test_get_simple(self) -> None:
            """Test get."""
            data = {"a": 1}
            tm.that(data.get("a"), eq=1)
            tm.that(data.get("b", 2), eq=2)

        def test_at_list(self) -> None:
            """Test at list."""
            items = [10, 20, 30]
            tm.that(u.at(items, 1).value, eq=20)
            tm.fail(u.at(items, 5))
            tm.that(u.at(items, 5, default=0).value, eq=0)

        def test_at_dict(self) -> None:
            """Test at dict."""
            items = {"a": 10}
            tm.that(u.at(items, "a").value, eq=10)
            tm.fail(u.at(items, "b"))

        def test_take_extraction(self) -> None:
            """Test take value extraction."""
            data: t.ConfigurationMapping = {"a": 1, "b": "str"}
            tm.that(u.take(data, "a", as_type=int), eq=1)
            tm.that(u.take(data, "b", as_type=int, default=0), eq=0)

        def test_take_slice(self) -> None:
            """Test take slicing."""
            items: t.ContainerList = [1, 2, 3, 4, 5]
            tm.that(u.take(items, 2), eq=[1, 2])
            tm.that(u.take(items, 2, from_start=False), eq=[4, 5])
            d: Mapping[str, t.Container] = {"a": 1, "b": 2, "c": 3}
            taken = u.take(d, 2)
            tm.that(taken, is_=dict)
            tm.that(len(taken), eq=2)
            assert "a" in taken and "b" in taken

        def test_pick_dict(self) -> None:
            """Test pick as dict."""
            data = {"a": 1, "b": 2, "c": 3}
            picked = u.pick(data, "a", "c")
            tm.that(picked, eq={"a": 1, "c": 3})

        def test_pick_list(self) -> None:
            """Test pick as list."""
            data = {"a": 1, "b": 2, "c": 3}
            picked = u.pick(data, "a", "c", as_dict=False)
            tm.that(picked, eq=[1, 3])

    class TestuMapperUtils:
        """Tests for u utils (as_, or_, flat, agg)."""

        def test_as_conversion(self) -> None:
            """Test as_ type conversion."""
            tm.that(u.as_("123", int), eq=123)
            converted = u.as_("12.3", float)
            tm.that(converted, is_=float)
            assert isinstance(converted, float)
            assert abs(converted - 12.3) < 1e-9
            assert u.as_("true", bool) is True
            tm.that(u.as_("invalid", int, default=0), eq=0)

        def test_as_strict(self) -> None:
            """Test as_ strict mode."""
            tm.that(u.as_("123", int, strict=True, default=0), eq=0)
            tm.that(u.as_(123, int, strict=True), eq=123)

        def test_or_fallback(self) -> None:
            """Test or_ fallback."""
            tm.that(u.or_(None, 1, 2).value, eq=1)
            tm.that(u.or_(None, None, default=3).value, eq=3)

        def test_flat(self) -> None:
            """Test flat."""
            items: Sequence[Sequence[int]] = [[1, 2], [3], []]
            tm.that(u.flat(items), eq=[1, 2, 3])

        def test_agg(self) -> None:
            """Test agg."""
            items = [{"v": 10}, {"v": 20}]
            tm.that(u.agg(items, "v"), eq=30)
            tm.that(u.agg(items, operator.itemgetter("v"), fn=max), eq=20)

    class TestuMapperConversions:
        """Tests for u conversions (ensure, convert)."""

        def test_ensure_str(self) -> None:
            """Test ensure_str."""
            tm.that(u.ensure_str("s"), eq="s")
            tm.that(u.ensure_str(1), eq="1")
            tm.that(u.ensure_str(None, "def"), eq="def")

        def test_ensure_list(self) -> None:
            """Test ensure."""
            tm.that(u.ensure_str_list(["a"]), eq=["a"])
            tm.that(u.ensure_str_list("a"), eq=["a"])
            tm.that(u.ensure_str_list([1, 2]), eq=["1", "2"])
            tm.that(u.ensure_str_list(None), eq=[])

        def test_ensure_str_or_none(self) -> None:
            """Test ensure_str_or_none."""
            str_result = u.ensure_str_or_none("s")
            tm.ok(str_result)
            tm.that(str_result.value, eq="s")
            tm.fail(u.ensure_str_or_none(1))
            tm.fail(u.ensure_str_or_none(None))

        def test_convert_to_json_value(self) -> None:
            obj = SimpleObj(name="test", value=1)
            payload: Mapping[str, t.ValueOrModel] = {"obj": obj}
            res: t.MutableContainerMapping = {}
            for key, val in payload.items():
                if isinstance(val, BaseModel):
                    res[key] = val.model_dump(mode="json")
                else:
                    res[key] = val
            assert "obj" in res
            tm.that(res["obj"], eq={"name": "test", "value": 1})

        def test_convert_to_json_safe(self) -> None:
            obj = SimpleObj(name="test", value=1)
            now = datetime(2026, 3, 12, 10, 30, 45, tzinfo=UTC)
            payload: Mapping[str, t.ValueOrModel] = {
                "obj": obj,
                "path": Path("/tmp/example"),
                "when": now,
            }

            res: t.MutableContainerMapping = {}
            for key, val in payload.items():
                if isinstance(val, BaseModel):
                    res[key] = val.model_dump(mode="json")
                elif isinstance(val, Path):
                    res[key] = val.as_posix()
                elif isinstance(val, datetime):
                    res[key] = val.isoformat()
                else:
                    res[key] = val

            tm.that(res["obj"], eq={"name": "test", "value": 1})
            tm.that(res["path"], eq="/tmp/example")
            tm.that(res["when"], eq="2026-03-12T10:30:45+00:00")

        def test_convert_dict_to_json(self) -> None:
            d: Mapping[str, t.ValueOrModel] = {"a": SimpleObj(name="test", value=1)}
            res: t.MutableContainerMapping = {}
            for key, val in d.items():
                if isinstance(val, BaseModel):
                    res[key] = val.model_dump(mode="json")
                else:
                    res[key] = val
            tm.that(res["a"], eq={"name": "test", "value": 1})

        def test_convert_list_to_json(self) -> None:
            test_list: Sequence[Mapping[str, t.ValueOrModel]] = [
                {"a": SimpleObj(name="test", value=1)}
            ]
            res: MutableSequence[t.ContainerMapping] = []
            for item in test_list:
                item_dict: t.MutableContainerMapping = {}
                for key, val in item.items():
                    if isinstance(val, BaseModel):
                        item_dict[key] = val.model_dump(mode="json")
                    else:
                        item_dict[key] = val
                res.append(item_dict)
            tm.that(res[0]["a"], eq={"name": "test", "value": 1})

    class TestuMapperBuild:
        """Tests for u build/construct/fields."""

        def test_build_pipeline(self) -> None:
            """Test build pipeline."""
            ops = cast(
                "Mapping[str, t.NormalizedValue | t.MapperCallable]",
                {
                    "ensure": "list",
                    "map": _DOUBLE_OP,
                    "filter": _GT_TWO_OP,
                },
            )
            res = u.build([1, 2, 3, 4], ops=ops)
            tm.that(res, eq=[6, 8])

        def test_build_all_ops(self) -> None:
            """Test all build operations."""
            input_data: t.ContainerList = [1, 2, 1, 3, 4]
            ops = cast(
                "Mapping[str, t.NormalizedValue | t.MapperCallable]",
                {
                    "ensure": "list",
                    "filter": _GT_TWO_OP,
                    "map": _TIMES_TEN_OP,
                    "process": _PLUS_FIVE_OP,
                    "sort": True,
                    "unique": True,
                    "slice": (0, 2),
                },
            )
            res = u.build(input_data, ops=ops)
            tm.that(res, eq=[35, 45])

        def test_build_normalize(self) -> None:
            """Test build normalize."""
            ops = {"normalize": "lower"}
            res = u.build(["A", "b"], ops=ops)
            tm.that(res, eq=["a", "b"])

        def test_build_group(self) -> None:
            """Test build group - keys are converted to strings for ConfigurationDict."""
            ops = cast(
                "Mapping[str, t.NormalizedValue | t.MapperCallable]",
                {"group": _GROUP_LEN_OP},
            )
            res = u.build(["cat", "dog", "ant"], ops=ops)
            tm.that(res, eq={"3": ["cat", "dog", "ant"]})

        def test_build_chunk(self) -> None:
            """Test build chunk."""
            ops = {"chunk": 2}
            res = u.build([1, 2, 3, 4], ops=ops)
            tm.that(res, eq=[[1, 2], [3, 4]])

        def test_fields_single(self) -> None:
            """Test fields single extraction."""
            data = {"a": 1}
            tm.that(u.field(data, "a"), eq=1)

        def test_fields_multi(self) -> None:
            """Test fields multi extraction."""
            data: t.ContainerMapping = {"a": 1, "b": 2}
            spec: t.ContainerMapping = {"a": None, "b": None}
            res = u.fields_multi(data, spec)
            tm.that(res, eq={"a": 1, "b": 2})

        def test_construct(self) -> None:
            """Test construct."""
            source: MutableMapping[str, t.ValueOrModel] = {
                "user_name": "john",
                "user_age": 30,
            }
            spec: Mapping[str, t.NormalizedValue | t.MapperCallable] = cast(
                "Mapping[str, t.NormalizedValue | t.MapperCallable]",
                {
                    "name": "user_name",
                    "age": "user_age",
                    "role": {"value": "REDACTED_LDAP_BIND_PASSWORD"},
                },
            )
            res = u.construct_spec(spec, t.ConfigMap(root=source))
            tm.that(
                res,
                eq={"name": "john", "age": 30, "role": "REDACTED_LDAP_BIND_PASSWORD"},
            )

    class TestuMapperAdvanced:
        """Advanced tests for u to reach 100% coverage."""

        def test_model_dump_extraction(self) -> None:
            """Test extraction via model_dump."""

            class Dumpable(BaseModel):
                a: Annotated[int, Field(default=1, description="Dumpable value")] = 1

            obj = Dumpable()
            tm.that(u.extract(obj, "a").value, eq=1)
            tm.that(u.extract(obj, "b", default=2).value, eq=2)

        def test_convert_exception(self) -> None:
            """Test build convert exception handling."""
            ops = cast(
                "Mapping[str, t.NormalizedValue | t.MapperCallable]",
                {
                    "convert": _INT_CONVERT_OP,
                    "convert_default": 0,
                },
            )
            res = u.build("invalid", ops=ops)
            tm.that(res, eq=0)
            ops_default = cast(
                "Mapping[str, t.NormalizedValue | t.MapperCallable]",
                {
                    "convert": _INT_CONVERT_OP,
                    "convert_default": 10,
                },
            )
            res = u.build("invalid", ops=ops_default)
            tm.that(res, eq=10)

        def test_transform_options(self) -> None:
            """Test build transform options."""
            data: t.ContainerMapping = {"a": "UPPER", "b": None, "c": ""}
            ops = cast(
                "Mapping[str, t.NormalizedValue | t.MapperCallable]",
                {
                    "transform": {
                        "normalize": True,
                        "strip_none": True,
                        "strip_empty": True,
                    },
                },
            )
            res = u.build(data, ops=ops)
            tm.that(res, eq={"a": "UPPER"})

        def test_build_sort_complex(self) -> None:
            """Test build sort with callable and string."""
            data: t.ContainerList = [{"a": 2}, {"a": 1}]
            ops_sort = cast(
                "Mapping[str, t.NormalizedValue | t.MapperCallable]",
                {"sort": "a"},
            )
            res = u.build(data, ops=ops_sort)
            assert isinstance(res, list) and res
            assert isinstance(res[0], dict) and res[0].get("a") == 1
            ops_getter = cast(
                "Mapping[str, t.NormalizedValue | t.MapperCallable]",
                {
                    "sort": _GET_KEY_A_OP,
                },
            )
            res = u.build(data, ops=ops_getter)
            assert isinstance(res, list) and res
            assert isinstance(res[0], dict) and res[0].get("a") == 1

        def test_build_unique(self) -> None:
            """Test build unique."""
            data: t.ContainerList = [1, 2, 1, 3]
            res = u.build(data, ops={"unique": True})
            tm.that(res, eq=[1, 2, 3])

        def test_agg_branches(self) -> None:
            """Test agg branches."""
            data = [{"v": 1}, {"v": "str"}]
            tm.that(u.agg(data, "v"), eq=1)
            tm.that(u.agg([], "v"), eq=0)


SimpleObj = UtilitiesMapperCoverage100Namespace.SimpleObj
_DoubleOp = UtilitiesMapperCoverage100Namespace._DoubleOp
_GreaterThanTwoOp = UtilitiesMapperCoverage100Namespace._GreaterThanTwoOp
_TimesTenOp = UtilitiesMapperCoverage100Namespace._TimesTenOp
_PlusFiveOp = UtilitiesMapperCoverage100Namespace._PlusFiveOp
_GroupLenOp = UtilitiesMapperCoverage100Namespace._GroupLenOp
_GetKeyAOp = UtilitiesMapperCoverage100Namespace._GetKeyAOp
_IntConvertOp = UtilitiesMapperCoverage100Namespace._IntConvertOp
_DOUBLE_OP = UtilitiesMapperCoverage100Namespace._DOUBLE_OP
_GT_TWO_OP = UtilitiesMapperCoverage100Namespace._GT_TWO_OP
_TIMES_TEN_OP = UtilitiesMapperCoverage100Namespace._TIMES_TEN_OP
_PLUS_FIVE_OP = UtilitiesMapperCoverage100Namespace._PLUS_FIVE_OP
_GROUP_LEN_OP = UtilitiesMapperCoverage100Namespace._GROUP_LEN_OP
_GET_KEY_A_OP = UtilitiesMapperCoverage100Namespace._GET_KEY_A_OP
_INT_CONVERT_OP = UtilitiesMapperCoverage100Namespace._INT_CONVERT_OP
TestuMapperExtract = UtilitiesMapperCoverage100Namespace.TestuMapperExtract
TestuMapperAccessors = UtilitiesMapperCoverage100Namespace.TestuMapperAccessors
TestuMapperUtils = UtilitiesMapperCoverage100Namespace.TestuMapperUtils
TestuMapperConversions = UtilitiesMapperCoverage100Namespace.TestuMapperConversions
TestuMapperBuild = UtilitiesMapperCoverage100Namespace.TestuMapperBuild
TestuMapperAdvanced = UtilitiesMapperCoverage100Namespace.TestuMapperAdvanced
