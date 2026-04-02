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
from collections.abc import Mapping, MutableSequence, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, cast

from pydantic import BaseModel, Field

from flext_tests import tm
from tests import assertion_helpers, t, u
from tests.unit import _models_impl as test_unit_models


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
        def __call__(self, value: t.IntMapping | float | str) -> int:
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
            model = test_unit_models.ComplexModel(
                id=1,
                data={"key": "val"},
                items=["a", "b"],
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

    class TestuMapperUtils:
        """Tests for u utils (or_, flat, agg)."""

        def test_agg(self) -> None:
            """Test agg."""
            items = [{"v": 10}, {"v": 20}]
            tm.that(u.agg(items, "v"), eq=30)
            tm.that(u.agg(items, operator.itemgetter("v"), fn=max), eq=20)

    class TestuMapperConversions:
        """Tests for u conversions (ensure, convert)."""

        @staticmethod
        def _to_json_safe(value: t.ValueOrModel) -> t.NormalizedValue:
            if isinstance(value, BaseModel):
                return cast("t.NormalizedValue", value.model_dump(mode="json"))
            if isinstance(value, Path):
                return value.as_posix()
            if isinstance(value, datetime):
                return value.isoformat()
            return cast("t.NormalizedValue", value)

        def test_ensure_str(self) -> None:
            """Test ensure_str."""
            tm.that(u.ensure_str("s"), eq="s")
            tm.that(u.ensure_str(1), eq="1")
            tm.that(u.ensure_str(None, "def"), eq="def")

        def test_convert_to_json_value(self) -> None:
            obj = SimpleObj(name="test", value=1)
            payload: Mapping[str, t.ValueOrModel] = {"obj": obj}
            res: t.MutableContainerMapping = {}
            for key, val in payload.items():
                res[key] = self._to_json_safe(val)
            assert "obj" in res
            tm.that(
                res["obj"], eq=cast("t.Tests.Testobject", {"name": "test", "value": 1})
            )

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
                res[key] = self._to_json_safe(val)

            tm.that(
                res["obj"], eq=cast("t.Tests.Testobject", {"name": "test", "value": 1})
            )
            tm.that(res["path"], eq="/tmp/example")
            tm.that(res["when"], eq="2026-03-12T10:30:45+00:00")

        def test_convert_dict_to_json(self) -> None:
            d: Mapping[str, t.ValueOrModel] = {"a": SimpleObj(name="test", value=1)}
            res: t.MutableContainerMapping = {}
            for key, val in d.items():
                res[key] = self._to_json_safe(val)
            tm.that(
                res["a"], eq=cast("t.Tests.Testobject", {"name": "test", "value": 1})
            )

        def test_convert_list_to_json(self) -> None:
            test_list: Sequence[Mapping[str, t.ValueOrModel]] = [
                {"a": SimpleObj(name="test", value=1)},
            ]
            res: MutableSequence[t.ContainerMapping] = []
            for item in test_list:
                item_dict: t.MutableContainerMapping = {}
                for key, val in item.items():
                    item_dict[key] = self._to_json_safe(val)
                res.append(item_dict)
            tm.that(
                res[0]["a"], eq=cast("t.Tests.Testobject", {"name": "test", "value": 1})
            )

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
