"""Utilities mapper full coverage tests."""

from __future__ import annotations

from collections import UserList
from collections.abc import (
    Iterator,
    Mapping,
    Sequence,
)
from typing import Annotated, ClassVar, Never, override

import pytest

from flext_tests import tm
from tests import m, p, t, u

_PortModel = m.Core.Tests.PortModel
_MaybeModel = m.Core.Tests.MaybeModel
_GroupModel = m.Core.Tests.GroupModel
_BadItems = m.Core.Tests.BadItems
_BadIter = m.Core.Tests.BadIter

_ExtractFieldCallable = p.Core.Tests.ExtractFieldCallable
_TakeCallable = p.Core.Tests.TakeCallable
_BuildApplyConvertCallable = p.Core.Tests.BuildApplyConvertCallable
_ExtractTransformOptionsCallable = p.Core.Tests.ExtractTransformOptionsCallable
_BuildApplyOpCallable = p.Core.Tests.BuildApplyOpCallable
_TransformCallable = p.Core.Tests.TransformCallable
_MapDictKeysCallable = p.Core.Tests.MapDictKeysCallable


class AttrObject(m.BaseModel):
    """AttrObject class."""

    name: Annotated[str, m.Field(description="Attribute recursive container name")] = (
        "name"
    )
    value: Annotated[
        int, m.Field(description="Attribute recursive container value")
    ] = 1


class BadString:
    """BadString class."""

    @override
    def __str__(self) -> str:
        """__str__ method."""
        msg = "cannot stringify"
        raise ValueError(msg)


class BadBool:
    """BadBool class."""

    def __bool__(self) -> bool:
        """__bool__ method."""
        msg = "cannot bool"
        raise ValueError(msg)


class ExplodingLenList(UserList[int]):
    """ExplodingLenList class."""

    @override
    def __len__(self) -> int:
        """__len__ method."""
        msg = "len exploded"
        raise TypeError(msg)


class BadMapping(t.ContainerMappingBase):
    @override
    def __getitem__(self, _key: str) -> t.RecursiveContainer:
        msg = "get exploded"
        raise TypeError(msg)

    @override
    def __iter__(self) -> Iterator[str]:
        msg = "iter exploded"
        raise TypeError(msg)

    @override
    def __len__(self) -> int:
        return 1


@pytest.fixture
def mapper() -> type[u]:
    return u


class TestUtilitiesMapperFullCoverage:
    _PYRIGHT_USED_HELPERS: ClassVar[tuple[object, ...]] = ()

    @staticmethod
    def _extract_field_obj(item: AttrObject, field_name: str) -> t.RecursiveContainer:
        """Call _extract_field_value with arbitrary recursive container inputs."""
        fn: _ExtractFieldCallable = getattr(u, "_extract_field_value")
        return fn(item, field_name)

    @staticmethod
    def _take_obj(
        data_or_items: _MaybeModel | _PortModel | int,
        key_or_index: int | str,
        *,
        default: str | None = None,
    ) -> t.RecursiveContainerMapping | t.RecursiveContainerList | t.RecursiveContainer:
        fn: _TakeCallable = getattr(u, "take")
        return fn(data_or_items, key_or_index, default=default)

    @staticmethod
    def _build_apply_convert_obj(
        current: tuple[str, ...] | str | int,
        operations: Mapping[str, t.MapperInput],
    ) -> t.RecursiveContainer:
        fn: _BuildApplyConvertCallable = getattr(u, "_op_convert")
        return fn(current, operations, None, "stop")

    @staticmethod
    def _extract_transform_options_obj(
        transform_opts: Mapping[str, t.MapperInput],
    ) -> tuple[
        bool,
        bool,
        bool,
        t.StrMapping | None,
        set[str] | None,
        set[str] | None,
    ]:
        fn: _ExtractTransformOptionsCallable = getattr(u, "_extract_transform_options")
        return fn(transform_opts)

    @staticmethod
    def _build_apply_sort_obj(
        current: tuple[str, str],
        operations: Mapping[str, t.MapperInput],
    ) -> t.RecursiveContainerMapping | t.RecursiveContainerList | t.RecursiveContainer:
        fn: _BuildApplyOpCallable = getattr(u, "_op_sort")
        return fn(current, operations, None, "stop")

    @staticmethod
    def _build_apply_unique_obj(
        current: tuple[int, int, int],
        operations: Mapping[str, t.MapperInput],
    ) -> t.RecursiveContainerMapping | t.RecursiveContainerList | t.RecursiveContainer:
        fn: _BuildApplyOpCallable = getattr(u, "_op_unique")
        return fn(current, operations, None, "stop")

    @staticmethod
    def _build_apply_slice_obj(
        current: tuple[int, int, int],
        operations: Mapping[str, t.MapperInput],
    ) -> t.RecursiveContainerMapping | t.RecursiveContainerList | t.RecursiveContainer:
        fn: _BuildApplyOpCallable = getattr(u, "_op_slice")
        return fn(current, operations, None, "stop")

    @staticmethod
    def _build_apply_group_obj(
        current: Sequence[_GroupModel],
        operations: Mapping[str, t.MapperInput],
    ) -> t.RecursiveContainerMapping | t.RecursiveContainerList | t.RecursiveContainer:
        fn: _BuildApplyOpCallable = getattr(u, "_op_group")
        return fn(current, operations, None, "stop")

    @staticmethod
    def _transform_obj(
        source: BadMapping,
        **kwargs: t.StrMapping,
    ) -> p.Result[t.RecursiveContainerMapping]:
        fn: _TransformCallable = getattr(u, "transform")
        return fn(source, **kwargs)

    @staticmethod
    def _map_dict_keys_obj(
        source: _BadItems,
        key_map: t.StrMapping,
        *,
        keep_unmapped: bool = True,
    ) -> p.Result[t.RecursiveContainerMapping]:
        fn: _MapDictKeysCallable = getattr(u, "map_dict_keys")
        return fn(source, key_map, keep_unmapped=keep_unmapped)

    @staticmethod
    def _parse_int(value: t.ValueOrModel) -> int:
        assert isinstance(value, str)
        return int(value)

    @staticmethod
    def _plus_one(value: int) -> int:
        return value + 1

    @staticmethod
    def _times_two(value: int) -> int:
        return value * 2

    @staticmethod
    def _raise_value_error(_value: t.Scalar) -> Never:
        msg = "x"
        raise ValueError(msg)

    @staticmethod
    def _normalize_not_dict(_value: t.RecursiveContainer) -> str:
        return "not-a-dict"

    @staticmethod
    def _negative(value: int) -> bool:
        return value < 0

    def test_bad_string_and_bad_bool_raise_value_error(self) -> None:
        with pytest.raises(ValueError, match="cannot stringify"):
            _ = str(BadString())
        with pytest.raises(ValueError, match="cannot bool"):
            _ = bool(BadBool())

    def test_extract_array_index_helpers(self, mapper: type[u]) -> None:
        idx_result = mapper._extract_handle_array_index("x", "0")
        tm.fail(idx_result)
        tm.that(str(idx_result.error), has="Not a sequence")
        idx_neg = mapper._extract_handle_array_index([1, 2], "-1")
        tm.ok(idx_neg)
        tm.that(idx_neg.value, eq=2)
        idx_bad = mapper._extract_handle_array_index([1, 2], "bad")
        tm.fail(idx_bad)
        tm.that(str(idx_bad.error), has="Invalid index")

    def test_extract_error_paths_and_prop_accessor(self, mapper: type[u]) -> None:
        res_none_intermediate = mapper.extract({"a": None}, "a.b")
        tm.fail(res_none_intermediate)
        tm.that(str(res_none_intermediate.error), has="default is None")
        res_missing_key = mapper.extract({"a": 1}, "b")
        tm.fail(res_missing_key)
        tm.that(str(res_missing_key.error), has="default is None")
        res_bad_index = mapper.extract({"a": [1]}, "a[bad]")
        tm.fail(res_bad_index)
        tm.that(str(res_bad_index.error), has="Array error")
        res_terminal_none = mapper.extract({"a": None}, "a")
        tm.ok(res_terminal_none)
        tm.that(res_terminal_none.value, eq="")

        class NotGeneral:
            @override
            def __str__(self) -> str:
                return "converted"

        class Container:
            field: NotGeneral = NotGeneral()

        extract_fn = getattr(mapper, "extract")
        res_non_general = extract_fn(
            Container(),
            "field",
        )
        tm.fail(res_non_general)
        tm.that(str(res_non_general.error), has="default is None")

        class ExplodingModelDump:
            def __init__(self) -> None:
                self.model_dump = lambda: (_ for _ in ()).throw(ValueError("boom"))

        res_exception = extract_fn(
            ExplodingModelDump(),
            "a",
        )
        tm.fail(res_exception)
        tm.that(str(res_exception.error).lower(), has="extract failed")
        accessor = mapper.prop("name")
        tm.that(
            accessor(
                AttrObject(name="x", value=1),
            ),
            eq="x",
        )
        tm.that(
            mapper.prop("missing")(
                {"a": 1},
            ),
            eq="",
        )


TestUtilitiesMapperFullCoverage._PYRIGHT_USED_HELPERS = (
    _BadIter,
    TestUtilitiesMapperFullCoverage._extract_field_obj,
    TestUtilitiesMapperFullCoverage._take_obj,
    TestUtilitiesMapperFullCoverage._build_apply_convert_obj,
    TestUtilitiesMapperFullCoverage._extract_transform_options_obj,
    TestUtilitiesMapperFullCoverage._build_apply_sort_obj,
    TestUtilitiesMapperFullCoverage._build_apply_unique_obj,
    TestUtilitiesMapperFullCoverage._build_apply_slice_obj,
    TestUtilitiesMapperFullCoverage._build_apply_group_obj,
    TestUtilitiesMapperFullCoverage._transform_obj,
    TestUtilitiesMapperFullCoverage._map_dict_keys_obj,
    TestUtilitiesMapperFullCoverage._parse_int,
    TestUtilitiesMapperFullCoverage._plus_one,
    TestUtilitiesMapperFullCoverage._times_two,
    TestUtilitiesMapperFullCoverage._raise_value_error,
    TestUtilitiesMapperFullCoverage._normalize_not_dict,
    TestUtilitiesMapperFullCoverage._negative,
)
