"""Utilities mapper full coverage tests."""

from __future__ import annotations

from collections import UserDict, UserList
from collections.abc import (
    ItemsView,
    Iterator,
    Mapping,
    Sequence,
)
from typing import Annotated, Never, Protocol, cast, override

import pytest
from pydantic import BaseModel, Field

from flext_core import r
from flext_tests import tm
from tests import p, t, u


class _PortModel(BaseModel):
    """Model with port/nested for mapper take/extract tests."""

    port: int = 0
    nested: Annotated[
        t.ContainerMapping,
        Field(default_factory=dict),
    ]


class _MaybeModel(BaseModel):
    """Model with optional field for take tests."""

    x: str | None = None


class _GroupModel(BaseModel):
    """Model with optional kind for group tests."""

    kind: str | None = None


class _BadItems(UserDict[str, t.RecursiveContainer]):
    """UserDict that explodes on items() for error-path testing."""

    @override
    def items(self) -> ItemsView[str, t.RecursiveContainer]:
        """Items method."""
        msg = "bad items"
        raise RuntimeError(msg)


class _BadIter(UserList[str]):
    """UserList that explodes on __iter__ for error-path testing."""

    @override
    def __iter__(self) -> Iterator[str]:
        """__iter__ method."""
        msg = "bad iter"
        raise RuntimeError(msg)


class _ExtractFieldCallable(Protocol):
    def __call__(self, item: AttrObject, field_name: str) -> t.RecursiveContainer: ...


class _TakeCallable(Protocol):
    def __call__(
        self,
        data_or_items: _MaybeModel | _PortModel | int,
        key_or_index: int | str,
        *,
        default: str | None = None,
    ) -> t.ContainerMapping | t.ContainerList | t.RecursiveContainer: ...


class _BuildApplyConvertCallable(Protocol):
    def __call__(
        self,
        current: tuple[str, ...] | str | int,
        operations: Mapping[str, t.MapperInput],
        default_val: t.RecursiveContainer,
        on_error: str,
    ) -> t.RecursiveContainer: ...


class _ExtractTransformOptionsCallable(Protocol):
    def __call__(
        self,
        transform_opts: Mapping[str, t.MapperInput],
    ) -> tuple[
        bool,
        bool,
        bool,
        t.StrMapping | None,
        set[str] | None,
        set[str] | None,
    ]: ...


class _BuildApplyOpCallable(Protocol):
    def __call__(
        self,
        current: tuple[str, str] | tuple[int, int, int] | Sequence[_GroupModel],
        operations: Mapping[str, t.MapperInput],
        default_val: t.RecursiveContainer,
        on_error: str,
    ) -> t.ContainerMapping | t.ContainerList | t.RecursiveContainer: ...


class _TransformCallable(Protocol):
    def __call__(
        self,
        source: BadMapping,
        **kwargs: t.StrMapping,
    ) -> r[t.ContainerMapping]: ...


class _MapDictKeysCallable(Protocol):
    def __call__(
        self,
        source: _BadItems,
        key_map: t.StrMapping,
        *,
        keep_unmapped: bool = True,
    ) -> r[t.ContainerMapping]: ...


def _extract_field_obj(item: AttrObject, field_name: str) -> t.RecursiveContainer:
    """Call _extract_field_value with arbitrary recursive container inputs."""
    fn: _ExtractFieldCallable = getattr(u, "_extract_field_value")
    return fn(item, field_name)


def _take_obj(
    data_or_items: _MaybeModel | _PortModel | int,
    key_or_index: int | str,
    *,
    default: str | None = None,
) -> t.ContainerMapping | t.ContainerList | t.RecursiveContainer:
    fn: _TakeCallable = getattr(u, "take")
    return fn(data_or_items, key_or_index, default=default)


def _build_apply_convert_obj(
    current: tuple[str, ...] | str | int,
    operations: Mapping[str, t.MapperInput],
) -> t.RecursiveContainer:
    fn: _BuildApplyConvertCallable = getattr(u, "_op_convert")
    return fn(current, operations, None, "stop")


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


def _build_apply_sort_obj(
    current: tuple[str, str],
    operations: Mapping[str, t.MapperInput],
) -> t.ContainerMapping | t.ContainerList | t.RecursiveContainer:
    fn: _BuildApplyOpCallable = getattr(u, "_op_sort")
    return fn(current, operations, None, "stop")


def _build_apply_unique_obj(
    current: tuple[int, int, int],
    operations: Mapping[str, t.MapperInput],
) -> t.ContainerMapping | t.ContainerList | t.RecursiveContainer:
    fn: _BuildApplyOpCallable = getattr(u, "_op_unique")
    return fn(current, operations, None, "stop")


def _build_apply_slice_obj(
    current: tuple[int, int, int],
    operations: Mapping[str, t.MapperInput],
) -> t.ContainerMapping | t.ContainerList | t.RecursiveContainer:
    fn: _BuildApplyOpCallable = getattr(u, "_op_slice")
    return fn(current, operations, None, "stop")


def _build_apply_group_obj(
    current: Sequence[_GroupModel],
    operations: Mapping[str, t.MapperInput],
) -> t.ContainerMapping | t.ContainerList | t.RecursiveContainer:
    fn: _BuildApplyOpCallable = getattr(u, "_op_group")
    return fn(current, operations, None, "stop")


def _transform_obj(
    source: BadMapping,
    **kwargs: t.StrMapping,
) -> r[t.ContainerMapping]:
    fn: _TransformCallable = getattr(u, "transform")
    return fn(source, **kwargs)


def _map_dict_keys_obj(
    source: _BadItems,
    key_map: t.StrMapping,
    *,
    keep_unmapped: bool = True,
) -> r[t.ContainerMapping]:
    fn: _MapDictKeysCallable = getattr(u, "map_dict_keys")
    return fn(source, key_map, keep_unmapped=keep_unmapped)


class AttrObject(BaseModel):
    """AttrObject class."""

    name: Annotated[
        str,
        Field(default="name", description="Attribute recursive container name"),
    ] = "name"
    value: Annotated[
        int,
        Field(default=1, description="Attribute recursive container value"),
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


def _parse_int(value: t.ValueOrModel) -> int:
    return int(cast("str", value))


def _plus_one(value: int) -> int:
    return value + 1


def _times_two(value: int) -> int:
    return value * 2


def _raise_value_error(_value: t.Scalar) -> Never:
    msg = "x"
    raise ValueError(msg)


def _normalize_not_dict(_value: t.RecursiveContainer) -> str:
    return "not-a-dict"


def _negative(value: int) -> bool:
    return value < 0


_PYRIGHT_USED_HELPERS = (
    _BadIter,
    _extract_field_obj,
    _take_obj,
    _build_apply_convert_obj,
    _extract_transform_options_obj,
    _build_apply_sort_obj,
    _build_apply_unique_obj,
    _build_apply_slice_obj,
    _build_apply_group_obj,
    _transform_obj,
    _map_dict_keys_obj,
    _parse_int,
    _plus_one,
    _times_two,
    _raise_value_error,
    _normalize_not_dict,
    _negative,
)


def test_bad_string_and_bad_bool_raise_value_error() -> None:
    with pytest.raises(ValueError, match="cannot stringify"):
        _ = str(BadString())
    with pytest.raises(ValueError, match="cannot bool"):
        _ = bool(BadBool())


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


def test_extract_array_index_helpers(mapper: type[u]) -> None:
    idx_result = mapper._extract_handle_array_index("x", "0")
    tm.fail(idx_result)
    tm.that(str(idx_result.error), has="Not a sequence")
    idx_neg = mapper._extract_handle_array_index([1, 2], "-1")
    tm.ok(idx_neg)
    tm.that(idx_neg.value, eq=2)
    idx_bad = mapper._extract_handle_array_index([1, 2], "bad")
    tm.fail(idx_bad)
    tm.that(str(idx_bad.error), has="Invalid index")


def test_extract_error_paths_and_prop_accessor(mapper: type[u]) -> None:
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
    tm.fail(res_terminal_none)
    tm.that(str(res_terminal_none.error), has="Extracted value is None")

    class NotGeneral:
        @override
        def __str__(self) -> str:
            return "converted"

    class Container:
        field: NotGeneral = NotGeneral()

    res_non_general = mapper.extract(
        cast("p.AccessibleData", cast("object", Container())),
        "field",
    )
    tm.ok(res_non_general)
    tm.that(res_non_general.value, eq="converted")

    class ExplodingModelDump:
        def __init__(self) -> None:
            self.model_dump = lambda: (_ for _ in ()).throw(ValueError("boom"))

    res_exception = mapper.extract(
        cast(
            "p.AccessibleData",
            cast("object", ExplodingModelDump()),
        ),
        "a",
    )
    tm.fail(res_exception)
    tm.that(str(res_exception.error).lower(), has="extract failed")
    accessor = mapper.prop("name")
    tm.that(
        accessor(
            cast(
                "t.ConfigMap | BaseModel",
                cast("p.AccessibleData", AttrObject(name="x", value=1)),
            ),
        ),
        eq="x",
    )
    tm.that(
        mapper.prop("missing")(
            cast("t.ConfigModelInput", cast("t.RecursiveContainer", {"a": 1})),
        ),
        eq="",
    )
