"""Utilities mapper full coverage tests."""

from __future__ import annotations

from collections import UserDict, UserList
from collections.abc import (
    ItemsView,
    Iterator,
    Mapping,
    MutableSequence,
    Sequence,
    Set as AbstractSet,
)
from pathlib import Path
from typing import Annotated, Never, Protocol, cast, override

import pytest
from pydantic import BaseModel, Field

from flext_core import r
from flext_tests import tm
from tests import p, t, u


class UtilitiesMapperFullCoverageNamespace:
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
        def __call__(
            self, item: AttrObject, field_name: str
        ) -> t.RecursiveContainer: ...

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
    ) -> t.ContainerMapping | t.ContainerList | t.RecursiveContainer:
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
    ) -> t.ContainerMapping | t.ContainerList | t.RecursiveContainer:
        fn: _BuildApplyOpCallable = getattr(u, "_op_sort")
        return fn(current, operations, None, "stop")

    @staticmethod
    def _build_apply_unique_obj(
        current: tuple[int, int, int],
        operations: Mapping[str, t.MapperInput],
    ) -> t.ContainerMapping | t.ContainerList | t.RecursiveContainer:
        fn: _BuildApplyOpCallable = getattr(u, "_op_unique")
        return fn(current, operations, None, "stop")

    @staticmethod
    def _build_apply_slice_obj(
        current: tuple[int, int, int],
        operations: Mapping[str, t.MapperInput],
    ) -> t.ContainerMapping | t.ContainerList | t.RecursiveContainer:
        fn: _BuildApplyOpCallable = getattr(u, "_op_slice")
        return fn(current, operations, None, "stop")

    @staticmethod
    def _build_apply_group_obj(
        current: Sequence[_GroupModel],
        operations: Mapping[str, t.MapperInput],
    ) -> t.ContainerMapping | t.ContainerList | t.RecursiveContainer:
        fn: _BuildApplyOpCallable = getattr(u, "_op_group")
        return fn(current, operations, None, "stop")

    @staticmethod
    def _transform_obj(
        source: BadMapping,
        **kwargs: t.StrMapping,
    ) -> r[t.ContainerMapping]:
        fn: _TransformCallable = getattr(u, "transform")
        return fn(source, **kwargs)

    @staticmethod
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

    @staticmethod
    def _parse_int(value: t.ValueOrModel) -> int:
        return int(cast("str", value))

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

    @staticmethod
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

    @staticmethod
    @pytest.fixture
    def mapper() -> type[u]:
        return u

    @staticmethod
    def test_type_guards_and_narrowing_failures(mapper: type[u]) -> None:
        with pytest.raises(TypeError, match="Cannot narrow"):
            mapper._narrow_to_configuration_dict(10)
        with pytest.raises(TypeError, match="Cannot narrow"):
            mapper._narrow_to_sequence("not-sequence")

    @staticmethod
    def test_narrow_to_string_keyed_dict_and_mapping_paths(mapper: type[u]) -> None:
        converted = mapper._narrow_to_string_keyed_dict(
            cast("t.RecursiveContainer", {1: "x", "b": "y"}),
        )
        tm.that(converted, has="1")
        tm.that(converted["b"], is_=str)
        with pytest.raises(TypeError, match="Cannot narrow"):
            mapper._narrow_to_string_keyed_dict(123)

    @staticmethod
    def test_general_value_helpers_and_logger(mapper: type[u]) -> None:

        class Stable:
            @override
            def __str__(self) -> str:
                return "stable"

        tm.that(
            mapper.narrow_to_container(cast("t.MetadataOrValue", Stable())),
            eq="stable",
        )
        tm.that(mapper._get_str_from_dict({"k": 2}, "k", default=""), eq="2")
        tm.that(mapper._get_str_from_dict({"k": None}, "k", default="d"), eq="d")
        callable_result = mapper._get_callable_from_dict({"x": 1}, "x")
        tm.fail(callable_result)
        tm.that(u().logger, none=False)

    @staticmethod
    def test_extract_array_index_helpers(mapper: type[u]) -> None:
        idx_result = mapper._extract_handle_array_index("x", "0")
        tm.fail(idx_result)
        tm.that(idx_result.error, eq="Not a sequence")
        idx_neg = mapper._extract_handle_array_index([1, 2], "-1")
        tm.ok(idx_neg)
        tm.that(idx_neg.value, eq=2)
        idx_bad = mapper._extract_handle_array_index([1, 2], "bad")
        tm.fail(idx_bad)
        tm.that(str(idx_bad.error), has="Invalid index")

    @staticmethod
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

    @staticmethod
    def test_take_and_as_branches(mapper: type[u]) -> None:
        model = _PortModel(port=8081, nested={})
        tm.that(_take_obj(model, "port"), eq=8081)
        tm.that(mapper.take(123, "port", default="d"), eq="d")
        tm.that(mapper.take({"port": None}, "port", default="x"), eq="x")
        tm.that(_take_obj(123, 2), eq="")

    @staticmethod
    def test_extract_field_value_and_ensure_variants(mapper: type[u]) -> None:
        tm.that(_extract_field_obj(AttrObject(name="a", value=2), "value"), eq=2)
        tm.that(_extract_field_obj(AttrObject(), "missing"), none=True)
        tm.that(mapper._op_ensure(5, {"ensure": "str"}, None, "stop"), eq="5")
        tm.that(mapper._op_ensure(5, {"ensure": "list"}, None, "stop"), eq=[5])
        tm.that(
            mapper._op_ensure([1, "a"], {"ensure": "str_list"}, None, "stop"),
            eq=["1", "a"],
        )
        tm.that(mapper._op_ensure(5, {"ensure": "dict"}, None, "stop"), eq={})
        tm.that(mapper._op_ensure(5, {"ensure": "unknown"}, None, "stop"), eq=5)

    @staticmethod
    def test_filter_map_normalize_convert_helpers(mapper: type[u]) -> None:
        plus_one = cast("t.MapperCallable", _plus_one)
        times_two = cast("t.MapperCallable", _times_two)
        tm.that(mapper._op_filter(1, {"filter": 1}, 0, "stop"), eq=1)
        tm.that(
            mapper._op_filter(
                {"a": 1, "b": 0},
                cast(
                    "Mapping[str, t.MapperInput]",
                    {"filter": bool},
                ),
                0,
                "stop",
            ),
            eq={"a": 1},
        )
        tm.that(
            mapper._op_filter(
                0,
                cast(
                    "Mapping[str, t.MapperInput]",
                    {"filter": bool},
                ),
                "d",
                "stop",
            ),
            eq="d",
        )
        tm.that(mapper._op_map(1, {"map": 1}, None, "stop"), eq=1)
        tm.that(
            mapper._op_map(
                {"a": 1},
                cast(
                    "Mapping[str, t.MapperInput]",
                    {"map": plus_one},
                ),
                None,
                "stop",
            ),
            eq={"a": 2},
        )
        tm.that(
            mapper._op_map(
                2,
                cast(
                    "Mapping[str, t.MapperInput]",
                    {"map": times_two},
                ),
                None,
                "stop",
            ),
            eq=4,
        )
        tm.that(
            mapper._op_normalize("ABC", {"normalize": "lower"}, None, "stop"), eq="abc"
        )
        tm.that(
            mapper._op_normalize(["ABC", 1], {"normalize": "lower"}, None, "stop"),
            eq=[
                "abc",
                1,
            ],
        )
        tm.that(mapper._op_normalize(1, {"normalize": "lower"}, None, "stop"), eq=1)
        tm.that(mapper._op_convert(1, {"convert": "not-callable"}, None, "stop"), eq=1)

    @staticmethod
    @pytest.mark.parametrize(
        ("value", "convert_spec", "expected"),
        [
            pytest.param("bad", int, 0, id="int-fallback"),
            pytest.param("bad", float, 0.0, id="float-fallback"),
            pytest.param(1, list, [], id="list-fallback"),
            pytest.param(1, dict, {}, id="dict-fallback"),
            pytest.param(1, tuple, (), id="tuple-fallback"),
            pytest.param(1, set, [], id="set-fallback"),
            pytest.param(1, _raise_value_error, 1, id="callable-error-fallback"),
        ],
    )
    def test_convert_default_fallback_matrix(
        mapper: type[u],
        value: str | int,
        convert_spec: type[
            int
            | float
            | MutableSequence[t.Scalar]
            | t.MutableConfigurationMapping
            | tuple[t.Scalar, ...]
            | set[t.Scalar]
        ]
        | t.MapperCallable,
        expected: float
        | MutableSequence[t.Scalar]
        | t.MutableConfigurationMapping
        | tuple[t.Scalar, ...],
    ) -> None:
        operations = cast(
            "Mapping[str, t.MapperInput]",
            {"convert": convert_spec},
        )
        tm.that(_build_apply_convert_obj(value, operations), eq=expected)

    @staticmethod
    def test_convert_sequence_branch_returns_tuple(mapper: type[u]) -> None:
        converted = _build_apply_convert_obj(
            ("bad",),
            cast(
                "Mapping[str, t.MapperInput]",
                {"convert": int},
            ),
        )
        tm.that(converted, eq=(0,))

    @staticmethod
    def test_transform_option_extract_and_step_helpers(
        mapper: type[u],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        opts = {
            "map_keys": {1: "one", "a": "b"},
            "normalize": True,
            "strip_none": True,
            "strip_empty": True,
            "filter_keys": {"a"},
            "exclude_keys": {"x"},
            "to_json": True,
        }
        extracted = _extract_transform_options_obj(
            cast("Mapping[str, t.MapperInput]", opts),
        )
        extracted_dict = extracted[3]
        if isinstance(extracted_dict, Mapping):
            tm.that(extracted_dict, eq={"1": "one", "a": "b"})
        monkeypatch.setattr(
            u,
            "normalize_component",
            staticmethod(_normalize_not_dict),
        )
        tm.that(mapper._apply_normalize({"a": 1}, normalize=True), eq={"a": 1})
        tm.that(mapper._apply_map_keys({"a": 1}, map_keys={"a": "A"}), eq={"A": 1})
        tm.that(
            mapper._apply_filter_keys({"a": 1, "b": 2}, filter_keys={"b"}),
            eq={"b": 2},
        )
        tm.that(
            mapper._apply_exclude_keys({"a": 1, "b": 2}, exclude_keys={"a"}),
            eq={"b": 2},
        )
        tm.that(mapper._apply_strip_none({"a": None}, strip_none=False), eq={"a": None})
        tm.that(mapper._apply_strip_empty({"a": ""}, strip_empty=False), eq={"a": ""})
        tm.that(
            {str(key): val.as_posix() for key, val in {"a": Path("/tmp")}.items()}["a"],
            eq="/tmp",
        )

    @staticmethod
    def test_build_apply_transform_and_process_error_paths(
        mapper: type[u],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tm.that(
            mapper._op_transform({"a": 1}, {"transform": 1}, {}, "stop"),
            eq={
                "a": 1,
            },
        )

        def explode_transform_steps(
            _result: t.ContainerMapping,
            *,
            _normalize: bool,
            _map_keys: t.StrMapping | None,
            _filter_keys: AbstractSet[str] | None,
            _exclude_keys: AbstractSet[str] | None,
            _strip_none: bool,
            _strip_empty: bool,
            _to_json: bool,
        ) -> t.ContainerMapping:
            raise RuntimeError(msg)

        msg = "explode transform"
        monkeypatch.setattr(
            mapper,
            "_apply_transform_steps",
            staticmethod(explode_transform_steps),
        )
        tm.that(
            mapper._op_transform({"a": 1}, {"transform": {}}, "d", "stop"),
            eq={
                "a": 1,
            },
        )
        tm.that(
            mapper._op_transform({"a": 1}, {"transform": {}}, "d", "skip"),
            eq={
                "a": 1,
            },
        )
        tm.that(mapper._op_process(1, {"process": 1}, 0, "stop"), eq=1)
        process_map_ops = cast(
            "Mapping[str, t.MapperInput]",
            {"process": _plus_one},
        )
        tm.that(
            mapper._op_process({"a": 1}, process_map_ops, 0, "stop"),
            eq={"a": 2},
        )
        process_fail_ops = cast(
            "Mapping[str, t.MapperInput]",
            {"process": _raise_value_error},
        )
        tm.that(mapper._op_process(1, process_fail_ops, 7, "stop"), eq=7)
        tm.that(mapper._op_process(1, process_fail_ops, 7, "skip"), eq=1)

    @staticmethod
    def test_group_sort_unique_slice_chunk_branches(mapper: type[u]) -> None:
        tm.that(mapper._op_group(1, {"group": "k"}, None, "stop"), eq=1)
        grouped = mapper._op_group(
            [{"kind": "a", "v": 1}, {"kind": "a", "v": 2}],
            {"group": "kind"},
            None,
            "stop",
        )
        tm.that(
            grouped,
            eq=cast(
                "t.Tests.Testobject",
                {"a": [{"kind": "a", "v": 1}, {"kind": "a", "v": 2}]},
            ),
        )
        tm.that(mapper._op_group([1, 2], {"group": 5}, None, "stop"), eq=[1, 2])
        tm.that(mapper._op_sort(1, {"sort": True}, None, "stop"), eq=1)
        sorted_with_scalar = mapper._op_sort(
            [{"name": "b"}, 3, {"name": "a"}],
            {"sort": "name"},
            None,
            "stop",
        )
        tm.that(sorted_with_scalar, is_=list)
        bad_sort_ops = cast(
            "Mapping[str, t.MapperInput]",
            {"sort": _raise_value_error},
        )
        bad_sort = mapper._op_sort([1, 2], bad_sort_ops, None, "stop")
        tm.that(bad_sort, eq=[1, 2])
        sorted_tuple = _build_apply_sort_obj(("b", "a"), {"sort": True})
        tm.that(sorted_tuple, eq=("a", "b"))
        tm.that(mapper._op_unique(1, {"unique": True}, None, "stop"), eq=1)
        tm.that(_build_apply_unique_obj((1, 2, 1), {"unique": True}), eq=(1, 2))
        tm.that(mapper._op_slice(1, {"slice": (0, 1)}, None, "stop"), eq=1)
        tm.that(_build_apply_slice_obj((1, 2, 3), {"slice": (1, 3)}), eq=(2, 3))
        tm.that(mapper._op_chunk(1, {"chunk": 2}, None, "stop"), eq=1)
        tm.that(mapper._op_chunk([1, 2], {"chunk": 0}, None, "stop"), eq=[1, 2])
        tm.that(mapper.build([1, 2], ops=None), eq=[1, 2])

    @staticmethod
    def test_transform_and_deep_eq_branches(mapper: type[u]) -> None:
        tm.ok(mapper.transform({"a": 1}, map_keys={"a": "A"}))
        bad_result = _transform_obj(BadMapping())
        tm.fail(bad_result)
        tm.that((bad_result.error or ""), has="iter exploded")
        d = {"a": 1}
        tm.that(mapper.deep_eq(d, d), eq=True)
        tm.that(not mapper.deep_eq({"a": 1}, {"a": 1, "b": 2}), eq=True)
        tm.that(not mapper.deep_eq({"a": 1}, {"b": 1}), eq=True)
        tm.that(not mapper.deep_eq({"a": None}, {"a": 1}), eq=True)
        tm.that(not mapper.deep_eq({"a": {"x": 1}}, {"a": {"x": 2}}), eq=True)
        tm.that(not mapper.deep_eq({"a": [1, 2]}, {"a": [1]}), eq=True)
        tm.that(not mapper.deep_eq({"a": [{"x": 1}]}, {"a": [{"x": 2}]}), eq=True)
        tm.that(not mapper.deep_eq({"a": [1, 2]}, {"a": [1, 3]}), eq=True)
        tm.that(not mapper.deep_eq({"a": 1}, {"a": 2}), eq=True)


_PortModel = UtilitiesMapperFullCoverageNamespace._PortModel
_MaybeModel = UtilitiesMapperFullCoverageNamespace._MaybeModel
_GroupModel = UtilitiesMapperFullCoverageNamespace._GroupModel
_BadItems = UtilitiesMapperFullCoverageNamespace._BadItems
_BadIter = UtilitiesMapperFullCoverageNamespace._BadIter
_ExtractFieldCallable = UtilitiesMapperFullCoverageNamespace._ExtractFieldCallable
_TakeCallable = UtilitiesMapperFullCoverageNamespace._TakeCallable
_BuildApplyConvertCallable = (
    UtilitiesMapperFullCoverageNamespace._BuildApplyConvertCallable
)
_ExtractTransformOptionsCallable = (
    UtilitiesMapperFullCoverageNamespace._ExtractTransformOptionsCallable
)
_BuildApplyOpCallable = UtilitiesMapperFullCoverageNamespace._BuildApplyOpCallable
_TransformCallable = UtilitiesMapperFullCoverageNamespace._TransformCallable
_MapDictKeysCallable = UtilitiesMapperFullCoverageNamespace._MapDictKeysCallable
_extract_field_obj = UtilitiesMapperFullCoverageNamespace._extract_field_obj
_take_obj = UtilitiesMapperFullCoverageNamespace._take_obj
_build_apply_convert_obj = UtilitiesMapperFullCoverageNamespace._build_apply_convert_obj
_extract_transform_options_obj = (
    UtilitiesMapperFullCoverageNamespace._extract_transform_options_obj
)
_build_apply_sort_obj = UtilitiesMapperFullCoverageNamespace._build_apply_sort_obj
_build_apply_unique_obj = UtilitiesMapperFullCoverageNamespace._build_apply_unique_obj
_build_apply_slice_obj = UtilitiesMapperFullCoverageNamespace._build_apply_slice_obj
_build_apply_group_obj = UtilitiesMapperFullCoverageNamespace._build_apply_group_obj
_transform_obj = UtilitiesMapperFullCoverageNamespace._transform_obj
_map_dict_keys_obj = UtilitiesMapperFullCoverageNamespace._map_dict_keys_obj
AttrObject = UtilitiesMapperFullCoverageNamespace.AttrObject
BadString = UtilitiesMapperFullCoverageNamespace.BadString
BadBool = UtilitiesMapperFullCoverageNamespace.BadBool
_parse_int = UtilitiesMapperFullCoverageNamespace._parse_int
_plus_one = UtilitiesMapperFullCoverageNamespace._plus_one
_times_two = UtilitiesMapperFullCoverageNamespace._times_two
_raise_value_error = UtilitiesMapperFullCoverageNamespace._raise_value_error
_normalize_not_dict = UtilitiesMapperFullCoverageNamespace._normalize_not_dict
_negative = UtilitiesMapperFullCoverageNamespace._negative
test_bad_string_and_bad_bool_raise_value_error = (
    UtilitiesMapperFullCoverageNamespace.test_bad_string_and_bad_bool_raise_value_error
)
ExplodingLenList = UtilitiesMapperFullCoverageNamespace.ExplodingLenList
BadMapping = UtilitiesMapperFullCoverageNamespace.BadMapping
mapper = UtilitiesMapperFullCoverageNamespace.mapper
test_type_guards_and_narrowing_failures = (
    UtilitiesMapperFullCoverageNamespace.test_type_guards_and_narrowing_failures
)
test_narrow_to_string_keyed_dict_and_mapping_paths = UtilitiesMapperFullCoverageNamespace.test_narrow_to_string_keyed_dict_and_mapping_paths
test_general_value_helpers_and_logger = (
    UtilitiesMapperFullCoverageNamespace.test_general_value_helpers_and_logger
)
test_extract_array_index_helpers = (
    UtilitiesMapperFullCoverageNamespace.test_extract_array_index_helpers
)
test_extract_error_paths_and_prop_accessor = (
    UtilitiesMapperFullCoverageNamespace.test_extract_error_paths_and_prop_accessor
)
test_take_and_as_branches = (
    UtilitiesMapperFullCoverageNamespace.test_take_and_as_branches
)
test_extract_field_value_and_ensure_variants = (
    UtilitiesMapperFullCoverageNamespace.test_extract_field_value_and_ensure_variants
)
test_filter_map_normalize_convert_helpers = (
    UtilitiesMapperFullCoverageNamespace.test_filter_map_normalize_convert_helpers
)
test_convert_default_fallback_matrix = (
    UtilitiesMapperFullCoverageNamespace.test_convert_default_fallback_matrix
)
test_convert_sequence_branch_returns_tuple = (
    UtilitiesMapperFullCoverageNamespace.test_convert_sequence_branch_returns_tuple
)
test_transform_option_extract_and_step_helpers = (
    UtilitiesMapperFullCoverageNamespace.test_transform_option_extract_and_step_helpers
)
test_build_apply_transform_and_process_error_paths = UtilitiesMapperFullCoverageNamespace.test_build_apply_transform_and_process_error_paths
test_group_sort_unique_slice_chunk_branches = (
    UtilitiesMapperFullCoverageNamespace.test_group_sort_unique_slice_chunk_branches
)
test_transform_and_deep_eq_branches = (
    UtilitiesMapperFullCoverageNamespace.test_transform_and_deep_eq_branches
)
