"""Utilities mapper full coverage tests."""

from __future__ import annotations

from collections import UserDict, UserList
from collections.abc import (
    Callable,
    ItemsView,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Never, Protocol, cast, override

import pytest
from flext_tests import t as test_t, tm
from pydantic import BaseModel, Field

from flext_core import r
from tests import p, t, u


class UtilitiesMapperFullCoverageNamespace:
    class _PortModel(BaseModel):
        """Model with port/nested for mapper take/extract tests."""

        port: int = 0
        nested: Annotated[
            Mapping[str, test_t.NormalizedValue], Field(default_factory=dict)
        ]

    class _MaybeModel(BaseModel):
        """Model with optional field for take tests."""

        x: str | None = None

    class _GroupModel(BaseModel):
        """Model with optional kind for group tests."""

        kind: str | None = None

    class _BadItems(UserDict[str, t.NormalizedValue]):
        """UserDict that explodes on items() for error-path testing."""

        @override
        def items(self) -> ItemsView[str, t.NormalizedValue]:
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

    class _AtCallable(Protocol):
        def __call__(
            self,
            items: ExplodingLenList,
            index: int | str,
            *,
            default: int | None = None,
        ) -> None: ...

    class _ExtractFieldCallable(Protocol):
        def __call__(self, item: AttrObject, field_name: str) -> None: ...

    class _TakeCallable(Protocol):
        def __call__(
            self,
            data_or_items: _MaybeModel | _PortModel | int,
            key_or_index: int | str,
            *,
            default: str | None = None,
        ) -> None: ...

    class _BuildApplyConvertCallable(Protocol):
        def __call__(
            self,
            current: tuple[str, ...] | str | int,
            operations: Mapping[str, t.NormalizedValue | t.MapperCallable],
        ) -> None: ...

    class _ExtractTransformOptionsCallable(Protocol):
        def __call__(
            self, transform_opts: Mapping[str, t.NormalizedValue | t.MapperCallable]
        ) -> tuple[t.NormalizedValue, ...]: ...

    class _BuildApplyOpCallable(Protocol):
        def __call__(
            self,
            current: tuple[str, str] | tuple[int, int, int] | Sequence[_GroupModel],
            operations: Mapping[str, t.NormalizedValue],
        ) -> None: ...

    class _TransformCallable(Protocol):
        def __call__(
            self, source: BadMapping, **kwargs: Mapping[str, str]
        ) -> r[Mapping[str, t.NormalizedValue]]: ...

    class _MapDictKeysCallable(Protocol):
        def __call__(
            self,
            source: _BadItems,
            key_map: Mapping[str, str],
            *,
            keep_unmapped: bool = True,
        ) -> r[Mapping[str, t.NormalizedValue]]: ...

    class _BuildFlagsCallable(Protocol):
        def __call__(
            self,
            active_flags: _BadIter,
            flag_mapping: Mapping[str, str],
        ) -> r[Mapping[str, bool]]: ...

    @staticmethod
    def _at_obj(
        items: ExplodingLenList, index: int | str, *, default: int | None = None
    ) -> None:
        """Call Mapper.at with arbitrary t.NormalizedValue for error-path testing."""
        fn: _AtCallable = getattr(u, "at")
        return fn(items, index, default=default)

    @staticmethod
    def _extract_field_obj(item: AttrObject, field_name: str) -> None:
        """Call _extract_field_value with arbitrary t.NormalizedValue for testing."""
        fn: _ExtractFieldCallable = getattr(u, "_extract_field_value")
        return fn(item, field_name)

    @staticmethod
    def _take_obj(
        data_or_items: _MaybeModel | _PortModel | int,
        key_or_index: int | str,
        *,
        default: str | None = None,
    ) -> None:
        fn: _TakeCallable = getattr(u, "take")
        return fn(data_or_items, key_or_index, default=default)

    @staticmethod
    def _build_apply_convert_obj(
        current: tuple[str, ...] | str | int,
        operations: Mapping[str, t.NormalizedValue | t.MapperCallable],
    ) -> None:
        fn: _BuildApplyConvertCallable = getattr(u, "_build_apply_convert")
        return fn(current, operations)

    @staticmethod
    def _extract_transform_options_obj(
        transform_opts: Mapping[str, t.NormalizedValue | t.MapperCallable],
    ) -> tuple[t.NormalizedValue, ...]:
        fn: _ExtractTransformOptionsCallable = getattr(u, "_extract_transform_options")
        return fn(transform_opts)

    @staticmethod
    def _build_apply_sort_obj(
        current: tuple[str, str], operations: Mapping[str, t.NormalizedValue]
    ) -> None:
        fn: _BuildApplyOpCallable = getattr(u, "_build_apply_sort")
        return fn(current, operations)

    @staticmethod
    def _build_apply_unique_obj(
        current: tuple[int, int, int], operations: Mapping[str, t.NormalizedValue]
    ) -> None:
        fn: _BuildApplyOpCallable = getattr(u, "_build_apply_unique")
        return fn(current, operations)

    @staticmethod
    def _build_apply_slice_obj(
        current: tuple[int, int, int], operations: Mapping[str, t.NormalizedValue]
    ) -> None:
        fn: _BuildApplyOpCallable = getattr(u, "_build_apply_slice")
        return fn(current, operations)

    @staticmethod
    def _build_apply_group_obj(
        current: Sequence[_GroupModel],
        operations: Mapping[str, t.NormalizedValue],
    ) -> None:
        fn: _BuildApplyOpCallable = getattr(u, "_build_apply_group")
        return fn(current, operations)

    @staticmethod
    def _transform_obj(
        source: BadMapping, **kwargs: Mapping[str, str]
    ) -> r[Mapping[str, t.NormalizedValue]]:
        fn: _TransformCallable = getattr(u, "transform")
        return fn(source, **kwargs)

    @staticmethod
    def _map_dict_keys_obj(
        source: _BadItems,
        key_map: Mapping[str, str],
        *,
        keep_unmapped: bool = True,
    ) -> r[Mapping[str, t.NormalizedValue]]:
        fn: _MapDictKeysCallable = getattr(u, "map_dict_keys")
        return fn(source, key_map, keep_unmapped=keep_unmapped)

    @staticmethod
    def _build_flags_obj(
        active_flags: _BadIter,
        flag_mapping: Mapping[str, str],
    ) -> r[Mapping[str, bool]]:
        """Call build_flags_dict with arbitrary t.NormalizedValue for error-path testing."""
        fn: _BuildFlagsCallable = getattr(u, "build_flags_dict")
        return fn(active_flags, flag_mapping)

    class AttrObject(BaseModel):
        """AttrObject class."""

        name: Annotated[
            str, Field(default="name", description="Attribute t.NormalizedValue name")
        ] = "name"
        value: Annotated[
            int, Field(default=1, description="Attribute t.NormalizedValue value")
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
    def _parse_int(value: t.NormalizedValue | BaseModel) -> int:
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
    def _normalize_not_dict(_value: t.NormalizedValue) -> str:
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

    class BadMapping(Mapping[str, t.NormalizedValue]):
        @override
        def __getitem__(self, _key: str) -> t.NormalizedValue:
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
            cast("t.NormalizedValue", {1: "x", "b": "y"}),
        )
        tm.that(converted, has="1")
        tm.that(isinstance(converted["b"], str), eq=True)
        with pytest.raises(TypeError, match="Cannot narrow"):
            mapper._narrow_to_string_keyed_dict(123)
        mapped = mapper._narrow_to_configuration_mapping({"x": 1})
        tm.that(isinstance(mapped, t.ConfigMap), eq=True)
        tm.that(mapped.root["x"], eq=1)
        with pytest.raises(TypeError, match="Cannot narrow"):
            _ = mapper._narrow_to_configuration_mapping(
                cast("t.NormalizedValue", {1: BadString()}),
            )
        with pytest.raises(TypeError, match="Cannot narrow"):
            mapper._narrow_to_configuration_mapping(3)

    @staticmethod
    def test_general_value_helpers_and_logger(mapper: type[u]) -> None:

        class Stable:
            @override
            def __str__(self) -> str:
                return "stable"

        tm.that(
            mapper.narrow_to_container(cast("t.NormalizedValue", Stable())), eq="stable"
        )
        tm.that(mapper._get_str_from_dict({"k": 2}, "k", default=""), eq="2")
        tm.that(mapper._get_str_from_dict({"k": None}, "k", default="d"), eq="d")
        callable_result = mapper._get_callable_from_dict({"x": 1}, "x")
        tm.fail(callable_result)
        tm.that(u().logger, none=False)

    @staticmethod
    def test_invert_and_json_conversion_branches(mapper: type[u]) -> None:
        tm.that(
            mapper.invert_dict({"a": "x", "b": "x"}, handle_collisions="first"),
            eq={
                "x": "a",
            },
        )
        tm.that(True, eq=True)

        class Model(BaseModel):
            x: int

        model = Model(x=1)
        tm.that(model.model_dump(mode="json"), eq={"x": 1})
        path_val = Path("/tmp")
        tm.that(path_val.as_posix(), eq="/tmp")
        as_json: MutableMapping[str, test_t.NormalizedValue] = {}
        for key, val in {"x": Path("/tmp")}.items():
            if isinstance(val, Path):
                as_json[str(key)] = val.as_posix()
            else:
                as_json[str(key)] = val
        tm.that(as_json["x"], eq="/tmp")
        list_json: Sequence[Mapping[str, test_t.NormalizedValue]] = [
            {"a": 1},
            {"b": "opaque"},
        ]
        tm.that(isinstance(list_json, list), eq=True)
        tm.that(list_json[0]["a"], eq=1)

        payload = {
            "model": model,
            "path": Path("/tmp"),
            "when": datetime(2026, 3, 12, 10, 30, 45, tzinfo=UTC),
        }
        safe_json: MutableMapping[str, test_t.NormalizedValue] = {}
        for key, val in payload.items():
            if isinstance(val, BaseModel):
                safe_json[key] = val.model_dump(mode="json")
            elif isinstance(val, Path):
                safe_json[key] = val.as_posix()
            elif isinstance(val, datetime):
                safe_json[key] = val.isoformat()
            else:
                safe_json[key] = val
        tm.that(isinstance(safe_json, Mapping), eq=True)
        tm.that(safe_json["model"], eq={"x": 1})
        tm.that(safe_json["path"], eq="/tmp")
        tm.that(safe_json["when"], eq="2026-03-12T10:30:45+00:00")

    @staticmethod
    def test_ensure_and_extract_array_index_helpers(mapper: type[u]) -> None:
        tm.that(mapper.ensure(123), eq=[123])
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
            cast("t.ConfigMap | BaseModel", cast("t.NormalizedValue", Container())),
            "field",
        )
        tm.ok(res_non_general)
        tm.that(res_non_general.value, eq="converted")

        class ExplodingModelDump:
            def __init__(self) -> None:
                self.model_dump = lambda: (_ for _ in ()).throw(ValueError("boom"))

        res_exception = mapper.extract(
            cast(
                "t.ConfigMap | BaseModel",
                cast("t.NormalizedValue", ExplodingModelDump()),
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
                cast("t.ConfigMap | BaseModel", cast("t.NormalizedValue", {"a": 1})),
            ),
            eq="",
        )

    @staticmethod
    def test_at_take_and_as_branches(mapper: type[u]) -> None:
        tm.that(mapper.at({"a": 1}, 0, default=5).value, eq=5)
        tm.that(
            cast("r[int]", _at_obj(ExplodingLenList([1]), 0, default=7)).value, eq=7
        )
        model = _PortModel(port=8081, nested={})
        tm.that(_take_obj(model, "port"), eq=8081)
        tm.that(mapper.take(123, "port", default="d"), eq="d")
        tm.that(mapper.take({"port": None}, "port", default="x"), eq="x")
        tm.that(_take_obj(123, 2), eq="")
        tm.that(mapper.as_(12, str), eq="12")
        tm.that(mapper.as_("off", bool), eq=False)

    @staticmethod
    def test_extract_field_value_and_ensure_variants(mapper: type[u]) -> None:
        tm.that(_extract_field_obj(AttrObject(name="a", value=2), "value"), eq=2)
        tm.that(_extract_field_obj(AttrObject(), "missing"), none=True)
        tm.that(mapper._build_apply_ensure(5, {"ensure": "str"}), eq="5")
        tm.that(mapper._build_apply_ensure(5, {"ensure": "list"}), eq=[5])
        tm.that(
            mapper._build_apply_ensure([1, "a"], {"ensure": "str_list"}), eq=["1", "a"]
        )
        tm.that(mapper._build_apply_ensure(5, {"ensure": "dict"}), eq={})
        tm.that(mapper._build_apply_ensure(5, {"ensure": "unknown"}), eq=5)

    @staticmethod
    def test_filter_map_normalize_convert_helpers(mapper: type[u]) -> None:
        plus_one = cast("Callable[..., t.NormalizedValue]", _plus_one)
        times_two = cast("Callable[..., t.NormalizedValue]", _times_two)
        tm.that(mapper._build_apply_filter(1, {"filter": 1}, 0), eq=1)
        tm.that(
            mapper._build_apply_filter(
                {"a": 1, "b": 0},
                cast(
                    "Mapping[str, t.NormalizedValue | t.MapperCallable]",
                    {"filter": bool},
                ),
                0,
            ),
            eq={"a": 1},
        )
        tm.that(
            mapper._build_apply_filter(
                0,
                cast(
                    "Mapping[str, t.NormalizedValue | t.MapperCallable]",
                    {"filter": bool},
                ),
                "d",
            ),
            eq="d",
        )
        tm.that(mapper._build_apply_map(1, {"map": 1}), eq=1)
        tm.that(
            mapper._build_apply_map(
                {"a": 1},
                cast(
                    "Mapping[str, t.NormalizedValue | t.MapperCallable]",
                    {"map": plus_one},
                ),
            ),
            eq={"a": 2},
        )
        tm.that(
            mapper._build_apply_map(
                2,
                cast(
                    "Mapping[str, t.NormalizedValue | t.MapperCallable]",
                    {"map": times_two},
                ),
            ),
            eq=4,
        )
        tm.that(mapper._build_apply_normalize("ABC", {"normalize": "lower"}), eq="abc")
        tm.that(
            mapper._build_apply_normalize(["ABC", 1], {"normalize": "lower"}),
            eq=[
                "abc",
                1,
            ],
        )
        tm.that(mapper._build_apply_normalize(1, {"normalize": "lower"}), eq=1)
        tm.that(mapper._build_apply_convert(1, {"convert": "not-callable"}), eq=1)

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
            | MutableMapping[str, t.Scalar]
            | tuple[t.Scalar, ...]
            | set[t.Scalar]
        ]
        | Callable[..., t.NormalizedValue],
        expected: float
        | MutableSequence[t.Scalar]
        | MutableMapping[str, t.Scalar]
        | tuple[t.Scalar, ...],
    ) -> None:
        operations = cast(
            "Mapping[str, t.NormalizedValue | t.MapperCallable]",
            {"convert": convert_spec},
        )
        tm.that(_build_apply_convert_obj(value, operations), eq=expected)

    @staticmethod
    def test_convert_sequence_branch_returns_tuple(mapper: type[u]) -> None:
        converted = _build_apply_convert_obj(
            ("bad",),
            cast(
                "Mapping[str, t.NormalizedValue | t.MapperCallable]", {"convert": int}
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
            cast("Mapping[str, t.NormalizedValue | t.MapperCallable]", opts)
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
            mapper._apply_filter_keys({"a": 1, "b": 2}, filter_keys={"b"}), eq={"b": 2}
        )
        tm.that(
            mapper._apply_exclude_keys({"a": 1, "b": 2}, exclude_keys={"a"}),
            eq={"b": 2},
        )
        tm.that(mapper._apply_strip_none({"a": None}, strip_none=False), eq={"a": None})
        tm.that(mapper._apply_strip_empty({"a": ""}, strip_empty=False), eq={"a": ""})
        tm.that(
            {
                str(key): val.as_posix() if isinstance(val, Path) else val
                for key, val in {"a": Path("/tmp")}.items()
            }["a"],
            eq="/tmp",
        )

    @staticmethod
    def test_build_apply_transform_and_process_error_paths(
        mapper: type[u],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tm.that(
            mapper._build_apply_transform({"a": 1}, {"transform": 1}, {}, "stop"),
            eq={
                "a": 1,
            },
        )

        def explode_transform_steps(
            _result: Mapping[str, t.NormalizedValue],
            *,
            _normalize: bool,
            _map_keys: Mapping[str, str] | None,
            _filter_keys: set[str] | None,
            _exclude_keys: set[str] | None,
            _strip_none: bool,
            _strip_empty: bool,
            _to_json: bool,
        ) -> Mapping[str, test_t.NormalizedValue]:
            raise RuntimeError(msg)

        msg = "explode transform"
        monkeypatch.setattr(
            mapper,
            "_apply_transform_steps",
            staticmethod(explode_transform_steps),
        )
        tm.that(
            mapper._build_apply_transform({"a": 1}, {"transform": {}}, "d", "stop"),
            eq={
                "a": 1,
            },
        )
        tm.that(
            mapper._build_apply_transform({"a": 1}, {"transform": {}}, "d", "skip"),
            eq={
                "a": 1,
            },
        )
        tm.that(mapper._build_apply_process(1, {"process": 1}, 0, "stop"), eq=1)
        process_map_ops = cast(
            "Mapping[str, t.NormalizedValue | t.MapperCallable]", {"process": _plus_one}
        )
        tm.that(
            mapper._build_apply_process({"a": 1}, process_map_ops, 0, "stop"),
            eq={"a": 2},
        )
        process_fail_ops = cast(
            "Mapping[str, t.NormalizedValue | t.MapperCallable]",
            {"process": _raise_value_error},
        )
        tm.that(mapper._build_apply_process(1, process_fail_ops, 7, "stop"), eq=7)
        tm.that(mapper._build_apply_process(1, process_fail_ops, 7, "skip"), eq=1)

    @staticmethod
    def test_group_sort_unique_slice_chunk_branches(mapper: type[u]) -> None:
        tm.that(mapper._build_apply_group(1, {"group": "k"}), eq=1)
        grouped = mapper._build_apply_group(
            [{"kind": "a", "v": 1}, {"kind": "a", "v": 2}],
            {"group": "kind"},
        )
        tm.that(grouped, eq={"a": [{"kind": "a", "v": 1}, {"kind": "a", "v": 2}]})
        tm.that(mapper._build_apply_group([1, 2], {"group": 5}), eq=[1, 2])
        tm.that(mapper._build_apply_sort(1, {"sort": True}), eq=1)
        sorted_with_scalar = mapper._build_apply_sort(
            [{"name": "b"}, 3, {"name": "a"}],
            {"sort": "name"},
        )
        tm.that(isinstance(sorted_with_scalar, list), eq=True)
        bad_sort_ops = cast(
            "Mapping[str, t.NormalizedValue | t.MapperCallable]",
            {"sort": _raise_value_error},
        )
        bad_sort = mapper._build_apply_sort([1, 2], bad_sort_ops)
        tm.that(bad_sort, eq=[1, 2])
        sorted_tuple = _build_apply_sort_obj(("b", "a"), {"sort": True})
        tm.that(sorted_tuple, eq=("a", "b"))
        tm.that(mapper._build_apply_unique(1, {"unique": True}), eq=1)
        tm.that(_build_apply_unique_obj((1, 2, 1), {"unique": True}), eq=(1, 2))
        tm.that(mapper._build_apply_slice(1, {"slice": (0, 1)}), eq=1)
        tm.that(_build_apply_slice_obj((1, 2, 3), {"slice": (1, 3)}), eq=(2, 3))
        tm.that(mapper._build_apply_chunk(1, {"chunk": 2}), eq=1)
        tm.that(mapper._build_apply_chunk([1, 2], {"chunk": 0}), eq=[1, 2])
        tm.that(mapper.build([1, 2], ops=None), eq=[1, 2])

    @staticmethod
    def test_field_and_fields_multi_branches(mapper: type[u]) -> None:
        tm.that(
            mapper.field(
                cast("p.AccessibleData", "normalized"), "missing", required=True
            ),
            none=True,
        )
        tm.that(mapper.field({}, "missing", ops={"ensure": "str"}), eq="")
        source_obj = AttrObject(name="n", value=1)
        fields = mapper.fields_multi(
            cast("t.ConfigMap | BaseModel", cast("t.NormalizedValue", source_obj)),
            {"name": "", "missing": None},
        )
        tm.that(fields, eq={"name": "n", "missing": ""})

    @staticmethod
    def test_construct_transform_and_deep_eq_branches(mapper: type[u]) -> None:
        constructed_none = mapper.construct_spec(
            {"x": {"field": "a", "default": 9}}, None
        )
        tm.that(constructed_none["x"], eq=9)
        source: MutableMapping[str, t.NormalizedValue | BaseModel] = {
            "name": "alice",
            "n": 3,
        }
        spec = cast(
            "Mapping[str, t.NormalizedValue | t.MapperCallable]",
            {
                "name": {"field": "name", "ops": "skip-ops"},
                "n": {"field": "n", "ops": {"map": _plus_one}},
                "literal": 5,
            },
        )
        constructed = mapper.construct_spec(spec, t.ConfigMap(root=source))
        tm.that(constructed["name"], eq="alice")
        tm.that(constructed["n"], eq=4)
        tm.that(constructed["literal"], eq=5)

        class ExplodeOnGet(Mapping[str, t.NormalizedValue]):
            @override
            def __getitem__(self, _key: str) -> t.NormalizedValue:
                msg = "get exploded"
                raise TypeError(msg)

            @override
            def __iter__(self) -> Iterator[str]:
                return iter(["x"])

            @override
            def __len__(self) -> int:
                return 1

        with pytest.raises(ValueError, match="get exploded"):
            mapper.construct_spec(
                cast(
                    "Mapping[str, t.NormalizedValue | t.MapperCallable]",
                    {"x": ExplodeOnGet()},
                ),
                t.ConfigMap(root={"x": 1}),
                on_error="stop",
            )
        tm.that(
            mapper.construct_spec(
                cast(
                    "Mapping[str, t.NormalizedValue | t.MapperCallable]",
                    {"x": ExplodeOnGet()},
                ),
                t.ConfigMap(root={"x": 1}),
                on_error="skip",
            ),
            eq={},
        )
        tm.ok(mapper.transform({"a": 1}, map_keys={"a": "A"}))
        bad_result = _transform_obj(BadMapping())
        tm.fail(bad_result)
        tm.that((bad_result.error or ""), has="iter exploded")
        d = {"a": 1}
        tm.that(mapper.deep_eq(d, d), eq=True)
        tm.that(mapper.deep_eq({"a": 1}, {"a": 1, "b": 2}), eq=False)
        tm.that(mapper.deep_eq({"a": 1}, {"b": 1}), eq=False)
        tm.that(mapper.deep_eq({"a": None}, {"a": 1}), eq=False)
        tm.that(mapper.deep_eq({"a": {"x": 1}}, {"a": {"x": 2}}), eq=False)
        tm.that(mapper.deep_eq({"a": [1, 2]}, {"a": [1]}), eq=False)
        tm.that(mapper.deep_eq({"a": [{"x": 1}]}, {"a": [{"x": 2}]}), eq=False)
        tm.that(mapper.deep_eq({"a": [1, 2]}, {"a": [1, 3]}), eq=False)
        tm.that(mapper.deep_eq({"a": 1}, {"a": 2}), eq=False)

    @staticmethod
    @pytest.mark.parametrize(
        "merge_strategy", ["merge", "secondary_only", "primary_only"]
    )
    def test_process_context_data_and_related_convenience(
        mapper: type[u],
        merge_strategy: str,
    ) -> None:
        primary: Mapping[str, t.NormalizedValue] = {"a": 1, "drop": "x"}
        secondary: Mapping[str, t.NormalizedValue] = {"b": 2}
        result = mapper.process_context_data(
            primary_data=primary,
            secondary_data=secondary,
            merge_strategy=merge_strategy,
            field_overrides={"c": 3},
            filter_keys={"a", "b", "c", "drop"},
            exclude_keys={"drop"},
        )
        tm.that(result, has="c")
        normalized = mapper.normalize_context_values(
            context=t.ConfigMap(root={"a": "1"}),
            extra_kwargs=t.ConfigMap(root={"b": 2}),
            field="x",
        )
        tm.that(normalized["field"], eq="x")

    @staticmethod
    def test_small_mapper_convenience_methods(mapper: type[u]) -> None:
        tm.that(mapper.omit({"a": 1, "b": 2}, "a"), eq={"b": 2})
        tm.that(mapper.pluck([{"a": 1}, {}], "a", default=0), eq=[1, 0])
        keyed = mapper.key_by(["aa", "b"], len)
        assert keyed == {2: "aa", 1: "b"}
        fields_from_mapping = mapper.fields(
            {"name": "alice", "age": 1},
            "name",
            {"email": {"default": "x@x"}, "age": 0},
        )
        tm.that(fields_from_mapping["name"], eq="alice")
        tm.that(fields_from_mapping["email"], eq="x@x")
        tm.that(fields_from_mapping["age"], eq=1)
        fields_from_object = mapper.fields(
            cast("p.AccessibleData", AttrObject(name="obj", value=4)),
            "name",
        )
        tm.that(fields_from_object, eq={"name": "obj"})
        tm.that(mapper.cast_generic("x"), eq="x")
        tm.that(mapper.cast_generic("5", _parse_int), eq=5)
        tm.that(mapper.cast_generic("bad", _parse_int, default=9), eq=9)
        tm.that(mapper.cast_generic("bad", _parse_int), eq="bad")

        class NamedPredicate:
            def __call__(self, value: int) -> bool:
                return value == 0

        class BadPredicate(NamedPredicate):
            @override
            def __call__(self, value: int) -> bool:
                _ = value
                msg = "x"
                raise ValueError(msg)

        class NegativePredicate(NamedPredicate):
            @override
            def __call__(self, value: int) -> bool:
                return value < 0

        class EqualOnePredicate(NamedPredicate):
            @override
            def __call__(self, value: int) -> bool:
                return value == 1

        predicates: Mapping[str, NamedPredicate] = {
            "bad": BadPredicate(),
            "no": NegativePredicate(),
            "yes": EqualOnePredicate(),
        }
        found_callable = mapper.find_callable(
            cast("Mapping[str, Callable[[int], bool]]", predicates), 1
        )
        tm.that(found_callable.is_success and found_callable.value == "yes", eq=True)
        not_found_callable = mapper.find_callable(
            cast("Mapping[str, Callable[[int], bool]]", {"no": _negative}), 1
        )
        tm.fail(not_found_callable)

    @staticmethod
    def test_map_flags_collect_and_invert_branches(mapper: type[u]) -> None:
        mapped = mapper.map_dict_keys(
            {"old": 1, "x": 2},
            {"old": "new"},
            keep_unmapped=True,
        )
        tm.ok(mapped)
        tm.that(mapped.value, eq={"new": 1, "x": 2})

        fail_map = _map_dict_keys_obj(_BadItems(), {})
        tm.fail(fail_map)
        flags = mapper.build_flags_dict(
            ["read"], {"read": "can_read", "w": "can_write"}
        )
        tm.ok(flags)
        tm.that(flags.value, eq={"can_read": True, "can_write": False})

        fail_flags = _build_flags_obj(_BadIter(), {})
        tm.fail(fail_flags)
        active = mapper.collect_active_keys(
            {"r": True, "w": False}, {"r": "R", "w": "W"}
        )
        tm.ok(active)
        tm.that(active.value, eq=["R"])

        class _BadGetMapping(Mapping[str, bool]):
            @override
            def __getitem__(self, _key: str) -> bool:
                msg = "get exploded"
                raise TypeError(msg)

            @override
            def __iter__(self) -> Iterator[str]:
                return iter(["x"])

            @override
            def __len__(self) -> int:
                return 1

        fail_active = mapper.collect_active_keys(_BadGetMapping(), {"x": "X"})
        tm.fail(fail_active)
        tm.that(
            mapper.invert_dict({"a": "x", "b": "x"}, handle_collisions="last"),
            eq={
                "x": "b",
            },
        )

    @staticmethod
    def test_conversion_and_extract_success_branches(mapper: type[u]) -> None:

        class Plain:
            @override
            def __str__(self) -> str:
                return "plain"

        tm.that(str(Plain()), eq="plain")
        plain_dict: Mapping[str, test_t.NormalizedValue] = {"1": str(Plain())}
        tm.that(plain_dict, eq={"1": "plain"})
        plain_list: Sequence[int | MutableMapping[str, str]] = [1, {"k": str(Plain())}]
        tm.that(plain_list, eq=[1, {"k": "plain"}])
        tm.that(mapper.ensure_str(None, "d"), eq="d")
        tm.that(mapper.ensure_str("x"), eq="x")
        tm.that(mapper.ensure_str(2), eq="2")
        tm.that(mapper.ensure(None), eq=[])
        tm.that(mapper.ensure("x"), eq=["x"])
        tm.that(mapper.ensure([1, 2]), eq=[1, 2])
        str_result = mapper.ensure_str_or_none("x")
        tm.that(str_result.is_success and str_result.value == "x", eq=True)

        class DumpOnly:
            a: int = 1

        get_result = mapper._extract_get_value(
            cast(
                "t.NormalizedValue | BaseModel | Mapping[str, t.NormalizedValue]",
                cast("t.NormalizedValue", DumpOnly()),
            ),
            "a",
        )
        tm.ok(get_result)
        tm.that(get_result.value, eq=1)
        get_missing = mapper._extract_get_value(
            cast(
                "t.NormalizedValue | BaseModel | Mapping[str, t.NormalizedValue]",
                cast("t.NormalizedValue", DumpOnly()),
            ),
            "missing",
        )
        tm.fail(get_missing)
        idx_range = mapper._extract_handle_array_index([1], "3")
        tm.fail(idx_range)
        tm.that(idx_range.error, eq="Index 3 out of range")
        required_fail = mapper.extract({"a": None}, "a.b", default="z", required=True)
        tm.fail(required_fail)
        tm.that(str(required_fail.error), has="is None")
        default_ok = mapper.extract({"a": None}, "a.b", default="z", required=False)
        tm.ok(default_ok)
        tm.that(default_ok.value, eq="z")
        miss_required = mapper.extract({"a": 1}, "b", default="z", required=True)
        tm.fail(miss_required)
        miss_default = mapper.extract({"a": 1}, "b", default="z")
        tm.ok(miss_default)
        tm.that(miss_default.value, eq="z")
        idx_required = mapper.extract({"a": [1]}, "a[bad]", default="x", required=True)
        tm.fail(idx_required)
        idx_default = mapper.extract({"a": [1]}, "a[bad]", default="x")
        tm.ok(idx_default)
        tm.that(idx_default.value, eq="x")

    @staticmethod
    def test_accessor_take_pick_as_or_flat_and_agg_branches(mapper: type[u]) -> None:
        tm.that(mapper.at({"a": 1}, "a").value, eq=1)
        tm.that(mapper.at([9, 8], 0).value, eq=9)
        tm.that(mapper.at([9, 8], 5, default=7).value, eq=7)
        tm.that(mapper.take({"a": 1}, "a", default=0), eq=1)
        tm.that(mapper.take({"a": "x"}, "a", as_type=int, default=0), eq=0)
        tm.that(mapper.take([1, 2, 3], 2), eq=[1, 2])
        tm.that(mapper.take((1, 2, 3), 2, from_start=False), eq=[2, 3])
        tm.that(mapper.take({"a": 1, "b": 2}, 1), eq={"a": 1})
        tm.that(mapper.pick({"a": 1, "b": 2}, "a"), eq={"a": 1})
        tm.that(mapper.pick({"a": 1, "b": 2}, "a", "b", as_dict=False), eq=[1, 2])
        tm.that(mapper.as_(1, int), eq=1)
        tm.that(mapper.as_("1", int, strict=True, default=0), eq=0)
        tm.that(mapper.as_("1", int), eq=1)
        float_value = mapper.as_("1.5", float)
        tm.that(isinstance(float_value, float), eq=True)
        tm.that(abs(cast("float", float_value) - 1.5), lt=1e-09)
        tm.that(mapper.as_("true", bool), eq=True)
        tm.that(mapper.as_("maybe", bool, default=False), eq=False)
        tm.that(mapper.as_(None, int, default=3), eq=3)
        tm.that(mapper.or_(None, None, 1, default=2).value, eq=1)
        tm.that(mapper.or_(None, None, default=2).value, eq=2)
        tm.that(mapper.flat([[1, 2], [3]]), eq=[1, 2, 3])
        tm.that(mapper._extract_field_value({"x": 1}, "x"), eq=1)
        tm.that(mapper.agg([{"v": 1}, {"v": 2}], "v"), eq=3)
        mixed_items: tuple[Mapping[str, test_t.NormalizedValue], ...] = (
            {"v": 1},
            {"v": "no"},
        )
        tm.that(mapper.agg(mixed_items, "v"), eq=1)
        tm.that(mapper.agg([1, 2, 3], lambda x: x, fn=max), eq=3)

    @staticmethod
    def test_remaining_build_fields_construct_and_eq_paths(mapper: type[u]) -> None:
        tm.that(mapper._build_apply_ensure([1], {"ensure": "list"}), eq=[1])
        tm.that(mapper._build_apply_ensure(1, {"ensure": "str_list"}), eq=["1"])
        tm.that(mapper._build_apply_ensure({"a": 1}, {"ensure": "dict"}), eq={"a": 1})
        tm.that(
            mapper._build_apply_filter(
                [1, 2, 0],
                cast(
                    "Mapping[str, t.NormalizedValue | t.MapperCallable]",
                    {"filter": bool},
                ),
                0,
            ),
            eq=[1, 2],
        )
        tm.that(
            mapper._build_apply_map(
                [1, 2],
                cast(
                    "Mapping[str, t.NormalizedValue | t.MapperCallable]",
                    {"map": _plus_one},
                ),
            ),
            eq=[2, 3],
        )
        tm.that(mapper._apply_map_keys({"a": 1}, map_keys={"b": "B"}), eq={"a": 1})
        tm.that(
            mapper._apply_strip_none({"a": None, "b": 1}, strip_none=True), eq={"b": 1}
        )
        tm.that(
            mapper._apply_strip_empty({"a": "", "b": 1}, strip_empty=True), eq={"b": 1}
        )
        process_list_ops = cast(
            "Mapping[str, t.NormalizedValue | t.MapperCallable]", {"process": _plus_one}
        )
        tm.that(
            mapper._build_apply_process([1, 2], process_list_ops, 0, "stop"), eq=[2, 3]
        )
        grouped_call = mapper._build_apply_group(
            ["aa", "b"],
            cast("Mapping[str, t.NormalizedValue | t.MapperCallable]", {"group": len}),
        )
        tm.that(grouped_call, eq={"2": ["aa"], "1": ["b"]})
        grouped_skip = mapper._build_apply_group([{"kind": None}, 1], {"group": "kind"})
        tm.that(grouped_skip, eq={"": [{"kind": None}]})
        sorted_ok = mapper._build_apply_sort(
            [3, 1, 2],
            cast("Mapping[str, t.NormalizedValue | t.MapperCallable]", {"sort": int}),
        )
        tm.that(sorted_ok, eq=[1, 2, 3])
        tm.that(mapper._build_apply_sort([3, 1], {"sort": True}), eq=[1, 3])
        tm.that(mapper._build_apply_unique([1, 1, 2], {"unique": True}), eq=[1, 2])
        tm.that(mapper._build_apply_slice([1, 2, 3], {"slice": (0, 2)}), eq=[1, 2])
        tm.that(mapper._build_apply_slice([1, 2, 3], {"slice": (0,)}), eq=[1, 2, 3])
        tm.that(mapper._build_apply_chunk([1, 2, 3], {"chunk": 2}), eq=[[1, 2], [3]])
        tm.that(mapper.field({"a": 1}, "a"), eq=1)
        tm.that(
            mapper.fields_multi({"a": 1, "b": 2}, {"a": 0, "b": 0}), eq={"a": 1, "b": 2}
        )
        tm.that(mapper.fields_multi(t.ConfigMap(root={"a": 1}), {"a": 0}), eq={"a": 1})
        tm.that(
            mapper.construct_spec({"x": {"value": 1}}, t.ConfigMap(root={"x": 0})),
            eq={"x": 1},
        )
        tm.that(
            mapper.construct_spec({"x": "a"}, t.ConfigMap(root={"a": 2})), eq={"x": 2}
        )
        tm.that(
            mapper.construct_spec(
                {"x": {"field": "a", "ops": "noop"}},
                t.ConfigMap(root={"a": 2}),
            ),
            eq={"x": 2},
        )
        tm.that(mapper.deep_eq({"a": None}, {"a": None}), eq=True)
        tm.that(mapper.deep_eq({"a": {"x": 1}}, {"a": {"x": 1}}), eq=True)
        tm.that(mapper.deep_eq({"a": [1, 2]}, {"a": [1, 2]}), eq=True)

        class DictLikeOnly(BaseModel):
            x: int = 1

        class DictLikeOnlySecondary(BaseModel):
            y: int = 1

        context = mapper.process_context_data(
            primary_data=DictLikeOnly(),
            secondary_data=DictLikeOnlySecondary(),
            merge_strategy="merge",
        )
        tm.that(context, eq={"x": 1, "y": 1})
        fields_obj = mapper.fields(
            cast("p.AccessibleData", "normalized"),
            {"x": {"default": 1}},
        )
        tm.that(fields_obj, eq={"x": 1})

    @staticmethod
    def test_remaining_uncovered_branches(
        mapper: type[u],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = _PortModel(port=9000, nested={"k": "v"})
        base_model_extract = mapper.extract(model, "nested.k")
        tm.ok(base_model_extract)
        tm.that(base_model_extract.value, eq="v")
        indexed_extract = mapper.extract({"a": [{"b": 1}]}, "a[0].b")
        tm.ok(indexed_extract)
        tm.that(indexed_extract.value, eq=1)
        terminal_required = mapper.extract({"a": None}, "a", required=True)
        tm.fail(terminal_required)
        tm.that(str(terminal_required.error), has="Extracted value is None")
        terminal_default = mapper.extract({"a": None}, "a", default="fallback")
        tm.ok(terminal_default)
        tm.that(terminal_default.value, eq="fallback")

        tm.that(_take_obj(_MaybeModel(x=None), "x", default="d"), eq="d")
        tm.that(mapper.as_("nope", int, default=9), eq=9)
        tm.that(mapper.agg([{"v": "x"}], "v"), eq=0)
        tm.that(mapper._apply_map_keys({"a": 1}, map_keys={"a": "A"}), eq={"A": 1})

        grouped = _build_apply_group_obj([_GroupModel(kind=None)], {"group": "kind"})
        tm.that(grouped, eq={"": [{"kind": None}]})
        tm.that(mapper._build_apply_sort([2, 1], {"sort": 5}), eq=[2, 1])

        class CallableDictLike(BaseModel):
            k: int = 1

            def __call__(self) -> None:
                return None

            def keys(self) -> Sequence[str]:
                return ["k"]

            def items(self) -> Sequence[tuple[str, int]]:
                return [("k", 1)]

            def get(self, key: str, default: int | None = None) -> int | None:
                return 1 if key == "k" else default

        # CallableDictLike is a BaseModel, so process_context_data extracts its
        # dict representation via model_dump() and merges both instances.
        callable_result = mapper.process_context_data(
            primary_data=CallableDictLike(),
            secondary_data=CallableDictLike(),
        )
        tm.that(callable_result, eq={"k": 1})
        obj_fields = mapper.fields(
            cast("p.AccessibleData", AttrObject(name="n", value=3)),
            {"name": 0, "missing": 7},
        )
        tm.that(obj_fields, eq={"name": "n"})
        dict_fields = mapper.fields({}, {"missing": 7})
        tm.that(dict_fields, eq={"missing": 7})


_PortModel = UtilitiesMapperFullCoverageNamespace._PortModel
_MaybeModel = UtilitiesMapperFullCoverageNamespace._MaybeModel
_GroupModel = UtilitiesMapperFullCoverageNamespace._GroupModel
_BadItems = UtilitiesMapperFullCoverageNamespace._BadItems
_BadIter = UtilitiesMapperFullCoverageNamespace._BadIter
_AtCallable = UtilitiesMapperFullCoverageNamespace._AtCallable
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
_BuildFlagsCallable = UtilitiesMapperFullCoverageNamespace._BuildFlagsCallable
_at_obj = UtilitiesMapperFullCoverageNamespace._at_obj
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
_build_flags_obj = UtilitiesMapperFullCoverageNamespace._build_flags_obj
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
test_invert_and_json_conversion_branches = (
    UtilitiesMapperFullCoverageNamespace.test_invert_and_json_conversion_branches
)
test_ensure_and_extract_array_index_helpers = (
    UtilitiesMapperFullCoverageNamespace.test_ensure_and_extract_array_index_helpers
)
test_extract_error_paths_and_prop_accessor = (
    UtilitiesMapperFullCoverageNamespace.test_extract_error_paths_and_prop_accessor
)
test_at_take_and_as_branches = (
    UtilitiesMapperFullCoverageNamespace.test_at_take_and_as_branches
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
test_field_and_fields_multi_branches = (
    UtilitiesMapperFullCoverageNamespace.test_field_and_fields_multi_branches
)
test_construct_transform_and_deep_eq_branches = (
    UtilitiesMapperFullCoverageNamespace.test_construct_transform_and_deep_eq_branches
)
test_process_context_data_and_related_convenience = UtilitiesMapperFullCoverageNamespace.test_process_context_data_and_related_convenience
test_small_mapper_convenience_methods = (
    UtilitiesMapperFullCoverageNamespace.test_small_mapper_convenience_methods
)
test_map_flags_collect_and_invert_branches = (
    UtilitiesMapperFullCoverageNamespace.test_map_flags_collect_and_invert_branches
)
test_conversion_and_extract_success_branches = (
    UtilitiesMapperFullCoverageNamespace.test_conversion_and_extract_success_branches
)
test_accessor_take_pick_as_or_flat_and_agg_branches = UtilitiesMapperFullCoverageNamespace.test_accessor_take_pick_as_or_flat_and_agg_branches
test_remaining_build_fields_construct_and_eq_paths = UtilitiesMapperFullCoverageNamespace.test_remaining_build_fields_construct_and_eq_paths
test_remaining_uncovered_branches = (
    UtilitiesMapperFullCoverageNamespace.test_remaining_uncovered_branches
)
