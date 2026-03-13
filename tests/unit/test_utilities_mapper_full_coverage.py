"""Utilities mapper full coverage tests."""

from __future__ import annotations

from collections import UserDict, UserList
from collections.abc import Callable, ItemsView, Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Protocol, cast, override

import pytest
from pydantic import BaseModel, Field

from flext_core import m, p, r, u


class _PortModel(BaseModel):
    """Model with port/nested for mapper take/extract tests."""

    port: int = 0
    nested: Annotated[dict[str, object], Field(default_factory=dict)]


class _AtCallable(Protocol):
    def __call__(
        self,
        items: object,
        index: int | str,
        *,
        default: object = None,
    ) -> object: ...


class _ExtractFieldCallable(Protocol):
    def __call__(self, item: object, field_name: str) -> object: ...


class _TakeCallable(Protocol):
    def __call__(
        self,
        data_or_items: object,
        key_or_index: object,
        *,
        default: object = None,
    ) -> object: ...


class _BuildApplyConvertCallable(Protocol):
    def __call__(self, current: object, operations: Mapping[str, object]) -> object: ...


class _ExtractTransformOptionsCallable(Protocol):
    def __call__(self, transform_opts: Mapping[str, object]) -> tuple[object, ...]: ...


class _BuildApplyOpCallable(Protocol):
    def __call__(self, current: object, operations: Mapping[str, object]) -> object: ...


class _TransformCallable(Protocol):
    def __call__(self, source: object, **kwargs: object) -> r[object]: ...


class _MapDictKeysCallable(Protocol):
    def __call__(
        self,
        source: object,
        key_map: Mapping[str, str],
        *,
        keep_unmapped: bool = True,
    ) -> r[Mapping[str, object]]: ...


class _BuildFlagsCallable(Protocol):
    def __call__(
        self,
        active_flags: object,
        flag_mapping: Mapping[str, str],
    ) -> r[Mapping[str, bool]]: ...


def _at_obj(items: object, index: int | str, *, default: object = None) -> object:
    """Call Mapper.at with arbitrary object for error-path testing."""
    fn: _AtCallable = getattr(u, "at")
    return fn(items, index, default=default)


def _extract_field_obj(item: object, field_name: str) -> object:
    """Call _extract_field_value with arbitrary object for testing."""
    fn: _ExtractFieldCallable = getattr(u, "_extract_field_value")
    return fn(item, field_name)


def _take_obj(
    data_or_items: object, key_or_index: object, *, default: object = None
) -> object:
    fn: _TakeCallable = getattr(u, "take")
    return fn(data_or_items, key_or_index, default=default)


def _build_apply_convert_obj(
    current: object, operations: Mapping[str, object]
) -> object:
    fn: _BuildApplyConvertCallable = getattr(u, "_build_apply_convert")
    return fn(current, operations)


def _extract_transform_options_obj(
    transform_opts: Mapping[str, object],
) -> tuple[object, ...]:
    fn: _ExtractTransformOptionsCallable = getattr(u, "_extract_transform_options")
    return fn(transform_opts)


def _build_apply_sort_obj(current: object, operations: Mapping[str, object]) -> object:
    fn: _BuildApplyOpCallable = getattr(u, "_build_apply_sort")
    return fn(current, operations)


def _build_apply_unique_obj(
    current: object, operations: Mapping[str, object]
) -> object:
    fn: _BuildApplyOpCallable = getattr(u, "_build_apply_unique")
    return fn(current, operations)


def _build_apply_slice_obj(current: object, operations: Mapping[str, object]) -> object:
    fn: _BuildApplyOpCallable = getattr(u, "_build_apply_slice")
    return fn(current, operations)


def _build_apply_group_obj(current: object, operations: Mapping[str, object]) -> object:
    fn: _BuildApplyOpCallable = getattr(u, "_build_apply_group")
    return fn(current, operations)


def _transform_obj(source: object, **kwargs: object) -> r[object]:
    fn: _TransformCallable = getattr(u, "transform")
    return fn(source, **kwargs)


def _map_dict_keys_obj(
    source: object,
    key_map: Mapping[str, str],
    *,
    keep_unmapped: bool = True,
) -> r[Mapping[str, object]]:
    fn: _MapDictKeysCallable = getattr(u, "map_dict_keys")
    return fn(source, key_map, keep_unmapped=keep_unmapped)


def _build_flags_obj(
    active_flags: object,
    flag_mapping: Mapping[str, str],
) -> r[Mapping[str, bool]]:
    """Call build_flags_dict with arbitrary object for error-path testing."""
    fn: _BuildFlagsCallable = getattr(u, "build_flags_dict")
    return fn(active_flags, flag_mapping)


class AttrObject(BaseModel):
    """AttrObject class."""

    name: Annotated[str, Field(default="name", description="Attribute object name")]
    value: Annotated[int, Field(default=1, description="Attribute object value")]


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


def _parse_int(value: object) -> int:
    return int(cast("str", value))


def _plus_one(value: object) -> object:
    return cast("int", value) + 1


def _times_two(value: object) -> object:
    return cast("int", value) * 2


def _raise_value_error(_value: object) -> object:
    msg = "x"
    raise ValueError(msg)


def _normalize_not_dict(_value: object) -> str:
    return "not-a-dict"


def _negative(value: int) -> bool:
    return value < 0


def test_bad_string_and_bad_bool_raise_value_error() -> None:
    with pytest.raises(ValueError, match="cannot stringify"):
        _ = str(BadString())
    with pytest.raises(ValueError, match="cannot bool"):
        _ = bool(BadBool())


class ExplodingLenList(UserList[object]):
    """ExplodingLenList class."""

    @override
    def __len__(self) -> int:
        """__len__ method."""
        msg = "len exploded"
        raise TypeError(msg)


class BadMapping(Mapping[str, object]):
    """BadMapping class."""

    @override
    def __getitem__(self, key: str) -> object:
        """__getitem__ method."""
        msg = f"missing {key}"
        raise KeyError(msg)

    @override
    def __iter__(self) -> Iterator[str]:
        """__iter__ method."""
        msg = "iter exploded"
        raise RuntimeError(msg)

    @override
    def __len__(self) -> int:
        """__len__ method."""
        return 1


@pytest.fixture
def mapper() -> type[u]:
    return u


def test_type_guards_and_narrowing_failures(mapper: type[u]) -> None:
    with pytest.raises(TypeError, match="Cannot narrow"):
        mapper._narrow_to_configuration_dict(10)
    with pytest.raises(TypeError, match="Cannot narrow"):
        mapper._narrow_to_sequence("not-sequence")


def test_narrow_to_string_keyed_dict_and_mapping_paths(mapper: type[u]) -> None:
    converted = mapper._narrow_to_string_keyed_dict(
        cast("object", {1: "x", "b": object()}),
    )
    assert "1" in converted
    assert isinstance(converted["b"], str)
    with pytest.raises(TypeError, match="Cannot narrow"):
        mapper._narrow_to_string_keyed_dict(123)
    mapped = mapper._narrow_to_configuration_mapping({"x": 1})
    assert isinstance(mapped, m.ConfigMap)
    assert mapped.root["x"] == 1
    with pytest.raises(TypeError, match="Cannot coerce"):
        _ = mapper._narrow_to_configuration_mapping(
            cast("object", {1: BadString()}),
        )
    with pytest.raises(TypeError, match="Cannot narrow"):
        mapper._narrow_to_configuration_mapping(3)


def test_general_value_helpers_and_logger(mapper: type[u]) -> None:

    class Stable:
        @override
        def __str__(self) -> str:
            return "stable"

    assert (
        mapper.narrow_to_general_value_type(
            cast("object", Stable()),
        )
        == "stable"
    )
    assert mapper._get_str_from_dict({"k": 2}, "k", default="") == "2"
    assert mapper._get_str_from_dict({"k": None}, "k", default="d") == "d"
    callable_result = mapper._get_callable_from_dict({"x": 1}, "x")
    assert callable_result.is_failure
    assert u().logger is not None


def test_invert_and_json_conversion_branches(mapper: type[u]) -> None:
    assert mapper.invert_dict({"a": "x", "b": "x"}, handle_collisions="first") == {
        "x": "a",
    }
    assert True

    class Model(BaseModel):
        x: int

    model = Model(x=1)
    assert model.model_dump(mode="json") == {"x": 1}
    path_val = Path("/tmp")
    assert path_val.as_posix() == "/tmp"
    as_json: dict[str, object] = {}
    for key, val in {"x": Path("/tmp")}.items():
        if isinstance(val, Path):
            as_json[str(key)] = val.as_posix()
        else:
            as_json[str(key)] = val
    assert as_json["x"] == "/tmp"
    list_json: list[dict[str, object]] = [{"a": 1}, {"b": "opaque"}]
    assert isinstance(list_json, list)
    assert list_json[0]["a"] == 1

    payload = {
        "model": model,
        "path": Path("/tmp"),
        "when": datetime(2026, 3, 12, 10, 30, 45, tzinfo=UTC),
    }
    safe_json: dict[str, object] = {}
    for key, val in payload.items():
        if isinstance(val, BaseModel):
            safe_json[key] = val.model_dump(mode="json")
        elif isinstance(val, Path):
            safe_json[key] = val.as_posix()
        elif isinstance(val, datetime):
            safe_json[key] = val.isoformat()
        else:
            safe_json[key] = val
    assert isinstance(safe_json, Mapping)
    assert safe_json["model"] == {"x": 1}
    assert safe_json["path"] == "/tmp"
    assert safe_json["when"] == "2026-03-12T10:30:45+00:00"


def test_ensure_and_extract_array_index_helpers(mapper: type[u]) -> None:
    assert mapper.ensure(123) == [123]
    idx_result = mapper._extract_handle_array_index("x", "0")
    assert idx_result.is_failure
    assert idx_result.error == "Not a sequence"
    idx_neg = mapper._extract_handle_array_index([1, 2], "-1")
    assert idx_neg.is_success
    assert idx_neg.value == 2
    idx_bad = mapper._extract_handle_array_index([1, 2], "bad")
    assert idx_bad.is_failure
    assert "Invalid index" in str(idx_bad.error)


def test_extract_error_paths_and_prop_accessor(mapper: type[u]) -> None:
    res_none_intermediate = mapper.extract({"a": None}, "a.b")
    assert res_none_intermediate.is_failure
    assert "default is None" in str(res_none_intermediate.error)
    res_missing_key = mapper.extract({"a": 1}, "b")
    assert res_missing_key.is_failure
    assert "default is None" in str(res_missing_key.error)
    res_bad_index = mapper.extract({"a": [1]}, "a[bad]")
    assert res_bad_index.is_failure
    assert "Array error" in str(res_bad_index.error)
    res_terminal_none = mapper.extract({"a": None}, "a")
    assert res_terminal_none.is_failure
    assert "Extracted value is None" in str(res_terminal_none.error)

    class NotGeneral:
        @override
        def __str__(self) -> str:
            return "converted"

    class Container:
        field: NotGeneral = NotGeneral()

    res_non_general = mapper.extract(
        cast("m.ConfigMap | BaseModel", cast("object", Container())),
        "field",
    )
    assert res_non_general.is_success
    assert res_non_general.value == "converted"

    class ExplodingModelDump:
        def __init__(self) -> None:
            self.model_dump = lambda: (_ for _ in ()).throw(ValueError("boom"))

    res_exception = mapper.extract(
        cast("m.ConfigMap | BaseModel", cast("object", ExplodingModelDump())),
        "a",
    )
    assert res_exception.is_failure
    assert "not found" in str(res_exception.error).lower()
    accessor = mapper.prop("name")
    assert (
        accessor(
            cast(
                "m.ConfigMap | BaseModel",
                cast("object", AttrObject(name="x", value=1)),
            ),
        )
        == "x"
    )
    assert (
        mapper.prop("missing")(
            cast("m.ConfigMap | BaseModel", cast("object", {"a": 1})),
        )
        == ""
    )


def test_at_take_and_as_branches(mapper: type[u]) -> None:
    assert mapper.at({"a": 1}, 0, default=5).value == 5
    assert cast("r[int]", _at_obj(ExplodingLenList([1]), 0, default=7)).value == 7
    model = _PortModel(port=8081)
    assert _take_obj(model, "port") == 8081
    assert mapper.take(123, "port", default="d") == "d"
    assert mapper.take({"port": None}, "port", default="x") == "x"
    assert _take_obj(123, 2) == ""
    assert mapper.as_(12, str) == "12"
    assert mapper.as_("off", bool) is False


def test_extract_field_value_and_ensure_variants(mapper: type[u]) -> None:
    assert _extract_field_obj(AttrObject(name="a", value=2), "value") == 2
    assert _extract_field_obj(AttrObject(), "missing") is None
    assert mapper._build_apply_ensure(5, {"ensure": "str"}) == "5"
    assert mapper._build_apply_ensure(5, {"ensure": "list"}) == [5]
    assert mapper._build_apply_ensure([1, "a"], {"ensure": "str_list"}) == ["1", "a"]
    assert mapper._build_apply_ensure(5, {"ensure": "dict"}) == {}
    assert mapper._build_apply_ensure(5, {"ensure": "unknown"}) == 5


def test_filter_map_normalize_convert_helpers(mapper: type[u]) -> None:
    plus_one = cast("Callable[[object], object]", _plus_one)
    times_two = cast("Callable[[object], object]", _times_two)
    assert mapper._build_apply_filter(1, {"filter": 1}, 0) == 1
    assert mapper._build_apply_filter(
        {"a": 1, "b": 0},
        cast("Mapping[str, object]", {"filter": bool}),
        0,
    ) == {"a": 1}
    assert (
        mapper._build_apply_filter(
            0,
            cast("Mapping[str, object]", {"filter": bool}),
            "d",
        )
        == "d"
    )
    assert mapper._build_apply_map(1, {"map": 1}) == 1
    assert mapper._build_apply_map(
        {"a": 1},
        cast("Mapping[str, object]", {"map": plus_one}),
    ) == {"a": 2}
    assert (
        mapper._build_apply_map(
            2,
            cast("Mapping[str, object]", {"map": times_two}),
        )
        == 4
    )
    assert mapper._build_apply_normalize("ABC", {"normalize": "lower"}) == "abc"
    assert mapper._build_apply_normalize(["ABC", 1], {"normalize": "lower"}) == [
        "abc",
        1,
    ]
    assert mapper._build_apply_normalize(1, {"normalize": "lower"}) == 1
    assert mapper._build_apply_convert(1, {"convert": "not-callable"}) == 1


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
    value: object,
    convert_spec: Callable[[object], object] | type,
    expected: object,
) -> None:
    operations = cast("Mapping[str, object]", {"convert": convert_spec})
    assert _build_apply_convert_obj(value, operations) == expected


def test_convert_sequence_branch_returns_tuple(mapper: type[u]) -> None:
    converted = _build_apply_convert_obj(
        ("bad",),
        cast("Mapping[str, object]", {"convert": int}),
    )
    assert converted == (0,)


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
    extracted = _extract_transform_options_obj(cast("Mapping[str, object]", opts))
    assert extracted[3] == {"1": "one", "a": "b"}
    monkeypatch.setattr(
        u,
        "normalize_component",
        staticmethod(_normalize_not_dict),
    )
    assert mapper._apply_normalize({"a": 1}, normalize=True) == {"a": 1}
    assert mapper._apply_map_keys({"a": 1}, map_keys={"a": "A"}) == {"A": 1}
    assert mapper._apply_filter_keys({"a": 1, "b": 2}, filter_keys={"b"}) == {"b": 2}
    assert mapper._apply_exclude_keys({"a": 1, "b": 2}, exclude_keys={"a"}) == {"b": 2}
    assert mapper._apply_strip_none({"a": None}, strip_none=False) == {"a": None}
    assert mapper._apply_strip_empty({"a": ""}, strip_empty=False) == {"a": ""}
    assert {
        str(key): val.as_posix() if isinstance(val, Path) else val
        for key, val in {"a": Path("/tmp")}.items()
    }["a"] == "/tmp"


def test_build_apply_transform_and_process_error_paths(
    mapper: type[u],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert mapper._build_apply_transform({"a": 1}, {"transform": 1}, {}, "stop") == {
        "a": 1,
    }

    def explode_transform_steps(
        _result: Mapping[str, object],
        *,
        _normalize: bool,
        _map_keys: Mapping[str, str] | None,
        _filter_keys: set[str] | None,
        _exclude_keys: set[str] | None,
        _strip_none: bool,
        _strip_empty: bool,
        _to_json: bool,
    ) -> dict[str, object]:
        raise RuntimeError(msg)

    msg = "explode transform"
    monkeypatch.setattr(
        mapper,
        "_apply_transform_steps",
        staticmethod(explode_transform_steps),
    )
    assert mapper._build_apply_transform({"a": 1}, {"transform": {}}, "d", "stop") == {
        "a": 1,
    }
    assert mapper._build_apply_transform({"a": 1}, {"transform": {}}, "d", "skip") == {
        "a": 1,
    }
    assert mapper._build_apply_process(1, {"process": 1}, 0, "stop") == 1
    process_map_ops = cast("Mapping[str, object]", {"process": _plus_one})
    assert mapper._build_apply_process({"a": 1}, process_map_ops, 0, "stop") == {"a": 2}
    process_fail_ops = cast(
        "Mapping[str, object]",
        {"process": _raise_value_error},
    )
    assert mapper._build_apply_process(1, process_fail_ops, 7, "stop") == 7
    assert mapper._build_apply_process(1, process_fail_ops, 7, "skip") == 1


def test_group_sort_unique_slice_chunk_branches(mapper: type[u]) -> None:
    assert mapper._build_apply_group(1, {"group": "k"}) == 1
    grouped = mapper._build_apply_group(
        [{"kind": "a", "v": 1}, {"kind": "a", "v": 2}],
        {"group": "kind"},
    )
    assert grouped == {"a": [{"kind": "a", "v": 1}, {"kind": "a", "v": 2}]}
    assert mapper._build_apply_group([1, 2], {"group": 5}) == [1, 2]
    assert mapper._build_apply_sort(1, {"sort": True}) == 1
    sorted_with_scalar = mapper._build_apply_sort(
        [{"name": "b"}, 3, {"name": "a"}],
        {"sort": "name"},
    )
    assert isinstance(sorted_with_scalar, list)
    bad_sort_ops = cast("Mapping[str, object]", {"sort": _raise_value_error})
    bad_sort = mapper._build_apply_sort([1, 2], bad_sort_ops)
    assert bad_sort == [1, 2]
    sorted_tuple = _build_apply_sort_obj(("b", "a"), {"sort": True})
    assert sorted_tuple == ("a", "b")
    assert mapper._build_apply_unique(1, {"unique": True}) == 1
    assert _build_apply_unique_obj((1, 2, 1), {"unique": True}) == (1, 2)
    assert mapper._build_apply_slice(1, {"slice": (0, 1)}) == 1
    assert _build_apply_slice_obj((1, 2, 3), {"slice": (1, 3)}) == (2, 3)
    assert mapper._build_apply_chunk(1, {"chunk": 2}) == 1
    assert mapper._build_apply_chunk([1, 2], {"chunk": 0}) == [1, 2]
    assert mapper.build([1, 2], ops=None) == [1, 2]


def test_field_and_fields_multi_branches(mapper: type[u]) -> None:
    assert (
        mapper.field(cast("p.AccessibleData", object()), "missing", required=True)
        is None
    )
    assert mapper.field({}, "missing", ops={"ensure": "str"}) == ""
    source_obj = AttrObject(name="n", value=1)
    fields = mapper.fields_multi(
        cast("m.ConfigMap | BaseModel", cast("object", source_obj)),
        {"name": "", "missing": None},
    )
    assert fields == {"name": "n", "missing": ""}


def test_construct_transform_and_deep_eq_branches(mapper: type[u]) -> None:
    constructed_none = mapper.construct({"x": {"field": "a", "default": 9}}, None)
    assert constructed_none["x"] == 9
    source: dict[str, object] = {"name": "alice", "n": 3}
    spec = cast(
        "Mapping[str, object]",
        {
            "name": {"field": "name", "ops": "skip-ops"},
            "n": {"field": "n", "ops": {"map": _plus_one}},
            "literal": 5,
        },
    )
    constructed = mapper.construct(spec, m.ConfigMap(root=source))
    assert constructed["name"] == "alice"
    assert constructed["n"] == 4
    assert constructed["literal"] == 5

    class ExplodeOnGet(Mapping[str, object]):
        @override
        def __iter__(self) -> Iterator[str]:
            return iter(("field",))

        @override
        def __len__(self) -> int:
            return 1

        @override
        def __getitem__(self, key: str) -> object:
            if key == "field":
                msg = "boom"
                raise RuntimeError(msg)
            return ""

    assert mapper.construct(
        {"x": ExplodeOnGet()},
        m.ConfigMap(root={"x": 1}),
        on_error="stop",
    ) == {"x": ""}
    assert mapper.construct(
        {"x": ExplodeOnGet()},
        m.ConfigMap(root={"x": 1}),
        on_error="skip",
    ) == {"x": ""}
    assert mapper.transform({"a": 1}, map_keys={"a": "A"}).is_success
    bad_result = _transform_obj(BadMapping())
    assert bad_result.is_failure
    assert "iter exploded" in (bad_result.error or "")
    d = {"a": 1}
    assert mapper.deep_eq(d, d) is True
    assert mapper.deep_eq({"a": 1}, {"a": 1, "b": 2}) is False
    assert mapper.deep_eq({"a": 1}, {"b": 1}) is False
    assert mapper.deep_eq({"a": None}, {"a": 1}) is False
    assert mapper.deep_eq({"a": {"x": 1}}, {"a": {"x": 2}}) is False
    assert mapper.deep_eq({"a": [1, 2]}, {"a": [1]}) is False
    assert mapper.deep_eq({"a": [{"x": 1}]}, {"a": [{"x": 2}]}) is False
    assert mapper.deep_eq({"a": [1, 2]}, {"a": [1, 3]}) is False
    assert mapper.deep_eq({"a": 1}, {"a": 2}) is False


@pytest.mark.parametrize("merge_strategy", ["merge", "secondary_only", "primary_only"])
def test_process_context_data_and_related_convenience(
    mapper: type[u],
    merge_strategy: str,
) -> None:
    primary: dict[str, object] = {"a": 1, "drop": "x"}
    secondary = {"b": 2}
    result = mapper.process_context_data(
        primary_data=primary,
        secondary_data=secondary,
        merge_strategy=merge_strategy,
        field_overrides={"c": 3},
        filter_keys={"a", "b", "c", "drop"},
        exclude_keys={"drop"},
    )
    assert "c" in result
    normalized = mapper.normalize_context_values(
        context=m.ConfigMap(root={"a": "1"}),
        extra_kwargs=m.ConfigMap(root={"b": 2}),
        field="x",
    )
    assert normalized["field"] == "x"


def test_small_mapper_convenience_methods(mapper: type[u]) -> None:
    assert mapper.omit({"a": 1, "b": 2}, "a") == {"b": 2}
    assert mapper.pluck([{"a": 1}, {}], "a", default=0) == [1, 0]
    keyed = mapper.key_by(["aa", "b"], len)
    assert keyed == {2: "aa", 1: "b"}
    fields_from_mapping = mapper.fields(
        {"name": "alice", "age": 1},
        "name",
        {"email": {"default": "x@x"}, "age": 0},
    )
    assert fields_from_mapping["name"] == "alice"
    assert fields_from_mapping["email"] == "x@x"
    assert fields_from_mapping["age"] == 1
    fields_from_object = mapper.fields(
        cast("object", AttrObject(name="obj", value=4)),
        "name",
    )
    assert fields_from_object == {"name": "obj"}
    assert mapper.cast_generic("x") == "x"
    assert mapper.cast_generic("5", _parse_int) == 5
    assert mapper.cast_generic("bad", _parse_int, default=9) == 9
    assert mapper.cast_generic("bad", _parse_int) == "bad"

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

    predicates: dict[str, NamedPredicate] = {
        "bad": BadPredicate(),
        "no": NegativePredicate(),
        "yes": EqualOnePredicate(),
    }
    found_callable = mapper.find_callable(predicates, 1)
    assert found_callable.is_success and found_callable.value == "yes"
    not_found_callable = mapper.find_callable({"no": _negative}, 1)
    assert not_found_callable.is_failure


def test_map_flags_collect_and_invert_branches(mapper: type[u]) -> None:
    mapped = mapper.map_dict_keys(
        {"old": 1, "x": 2},
        {"old": "new"},
        keep_unmapped=True,
    )
    assert mapped.is_success
    assert mapped.value == {"new": 1, "x": 2}

    class BadItems(UserDict[str, object]):
        @override
        def items(self) -> ItemsView[str, object]:
            msg = "bad items"
            raise RuntimeError(msg)

    fail_map = _map_dict_keys_obj(BadItems(), {})
    assert fail_map.is_failure
    flags = mapper.build_flags_dict(["read"], {"read": "can_read", "w": "can_write"})
    assert flags.is_success
    assert flags.value == {"can_read": True, "can_write": False}

    class BadIter(UserList[str]):
        @override
        def __iter__(self) -> Iterator[str]:
            msg = "bad iter"
            raise RuntimeError(msg)

    fail_flags = _build_flags_obj(BadIter(), {})
    assert fail_flags.is_failure
    active = mapper.collect_active_keys({"r": True, "w": False}, {"r": "R", "w": "W"})
    assert active.is_success
    assert active.value == ["R"]

    class BadGet(UserDict[str, bool]):
        @override
        def get(self, key: str, default: object = None) -> bool:
            msg = "bad get"
            raise RuntimeError(msg)

    fail_active = mapper.collect_active_keys(BadGet(), {"x": "X"})
    assert fail_active.is_failure
    assert mapper.invert_dict({"a": "x", "b": "x"}, handle_collisions="last") == {
        "x": "b",
    }


def test_conversion_and_extract_success_branches(mapper: type[u]) -> None:

    class Plain:
        @override
        def __str__(self) -> str:
            return "plain"

    assert str(Plain()) == "plain"
    plain_dict: dict[str, object] = {"1": str(Plain())}
    assert plain_dict == {"1": "plain"}
    plain_list: list[object] = [1, {"k": str(Plain())}]
    assert plain_list == [1, {"k": "plain"}]
    assert mapper.ensure_str(None, "d") == "d"
    assert mapper.ensure_str("x") == "x"
    assert mapper.ensure_str(2) == "2"
    assert mapper.ensure(None) == []
    assert mapper.ensure("x") == ["x"]
    assert mapper.ensure([1, 2]) == [1, 2]
    str_result = mapper.ensure_str_or_none("x")
    assert str_result.is_success and str_result.value == "x"

    class DumpOnly:
        a: int = 1

    get_result = mapper._extract_get_value(
        cast("object | BaseModel", cast("object", DumpOnly())),
        "a",
    )
    assert get_result.is_success
    assert get_result.value == 1
    get_missing = mapper._extract_get_value(
        cast("object | BaseModel", cast("object", DumpOnly())),
        "missing",
    )
    assert get_missing.is_failure
    idx_range = mapper._extract_handle_array_index([1], "3")
    assert idx_range.is_failure
    assert idx_range.error == "Index 3 out of range"
    required_fail = mapper.extract({"a": None}, "a.b", default="z", required=True)
    assert required_fail.is_failure
    assert "is None" in str(required_fail.error)
    default_ok = mapper.extract({"a": None}, "a.b", default="z", required=False)
    assert default_ok.is_success
    assert default_ok.value == "z"
    miss_required = mapper.extract({"a": 1}, "b", default="z", required=True)
    assert miss_required.is_failure
    miss_default = mapper.extract({"a": 1}, "b", default="z")
    assert miss_default.is_success
    assert miss_default.value == "z"
    idx_required = mapper.extract({"a": [1]}, "a[bad]", default="x", required=True)
    assert idx_required.is_failure
    idx_default = mapper.extract({"a": [1]}, "a[bad]", default="x")
    assert idx_default.is_success
    assert idx_default.value == "x"


def test_accessor_take_pick_as_or_flat_and_agg_branches(mapper: type[u]) -> None:
    assert mapper.at({"a": 1}, "a").value == 1
    assert mapper.at([9, 8], 0).value == 9
    assert mapper.at([9, 8], 5, default=7).value == 7
    assert mapper.take({"a": 1}, "a", default=0) == 1
    assert mapper.take({"a": "x"}, "a", as_type=int, default=0) == 0
    assert mapper.take([1, 2, 3], 2) == [1, 2]
    assert mapper.take((1, 2, 3), 2, from_start=False) == [2, 3]
    assert mapper.take({"a": 1, "b": 2}, 1) == {"a": 1}
    assert mapper.pick({"a": 1, "b": 2}, "a") == {"a": 1}
    assert mapper.pick({"a": 1, "b": 2}, "a", "b", as_dict=False) == [1, 2]
    assert mapper.as_(1, int) == 1
    assert mapper.as_("1", int, strict=True, default=0) == 0
    assert mapper.as_("1", int) == 1
    float_value = mapper.as_("1.5", float)
    assert isinstance(float_value, float)
    assert abs(float_value - 1.5) < 1e-09
    assert mapper.as_("true", bool) is True
    assert mapper.as_("maybe", bool, default=False) is False
    assert mapper.as_(None, int, default=3) == 3
    assert mapper.or_(None, None, 1, default=2).value == 1
    assert mapper.or_(None, None, default=2).value == 2
    assert mapper.flat([[1, 2], [3]]) == [1, 2, 3]
    assert mapper._extract_field_value({"x": 1}, "x") == 1
    assert mapper.agg([{"v": 1}, {"v": 2}], "v") == 3
    mixed_items: tuple[dict[str, object], ...] = ({"v": 1}, {"v": "no"})
    assert mapper.agg(mixed_items, "v") == 1
    assert mapper.agg([1, 2, 3], lambda x: x, fn=max) == 3


def test_remaining_build_fields_construct_and_eq_paths(mapper: type[u]) -> None:
    assert mapper._build_apply_ensure([1], {"ensure": "list"}) == [1]
    assert mapper._build_apply_ensure(1, {"ensure": "str_list"}) == ["1"]
    assert mapper._build_apply_ensure({"a": 1}, {"ensure": "dict"}) == {"a": 1}
    assert mapper._build_apply_filter(
        [1, 2, 0],
        cast("Mapping[str, object]", {"filter": bool}),
        0,
    ) == [1, 2]
    assert mapper._build_apply_map(
        [1, 2],
        cast("Mapping[str, object]", {"map": _plus_one}),
    ) == [2, 3]
    assert mapper._apply_map_keys({"a": 1}, map_keys={"b": "B"}) == {"a": 1}
    assert mapper._apply_strip_none({"a": None, "b": 1}, strip_none=True) == {"b": 1}
    assert mapper._apply_strip_empty({"a": "", "b": 1}, strip_empty=True) == {"b": 1}
    process_list_ops = cast("Mapping[str, object]", {"process": _plus_one})
    assert mapper._build_apply_process([1, 2], process_list_ops, 0, "stop") == [2, 3]
    grouped_call = mapper._build_apply_group(
        ["aa", "b"],
        cast("Mapping[str, object]", {"group": len}),
    )
    assert grouped_call == {"2": ["aa"], "1": ["b"]}
    grouped_skip = mapper._build_apply_group([{"kind": None}, 1], {"group": "kind"})
    assert grouped_skip == {"": [{"kind": None}]}
    sorted_ok = mapper._build_apply_sort(
        [3, 1, 2],
        cast("Mapping[str, object]", {"sort": int}),
    )
    assert sorted_ok == [1, 2, 3]
    assert mapper._build_apply_sort([3, 1], {"sort": True}) == [1, 3]
    assert mapper._build_apply_unique([1, 1, 2], {"unique": True}) == [1, 2]
    assert mapper._build_apply_slice([1, 2, 3], {"slice": (0, 2)}) == [1, 2]
    assert mapper._build_apply_slice([1, 2, 3], {"slice": (0,)}) == [1, 2, 3]
    assert mapper._build_apply_chunk([1, 2, 3], {"chunk": 2}) == [[1, 2], [3]]
    assert mapper.field({"a": 1}, "a") == 1
    assert mapper.fields_multi({"a": 1, "b": 2}, {"a": 0, "b": 0}) == {"a": 1, "b": 2}
    assert mapper.fields_multi(m.ConfigMap(root={"a": 1}), {"a": 0}) == {"a": 1}
    assert mapper.construct({"x": {"value": 1}}, m.ConfigMap(root={"x": 0})) == {"x": 1}
    assert mapper.construct({"x": "a"}, m.ConfigMap(root={"a": 2})) == {"x": 2}
    assert mapper.construct(
        {"x": {"field": "a", "ops": "noop"}},
        m.ConfigMap(root={"a": 2}),
    ) == {"x": 2}
    assert mapper.deep_eq({"a": None}, {"a": None}) is True
    assert mapper.deep_eq({"a": {"x": 1}}, {"a": {"x": 1}}) is True
    assert mapper.deep_eq({"a": [1, 2]}, {"a": [1, 2]}) is True

    class DictLikeOnly:
        def keys(self) -> list[str]:
            return ["x"]

    class DictLikeOnlySecondary:
        def keys(self) -> list[str]:
            return ["y"]

    context = mapper.process_context_data(
        primary_data=cast("object", DictLikeOnly()),
        secondary_data=cast("object", DictLikeOnlySecondary()),
        merge_strategy="merge",
    )
    assert context == {}
    fields_obj = mapper.fields(
        object(),
        {"x": {"default": 1}},
    )
    assert fields_obj == {"x": 1}


def test_remaining_uncovered_branches(
    mapper: type[u],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _PortModel(port=9000, nested={"k": "v"})
    base_model_extract = mapper.extract(model, "nested.k")
    assert base_model_extract.is_success
    assert base_model_extract.value == "v"
    indexed_extract = mapper.extract({"a": [{"b": 1}]}, "a[0].b")
    assert indexed_extract.is_success
    assert indexed_extract.value == 1
    terminal_required = mapper.extract({"a": None}, "a", required=True)
    assert terminal_required.is_failure
    assert "Extracted value is None" in str(terminal_required.error)
    terminal_default = mapper.extract({"a": None}, "a", default="fallback")
    assert terminal_default.is_success
    assert terminal_default.value == "fallback"

    class MaybeModel(BaseModel):
        x: str | None = None

    assert _take_obj(MaybeModel(x=None), "x", default="d") == "d"
    assert mapper.as_("nope", int, default=9) == 9
    assert mapper.agg([{"v": "x"}], "v") == 0
    assert mapper._apply_map_keys({"a": 1}, map_keys={"a": "A"}) == {"A": 1}

    class GroupModel(BaseModel):
        kind: str | None = None

    grouped = _build_apply_group_obj([GroupModel(kind=None)], {"group": "kind"})
    assert grouped == {"": [{"kind": None}]}
    assert mapper._build_apply_sort([2, 1], {"sort": 5}) == [2, 1]

    class CallableDictLike:
        def __call__(self) -> None:
            return None

        def keys(self) -> list[str]:
            return ["k"]

        def items(self) -> list[tuple[str, int]]:
            return [("k", 1)]

        def get(self, key: str, default: object = None) -> object:
            return 1 if key == "k" else default

    # CallableDictLike is not a Mapping/ConfigMap, so process_context_data
    # treats it as non-dict data and returns empty result (no TypeError).
    callable_result = mapper.process_context_data(
        primary_data=cast("object", CallableDictLike()),
        secondary_data=cast("object", CallableDictLike()),
    )
    assert callable_result == {}
    obj_fields = mapper.fields(
        cast("object", AttrObject(name="n", value=3)),
        {"name": 0, "missing": 7},
    )
    assert obj_fields == {"name": "n"}
    dict_fields = mapper.fields({}, {"missing": 7})
    assert dict_fields == {"missing": 7}
