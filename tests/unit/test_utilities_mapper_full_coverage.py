from __future__ import annotations

# mypy: follow_imports=skip
# mypy: disable-error-code=valid-type
# mypy: disable-error-code=misc

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from pydantic import BaseModel

from flext_core.constants import c
from flext_core._utilities.cache import FlextUtilitiesCache as Cache
from flext_core._utilities.mapper import FlextUtilitiesMapper as Mapper
from flext_core.typings import t


@dataclass
class AttrObject:
    name: str = "name"
    value: int = 1


class SampleModel(BaseModel):
    port: int = c.Platform.DEFAULT_HTTP_PORT
    nested: dict[str, t.GeneralValueType] = {"k": "v"}


class BadString:
    def __str__(self) -> str:
        msg = "cannot stringify"
        raise ValueError(msg)


class BadBool:
    def __bool__(self) -> bool:
        msg = "cannot bool"
        raise ValueError(msg)


def _parse_int(value: object) -> int:
    return int(cast("str", value))


def _plus_one(value: t.ConfigMapValue) -> t.ConfigMapValue:
    return cast("int", value) + 1


def _times_two(value: t.ConfigMapValue) -> t.ConfigMapValue:
    return cast("int", value) * 2


def _identity(value: t.ConfigMapValue) -> t.ConfigMapValue:
    return value


class ExplodingLenList(list[object]):
    def __len__(self) -> int:
        msg = "len exploded"
        raise TypeError(msg)


class BadMapping(Mapping[str, t.GeneralValueType]):
    def __getitem__(self, key: str) -> t.GeneralValueType:
        msg = f"missing {key}"
        raise KeyError(msg)

    def __iter__(self) -> Iterator[str]:
        msg = "iter exploded"
        raise RuntimeError(msg)

    def __len__(self) -> int:
        return 1


@pytest.fixture
def mapper() -> type[Mapper]:
    return Mapper


def test_type_guards_and_narrowing_failures(mapper: type[Mapper]) -> None:
    assert mapper._is_configuration_dict([1]) is False
    assert mapper._is_configuration_dict({1: "x"}) is True
    assert mapper._is_configuration_mapping([1]) is False
    assert mapper._is_configuration_mapping({1: "x"}) is True

    with pytest.raises(TypeError, match="Cannot narrow"):
        mapper._narrow_to_configuration_dict(10)

    with pytest.raises(TypeError, match="Cannot narrow"):
        mapper._narrow_to_sequence("not-sequence")


def test_narrow_to_string_keyed_dict_and_mapping_paths(mapper: type[Mapper]) -> None:
    converted = mapper._narrow_to_string_keyed_dict(
        cast("t.ConfigMapValue", {1: "x", "b": object()})
    )
    assert "1" in converted
    assert isinstance(converted["b"], str)

    with pytest.raises(TypeError, match="Cannot narrow"):
        mapper._narrow_to_string_keyed_dict(123)

    mapped = mapper._narrow_to_configuration_mapping({"x": 1})
    assert isinstance(mapped, t.ConfigMap)
    assert mapped.root["x"] == 1

    with pytest.raises(TypeError, match="Cannot coerce"):
        _ = mapper._narrow_to_configuration_mapping(
            cast("t.ConfigMapValue", {1: BadString()})
        )

    with pytest.raises(TypeError, match="Cannot narrow"):
        mapper._narrow_to_configuration_mapping(3)


def test_general_value_helpers_and_logger(mapper: type[Mapper]) -> None:
    class Stable:
        def __str__(self) -> str:
            return "stable"

    assert (
        mapper.narrow_to_general_value_type(
            cast("t.ConfigMapValue", cast(object, Stable()))
        )
        == "stable"
    )
    assert mapper._get_str_from_dict({"k": 2}, "k", default="") == "2"
    assert mapper._get_str_from_dict({"k": None}, "k", default="d") == "d"
    assert mapper._get_callable_from_dict({"x": 1}, "x") is None
    assert Mapper().logger is not None


def test_invert_and_json_conversion_branches(mapper: type[Mapper]) -> None:
    assert mapper.invert_dict({"a": "x", "b": "x"}, handle_collisions="first") == {
        "x": "a",
    }

    assert mapper.convert_to_json_value(None) is None

    class Model(BaseModel):
        x: int

    model = Model(x=1)
    assert mapper.convert_to_json_value(model) == model

    unknown = mapper.convert_to_json_value(Path("/tmp"))
    assert unknown == Path("/tmp")

    as_json = mapper.convert_dict_to_json({"x": Path("/tmp")})
    assert as_json["x"] == Path("/tmp")

    list_json = mapper.convert_list_to_json(
        cast("Sequence[t.ConfigMapValue]", [{"a": 1}, {"b": object()}])
    )
    assert isinstance(list_json, list)
    assert list_json[0]["a"] == 1


def test_ensure_and_extract_array_index_helpers(mapper: type[Mapper]) -> None:
    assert mapper.ensure(123) == ["123"]

    value, error = mapper._extract_handle_array_index("x", "0")
    assert value is None
    assert error == "Not a sequence"

    value, error = mapper._extract_handle_array_index([1, 2], "-1")
    assert value == 2
    assert error is None

    value, error = mapper._extract_handle_array_index([1, 2], "bad")
    assert value is None
    assert "Invalid index" in str(error)


def test_extract_error_paths_and_prop_accessor(mapper: type[Mapper]) -> None:
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
        def __str__(self) -> str:
            return "converted"

    class Container:
        field = NotGeneral()

    res_non_general = mapper.extract(
        cast("t.ConfigMap | BaseModel", cast(object, Container())), "field"
    )
    assert res_non_general.is_success
    assert res_non_general.value == "converted"

    class ExplodingModelDump:
        def model_dump(self) -> dict[str, t.GeneralValueType]:
            msg = "boom"
            raise ValueError(msg)

    res_exception = mapper.extract(
        cast("t.ConfigMap | BaseModel", cast(object, ExplodingModelDump())),
        "a",
    )
    assert res_exception.is_failure
    assert "Extract failed" in str(res_exception.error)

    accessor = mapper.prop("name")
    assert (
        accessor(
            cast("t.ConfigMap | BaseModel", cast(object, AttrObject(name="x", value=1)))
        )
        == "x"
    )
    assert (
        mapper.prop("missing")(cast("t.ConfigMap | BaseModel", cast(object, {"a": 1})))
        == ""
    )


def test_at_take_and_as_branches(mapper: type[Mapper]) -> None:
    assert mapper.at({"a": 1}, 0, default=5) == 5
    assert mapper.at(ExplodingLenList([1]), 0, default=7) == 7

    model = SampleModel(port=8081)
    assert mapper.take(model, "port") == 8081
    assert mapper.take(123, "port", default="d") == "d"
    assert mapper.take({"port": None}, "port", default="x") == "x"
    assert (
        mapper.take(cast("Mapping[str, t.ConfigMapValue]", cast(object, 123)), 2)
        is None
    )

    assert mapper.as_(12, str) == "12"
    assert mapper.as_("off", bool) is False


def test_extract_field_value_and_ensure_variants(mapper: type[Mapper]) -> None:
    assert mapper._extract_field_value(AttrObject(name="a", value=2), "value") == 2
    assert mapper._extract_field_value(AttrObject(), "missing") is None

    assert mapper._build_apply_ensure(5, {"ensure": "str"}) == "5"
    assert mapper._build_apply_ensure(5, {"ensure": "list"}) == [5]
    assert mapper._build_apply_ensure([1, "a"], {"ensure": "str_list"}) == ["1", "a"]
    assert mapper._build_apply_ensure(5, {"ensure": "dict"}) == {}
    assert mapper._build_apply_ensure(5, {"ensure": "unknown"}) == 5


def test_filter_map_normalize_convert_helpers(mapper: type[Mapper]) -> None:
    plus_one = cast("Callable[[t.ConfigMapValue], t.ConfigMapValue]", _plus_one)
    times_two = cast("Callable[[t.ConfigMapValue], t.ConfigMapValue]", _times_two)
    assert mapper._build_apply_filter(1, {"filter": 1}, 0) == 1
    assert mapper._build_apply_filter(
        {"a": 1, "b": 0},
        cast("Mapping[str, t.ConfigMapValue]", {"filter": bool}),
        0,
    ) == {
        "a": 1,
    }
    assert (
        mapper._build_apply_filter(
            0,
            cast("Mapping[str, t.ConfigMapValue]", {"filter": bool}),
            "d",
        )
        == "d"
    )

    assert mapper._build_apply_map(1, {"map": 1}) == 1
    assert mapper._build_apply_map(
        {"a": 1},
        cast("Mapping[str, t.ConfigMapValue]", {"map": plus_one}),
    ) == {"a": 2}
    assert (
        mapper._build_apply_map(
            2,
            cast("Mapping[str, t.ConfigMapValue]", {"map": times_two}),
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
    ("convert_spec", "value", "expected"),
    [
        (int, "bad", 0),
        (float, "bad", 0.0),
        (str, BadString(), ""),
        (bool, BadBool(), False),
        (list, 1, []),
        (dict, 1, {}),
        (tuple, 1, ()),
        (set, 1, []),
        (lambda _x: (_ for _ in ()).throw(ValueError("x")), 1, 1),
    ],
)
def test_convert_default_fallback_matrix(
    mapper: type[Mapper],
    convert_spec: Callable[[object], object] | type,
    value: t.ConfigMapValue,
    expected: t.ConfigMapValue,
) -> None:
    result = mapper._build_apply_convert(
        value,
        cast("Mapping[str, t.ConfigMapValue]", {"convert": convert_spec}),
    )
    assert result == expected


def test_convert_sequence_branch_returns_tuple(mapper: type[Mapper]) -> None:
    converted = mapper._build_apply_convert(
        ("bad",),
        cast("Mapping[str, t.ConfigMapValue]", {"convert": int}),
    )
    assert converted == (0,)


def test_transform_option_extract_and_step_helpers(
    mapper: type[Mapper],
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
    extracted = mapper._extract_transform_options(
        cast("Mapping[str, t.ConfigMapValue]", opts)
    )
    assert extracted[3] == {"1": "one", "a": "b"}

    monkeypatch.setattr(
        Cache,
        "normalize_component",
        staticmethod(lambda _x: "not-a-dict"),
    )
    assert mapper._apply_normalize({"a": 1}, normalize=True) == {"a": 1}

    assert mapper._apply_map_keys({"a": 1}, map_keys={"a": "A"}) == {"A": 1}
    assert mapper._apply_filter_keys({"a": 1, "b": 2}, filter_keys={"b"}) == {"b": 2}
    assert mapper._apply_exclude_keys({"a": 1, "b": 2}, exclude_keys={"a"}) == {
        "b": 2,
    }
    assert mapper._apply_strip_none({"a": None}, strip_none=False) == {"a": None}
    assert mapper._apply_strip_empty({"a": ""}, strip_empty=False) == {"a": ""}
    assert mapper._apply_to_json({"a": Path("/tmp")}, to_json=True)["a"] == Path("/tmp")


def test_build_apply_transform_and_process_error_paths(
    mapper: type[Mapper],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert mapper._build_apply_transform({"a": 1}, {"transform": 1}, {}, "stop") == {
        "a": 1,
    }

    def explode_transform_steps(
        _result: dict[str, t.GeneralValueType],
        *,
        normalize: bool,
        map_keys: Mapping[str, str] | None,
        filter_keys: set[str] | None,
        exclude_keys: set[str] | None,
        strip_none: bool,
        strip_empty: bool,
        to_json: bool,
    ) -> dict[str, t.GeneralValueType]:
        _ = (
            normalize,
            map_keys,
            filter_keys,
            exclude_keys,
            strip_none,
            strip_empty,
            to_json,
        )
        raise RuntimeError(msg)

    msg = "explode transform"
    monkeypatch.setattr(
        Mapper,
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
    process_map_ops = cast(
        "Mapping[str, t.ConfigMapValue]",
        {"process": lambda x: x + 1},
    )
    assert mapper._build_apply_process({"a": 1}, process_map_ops, 0, "stop") == {
        "a": 2,
    }
    process_fail_ops = cast(
        "Mapping[str, t.ConfigMapValue]",
        {"process": lambda _x: (_ for _ in ()).throw(ValueError("x"))},
    )
    assert mapper._build_apply_process(1, process_fail_ops, 7, "stop") == 7
    assert mapper._build_apply_process(1, process_fail_ops, 7, "skip") == 1


def test_group_sort_unique_slice_chunk_branches(mapper: type[Mapper]) -> None:
    assert mapper._build_apply_group(1, {"group": "k"}) == 1
    grouped = mapper._build_apply_group(
        [{"kind": "a", "v": 1}, {"kind": "a", "v": 2}],
        {"group": "kind"},
    )
    assert grouped == {"a": [{"kind": "a", "v": 1}, {"kind": "a", "v": 2}]}
    assert mapper._build_apply_group([1, 2], {"group": 5}) == [1, 2]

    assert mapper._build_apply_sort(1, {"sort": True}) == 1
    sorted_with_scalar = mapper._build_apply_sort(
        [{"name": "b"}, 3, {"name": "a"}], {"sort": "name"}
    )
    assert isinstance(sorted_with_scalar, list)

    bad_sort_ops = cast(
        "Mapping[str, t.ConfigMapValue]",
        {"sort": lambda _x: (_ for _ in ()).throw(ValueError("x"))},
    )
    bad_sort = mapper._build_apply_sort([1, 2], bad_sort_ops)
    assert bad_sort == [1, 2]

    sorted_tuple = mapper._build_apply_sort(("b", "a"), {"sort": True})
    assert sorted_tuple == ("a", "b")

    assert mapper._build_apply_unique(1, {"unique": True}) == 1
    assert mapper._build_apply_unique((1, 2, 1), {"unique": True}) == (1, 2)

    assert mapper._build_apply_slice(1, {"slice": (0, 1)}) == 1
    assert mapper._build_apply_slice((1, 2, 3), {"slice": (1, 3)}) == (2, 3)

    assert mapper._build_apply_chunk(1, {"chunk": 2}) == 1
    assert mapper._build_apply_chunk([1, 2], {"chunk": 0}) == [1, 2]
    assert mapper.build([1, 2], ops=None) == [1, 2]


def test_field_and_fields_multi_branches(mapper: type[Mapper]) -> None:
    assert (
        mapper.field(
            cast("t.ConfigMap | BaseModel", cast(object, object())),
            "missing",
            required=True,
        )
        is None
    )
    assert mapper.field({}, "missing", ops={"ensure": "str"}) == ""

    source_obj = AttrObject(name="n", value=1)
    spec_stop = {"must": None}
    res_stop = mapper._fields_multi(
        cast("t.ConfigMap | BaseModel", cast(object, source_obj)),
        spec_stop,
        on_error="stop",
    )
    assert res_stop.is_failure  # type: ignore[union-attr]

    spec_collect = {"must": None, "name": ""}
    res_collect = mapper._fields_multi(
        cast("t.ConfigMap | BaseModel", cast(object, source_obj)),
        spec_collect,
        on_error="collect",
    )
    assert res_collect.is_failure  # type: ignore[union-attr]
    assert "Field extraction errors" in str(res_collect.error)  # type: ignore[union-attr]

    spec_skip = {"must": None, "name": ""}
    res_skip = mapper._fields_multi(
        cast("t.ConfigMap | BaseModel", cast(object, source_obj)),
        spec_skip,
        on_error="skip",
    )
    assert res_skip == {"name": "n"}

    spec_ops_not_dict = {"x": {"ops": "bad"}}
    res_ops = mapper._fields_multi({"x": 2}, spec_ops_not_dict, on_error="skip")  # type: ignore[arg-type]
    assert isinstance(res_ops, dict)


def test_construct_transform_and_deep_eq_branches(
    mapper: type[Mapper],
) -> None:
    constructed_none = mapper.construct({"x": {"field": "a", "default": 9}}, None)
    assert constructed_none["x"] == 9

    source: dict[str, t.ConfigMapValue] = {"name": "alice", "n": 3}
    spec = cast(
        "Mapping[str, t.ConfigMapValue]",
        {
            "name": {"field": "name", "ops": "skip-ops"},
            "n": {"field": "n", "ops": {"map": lambda x: x + 1}},
            "literal": 5,
        },
    )
    constructed = mapper.construct(spec, source)  # type: ignore[arg-type]
    assert constructed["name"] == "alice"
    assert constructed["n"] == 4
    assert constructed["literal"] == 5

    class ExplodeOnGet(Mapping[str, t.GeneralValueType]):
        def __iter__(self) -> Iterator[str]:
            return iter(("field",))

        def __len__(self) -> int:
            return 1

        def __getitem__(self, key: str) -> t.GeneralValueType:
            if key == "field":
                msg = "boom"
                raise RuntimeError(msg)
            return ""

    assert mapper.construct({"x": ExplodeOnGet()}, {"x": 1}, on_error="stop") == {  # type: ignore[arg-type]
        "x": "",
    }
    assert mapper.construct({"x": ExplodeOnGet()}, {"x": 1}, on_error="skip") == {  # type: ignore[arg-type]
        "x": "",
    }

    assert mapper.transform({"a": 1}, map_keys={"a": "A"}).is_success
    with pytest.raises(RuntimeError, match="iter exploded"):
        mapper.transform(BadMapping())

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
    mapper: type[Mapper],
    merge_strategy: str,
) -> None:
    primary: dict[str, t.ConfigMapValue] = {"a": 1, "drop": "x"}
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
        context={"a": "1"},  # type: ignore[arg-type]
        extra_kwargs={"b": 2},  # type: ignore[arg-type]
        field="x",
    )
    assert normalized["field"] == "x"


def test_small_mapper_convenience_methods(mapper: type[Mapper]) -> None:
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
        cast("t.ConfigMapValue", cast(object, AttrObject(name="obj", value=4))),
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
        def __call__(self, value: int) -> bool:
            _ = value
            raise ValueError("x")

    class NegativePredicate(NamedPredicate):
        def __call__(self, value: int) -> bool:
            return value < 0

    class EqualOnePredicate(NamedPredicate):
        def __call__(self, value: int) -> bool:
            return value == 1

    predicates: dict[str, Callable[[int], bool]] = {
        "bad": BadPredicate(),
        "no": NegativePredicate(),
        "yes": EqualOnePredicate(),
    }
    assert mapper.find_callable(predicates, 1) == "yes"  # type: ignore[arg-type]
    assert mapper.find_callable({"no": lambda value: value < 0}, 1) is None


def test_map_flags_collect_and_invert_branches(mapper: type[Mapper]) -> None:
    mapped = mapper.map_dict_keys(
        {"old": 1, "x": 2}, {"old": "new"}, keep_unmapped=True
    )
    assert mapped.is_success
    assert mapped.value == {"new": 1, "x": 2}

    class BadItems(dict[str, t.GeneralValueType]):
        def items(self):
            raise RuntimeError("bad items")

    fail_map = mapper.map_dict_keys(BadItems(), {})
    assert fail_map.is_failure

    flags = mapper.build_flags_dict(["read"], {"read": "can_read", "w": "can_write"})
    assert flags.is_success
    assert flags.value == {"can_read": True, "can_write": False}

    class BadIter(list[str]):
        def __iter__(self):
            raise RuntimeError("bad iter")

    fail_flags = mapper.build_flags_dict(BadIter(), {})
    assert fail_flags.is_failure

    active = mapper.collect_active_keys({"r": True, "w": False}, {"r": "R", "w": "W"})
    assert active.is_success
    assert active.value == ["R"]

    class BadGet(dict[str, bool]):
        def get(self, key: str, default: object = None):
            raise RuntimeError("bad get")

    fail_active = mapper.collect_active_keys(BadGet(), {"x": "X"})
    assert fail_active.is_failure

    assert mapper.invert_dict({"a": "x", "b": "x"}, handle_collisions="last") == {
        "x": "b",
    }


def test_conversion_and_extract_success_branches(mapper: type[Mapper]) -> None:
    class Plain:
        def __str__(self) -> str:
            return "plain"

    assert (
        mapper.convert_to_json_value(cast("t.ConfigMapValue", cast(object, Plain())))
        == "plain"
    )
    assert mapper.convert_to_json_value(
        cast("t.ConfigMapValue", cast(object, {1: Plain()}))
    ) == {"1": "plain"}
    assert mapper.convert_to_json_value(
        cast("t.ConfigMapValue", cast(object, [1, {"k": Plain()}]))
    ) == [1, {"k": "plain"}]

    assert mapper.ensure_str(None, "d") == "d"
    assert mapper.ensure_str("x") == "x"
    assert mapper.ensure_str(2) == "2"

    assert mapper.ensure(None) == []
    assert mapper.ensure("x") == ["x"]
    assert mapper.ensure([1, 2]) == ["1", "2"]
    assert mapper.ensure_str_or_none("x") == "x"

    class DumpOnly:
        def model_dump(self) -> dict[str, t.GeneralValueType]:
            return {"a": 1}

    value, found = mapper._extract_get_value(
        cast("t.ConfigMapValue | BaseModel", cast(object, DumpOnly())), "a"
    )
    assert found is True
    assert value == 1

    value, found = mapper._extract_get_value(
        cast("t.ConfigMapValue | BaseModel", cast(object, DumpOnly())), "missing"
    )
    assert found is False
    assert value is None

    value, err = mapper._extract_handle_array_index([1], "3")
    assert value is None
    assert err == "Index 3 out of range"

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


def test_accessor_take_pick_as_or_flat_and_agg_branches(mapper: type[Mapper]) -> None:
    assert mapper.at({"a": 1}, "a") == 1
    assert mapper.at([9, 8], 0) == 9
    assert mapper.at([9, 8], 5, default=7) == 7

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
    assert mapper.as_("1.5", float) == 1.5
    assert mapper.as_("true", bool) is True
    assert mapper.as_("maybe", bool, default=False) is False
    assert mapper.as_(None, int, default=3) == 3

    assert mapper.or_(None, None, 1, default=2) == 1
    assert mapper.or_(None, None, default=2) == 2
    assert mapper.flat([[1, 2], [3]]) == [1, 2, 3]

    assert mapper._extract_field_value({"x": 1}, "x") == 1
    assert mapper.agg([{"v": 1}, {"v": 2}], "v") == 3
    assert mapper.agg(({"v": 1}, {"v": "no"}), "v") == 1
    assert mapper.agg([1, 2, 3], lambda x: x, fn=max) == 3


def test_remaining_build_fields_construct_and_eq_paths(mapper: type[Mapper]) -> None:
    assert mapper._build_apply_ensure([1], {"ensure": "list"}) == [1]
    assert mapper._build_apply_ensure(1, {"ensure": "str_list"}) == ["1"]
    assert mapper._build_apply_ensure({"a": 1}, {"ensure": "dict"}) == {"a": 1}

    assert mapper._build_apply_filter(
        [1, 2, 0],
        cast("Mapping[str, t.ConfigMapValue]", {"filter": bool}),
        0,
    ) == [1, 2]
    assert mapper._build_apply_map(
        [1, 2],
        cast("Mapping[str, t.ConfigMapValue]", {"map": lambda x: x + 1}),
    ) == [2, 3]
    assert mapper._apply_map_keys({"a": 1}, map_keys={"b": "B"}) == {"a": 1}
    assert mapper._apply_strip_none({"a": None, "b": 1}, strip_none=True) == {"b": 1}
    assert mapper._apply_strip_empty({"a": "", "b": 1}, strip_empty=True) == {"b": 1}
    process_list_ops = cast(
        "Mapping[str, t.ConfigMapValue]",
        {"process": lambda x: x + 1},
    )
    assert mapper._build_apply_process([1, 2], process_list_ops, 0, "stop") == [2, 3]

    grouped_call = mapper._build_apply_group(
        ["aa", "b"],
        cast("Mapping[str, t.ConfigMapValue]", {"group": len}),
    )
    assert grouped_call == {"2": ["aa"], "1": ["b"]}
    grouped_skip = mapper._build_apply_group([{"kind": None}, 1], {"group": "kind"})
    assert grouped_skip == {"": [{"kind": None}]}

    sorted_ok = mapper._build_apply_sort(
        [3, 1, 2],
        cast("Mapping[str, t.ConfigMapValue]", {"sort": lambda x: x}),
    )
    assert sorted_ok == [1, 2, 3]
    assert mapper._build_apply_sort([3, 1], {"sort": True}) == [1, 3]
    assert mapper._build_apply_unique([1, 1, 2], {"unique": True}) == [1, 2]
    assert mapper._build_apply_slice([1, 2, 3], {"slice": (0, 2)}) == [1, 2]
    assert mapper._build_apply_slice([1, 2, 3], {"slice": (0,)}) == [1, 2, 3]
    assert mapper._build_apply_chunk([1, 2, 3], {"chunk": 2}) == [[1, 2], [3]]

    assert mapper.field({"a": 1}, "a") == 1
    assert mapper.fields_multi({"a": 1, "b": 2}, {"a": 0, "b": 0}) == {"a": 1, "b": 2}
    spec_with_ops = cast(
        "Mapping[str, Mapping[str, t.ConfigMapValue]]",
        {"a": {"default": 0, "ops": {"map": lambda x: x + 1}}},
    )
    assert mapper._fields_multi(
        {"a": 1},  # type: ignore[arg-type]
        spec_with_ops,
        on_error="skip",
    ) == {"a": 2}

    assert mapper.construct({"x": {"value": 1}}, {"x": 0}) == {"x": 1}  # type: ignore[arg-type]
    assert mapper.construct({"x": "a"}, {"a": 2}) == {"x": 2}  # type: ignore[arg-type]
    assert mapper.construct(
        {"x": {"field": "a", "ops": "noop"}},
        {"a": 2},  # type: ignore[arg-type]
    ) == {"x": 2}
    assert mapper.to_dict({"a": 1}) == {"a": 1}

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
        primary_data=cast("t.ConfigMapValue", cast(object, DictLikeOnly())),
        secondary_data=cast("t.ConfigMapValue", cast(object, DictLikeOnlySecondary())),
        merge_strategy="merge",
    )
    assert context == {}

    fields_obj = mapper.fields(
        cast("t.ConfigMapValue", object()),
        {"x": {"default": 1}},
    )
    assert fields_obj == {"x": 1}


def test_remaining_uncovered_branches(
    mapper: type[Mapper],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = SampleModel(port=9000, nested={"k": "v"})
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

    class HasNone:
        x = None

    class MaybeModel(BaseModel):
        x: str | None = None

    assert mapper.take(MaybeModel(x=None), "x", default="d") == "d"
    assert mapper.as_("nope", int, default=9) == 9
    assert mapper.agg([{"v": "x"}], "v") == 0

    assert mapper._apply_map_keys({"a": 1}, map_keys={"a": "A"}) == {"A": 1}

    class GroupModel(BaseModel):
        kind: str | None = None

    grouped = mapper._build_apply_group([GroupModel(kind=None)], {"group": "kind"})
    assert grouped == {}
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

    processed = mapper.process_context_data(
        primary_data=cast("t.ConfigMapValue", cast(object, CallableDictLike())),
        secondary_data=cast("t.ConfigMapValue", cast(object, CallableDictLike())),
    )
    assert processed == {}

    obj_fields = mapper.fields(
        cast("t.ConfigMapValue", cast(object, AttrObject(name="n", value=3))),
        {"name": 0, "missing": 7},
    )
    assert obj_fields == {"name": "n"}

    dict_fields = mapper.fields({}, {"missing": 7})
    assert dict_fields == {"missing": 7}
