from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import Never, cast

import pytest

from flext_core import c, m, r, t, u


class _Color(StrEnum):
    RED = "red"
    BLUE = "blue"


class _BadMapping(Mapping[str, str]):
    def __iter__(self):
        raise RuntimeError("boom")

    def __len__(self) -> int:
        return 0

    def __getitem__(self, key: str) -> str:
        raise KeyError(key)


class _BadSequence:
    def __iter__(self):
        raise TypeError("iter failed")


class _BadCopyDict(dict[str, t.GeneralValueType]):
    def copy(self) -> dict[str, t.GeneralValueType]:
        raise TypeError("copy failed")


def test_find_mapping_no_match_and_merge_error_paths() -> None:
    assert c.Errors.UNKNOWN_ERROR
    assert isinstance(m.Categories(), m.Categories)
    assert r[int].ok(1).is_success
    assert isinstance(t.ConfigMap.model_validate({"a": 1}), t.ConfigMap)

    not_found = u.Collection.find({"a": 1}, lambda value: value == 2)
    assert not_found is None

    nested = u.Collection._merge_deep_single_key(
        {"x": _BadCopyDict({"a": 1})},
        "x",
        {"b": 2},
    )
    assert nested.is_failure

    deep = u.Collection.merge(
        {"x": _BadCopyDict({"a": 1})},
        {"x": {"b": 2}},
        strategy="deep",
    )
    assert deep.is_failure

    broken = u.Collection.merge(
        cast("dict[str, t.GeneralValueType]", None), {"x": 1}, strategy="deep"
    )
    assert broken.is_failure


def test_batch_fail_collect_flatten_and_progress() -> None:
    flattened = u.Collection.batch(
        [1],
        cast(
            "Callable[[int], r[Never]]",
            (lambda _item: cast("r[Never]", r[list[int]].ok([1, 2]))),
        ),
        flatten=True,
    )
    assert flattened.is_success
    flat_value = cast("dict[str, t.GeneralValueType]", flattened.value)
    assert flat_value["results"] == [1, 2]

    collected = u.Collection.batch(
        [1],
        cast(
            "Callable[[int], r[Never]]",
            (lambda _item: cast("r[Never]", r[int].fail("err"))),
        ),
        on_error="collect",
    )
    assert collected.is_success
    collected_value = cast("dict[str, t.GeneralValueType]", collected.value)
    errors = cast("list[tuple[int, str]]", collected_value["errors"])
    assert len(errors) == 1
    assert "err" in errors[0][1]

    failed = u.Collection.batch([1], lambda _item: r[int].fail("hard"), on_error="fail")
    assert failed.is_failure

    failed_exc = u.Collection.batch(
        [1], lambda _item: (_ for _ in ()).throw(ValueError("x"))
    )
    assert failed_exc.is_failure

    progress_calls: list[tuple[int, int]] = []
    ok = u.Collection.batch(
        [1, 2],
        lambda item: item,
        progress=lambda processed, total: progress_calls.append((processed, total)),
    )
    assert ok.is_success
    assert progress_calls[-1] == (2, 2)


def test_process_outer_exception_and_coercion_branches() -> None:
    processed: r[list[object]] = u.Collection.process(
        cast("list[object]", _BadSequence()),
        lambda x: x,
    )
    assert processed.is_failure

    assert u.Collection._coerce_value_to_float(1.5) == 1.5
    assert u.Collection._coerce_value_to_bool(True) is True

    enum_dict = u.Collection.coerce_dict_to_enum(_Color)({"a": _Color.RED})
    assert enum_dict["a"] is _Color.RED

    enum_list = u.Collection.coerce_list_to_enum(_Color)([_Color.BLUE])
    assert enum_list == [_Color.BLUE]

    assert u.Collection.first([], default=9) == 9
    assert u.Collection.last([], default=8) == 8


def test_parse_mapping_outer_exception() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        u.Collection.parse_mapping(
            _Color,
            cast("dict[str, str | _Color]", _BadMapping()),
        )


def test_collection_batch_failure_error_capture_and_parse_sequence_outer_error() -> (
    None
):
    class _FailureResult:
        is_success = False
        value = None
        error = "boom"

    collected = u.Collection.batch(
        [1],
        lambda _item: _FailureResult(),
        on_error="collect",
    )
    assert collected.is_success
    collected_value = cast("dict[str, t.GeneralValueType]", collected.value)
    errors = cast("list[tuple[int, str]]", collected_value["errors"])
    assert errors[0][1] == "boom"

    failed = u.Collection.batch([1], lambda _item: _FailureResult(), on_error="fail")
    assert failed.is_failure

    class _ExplodingMeta(type):
        def __call__(cls, _value: object):
            raise ValueError("parse exploded")

    class _ExplodingEnum(metaclass=_ExplodingMeta):
        pass

    parsed = u.Collection.parse_sequence(cast("type[_Color]", _ExplodingEnum), ["x"])
    assert parsed.is_failure
