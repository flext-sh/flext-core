"""Tests for Collection utilities mapping and edge cases."""

from __future__ import annotations

from collections import UserDict, UserList
from collections.abc import Iterator, Mapping
from enum import StrEnum
from typing import NoReturn, cast, override

import pytest

from flext_core import c, m, r, u
from flext_tests import t


class _Color(StrEnum):
    RED = "red"
    BLUE = "blue"


class _BadMapping(Mapping[str, str]):
    @override
    def __iter__(self) -> Iterator[str]:
        msg = "boom"
        raise RuntimeError(msg)

    @override
    def __len__(self) -> int:
        return 0

    @override
    def __getitem__(self, key: str) -> str:
        raise KeyError(key)


class _BadSequence:
    def __iter__(self) -> Iterator[str]:
        msg = "iter failed"
        raise TypeError(msg)


class _BadCopyDict(UserDict[str, object]):
    @override
    def copy(self) -> _BadCopyDict:
        msg = "copy failed"
        raise TypeError(msg)


class _ListSubclass(UserList[int]):
    pass


def test_find_mapping_no_match_and_merge_error_paths() -> None:
    assert c.Errors.UNKNOWN_ERROR
    assert isinstance(m.Categories(categories={}), m.Categories)
    assert r[int].ok(1).is_success
    assert isinstance(m.ConfigMap({"a": 1}), m.ConfigMap)
    not_found = u.find({"a": 1}, lambda value: value == 2)
    assert not_found.is_failure
    nested = u._merge_deep_single_key(
        cast("dict[str, t.NormalizedValue]", {"x": _BadCopyDict({"a": 1})}),
        "x",
        cast("t.NormalizedValue", {"b": 2}),
    )
    assert nested.is_failure
    deep = u.merge(
        cast("dict[str, t.NormalizedValue]", {"x": _BadCopyDict({"a": 1})}),
        cast("dict[str, t.NormalizedValue]", {"x": {"b": 2}}),
        strategy="deep",
    )
    assert deep.is_failure
    broken = u.merge(
        cast("dict[str, t.NormalizedValue]", None),
        {"x": 1},
        strategy="deep",
    )
    assert broken.is_failure


def test_batch_fail_collect_flatten_and_progress() -> None:
    def _success_list(_item: int) -> list[int]:
        return [1, 2]

    def _failure_result(_item: int) -> NoReturn:
        msg = "err"
        raise ValueError(msg)

    def _hard_failure(_item: int) -> NoReturn:
        msg = "hard"
        raise ValueError(msg)

    def _raise_value_error(_item: int) -> NoReturn:
        msg = "x"
        raise ValueError(msg)

    def _identity(item: int) -> int:
        return item

    flattened = u.batch(
        [1],
        _success_list,
        flatten=True,
    )
    assert flattened.is_success
    flat_value = flattened.value
    assert flat_value.results == [1, 2]
    collected = u.batch(
        [1],
        _failure_result,
        on_error="collect",
    )
    assert collected.is_success
    collected_value = collected.value
    assert len(collected_value.errors) == 1
    assert "err" in collected_value.errors[0][1]
    failed = u.batch([1], _hard_failure, on_error="fail")
    assert failed.is_failure
    failed_exc = u.batch(
        [1],
        _raise_value_error,
    )
    assert failed_exc.is_failure
    progress_calls: list[tuple[int, int]] = []
    ok = u.batch(
        [1, 2],
        _identity,
        progress=lambda processed, total: progress_calls.append((processed, total)),
    )
    assert ok.is_success
    assert progress_calls[-1] == (2, 2)


def test_process_outer_exception_and_coercion_branches() -> None:
    with pytest.raises(TypeError, match="iter failed"):
        _ = u.process(
            cast("list[str]", _BadSequence()),
            lambda x: x,
        )
    value = u._coerce_value_to_float(1.5)
    assert abs(value - 1.5) < 1e-9
    assert u._coerce_value_to_bool(True) is True
    enum_dict = u.coerce_dict_to_enum(_Color)({"a": _Color.RED})
    assert enum_dict["a"] is _Color.RED
    enum_list = u.coerce_list_to_enum(_Color)([_Color.BLUE])
    assert enum_list == [_Color.BLUE]
    assert u.first([], default=9).value == 9
    assert u.last([], default=8).value == 8


def test_parse_mapping_outer_exception() -> None:
    result = u.parse_mapping(
        _Color,
        cast("dict[str, str | _Color]", _BadMapping()),
    )
    assert result.is_failure
    assert result.error is not None and "Parse mapping failed" in result.error


def test_collection_batch_failure_error_capture_and_parse_sequence_outer_error() -> (
    None
):

    class _FailureResult:
        is_success = False
        value: None = None
        error = "boom"

    collected = u.batch(
        [1],
        lambda _item: _FailureResult(),
        on_error="collect",
    )
    assert collected.is_success
    collected_value = collected.value
    assert collected_value.errors == []
    assert (
        collected_value.results[0]
        == "<tests.unit.test_utilities_collection_full_coverage.test_collection_batch_failure_error_capture_and_parse_sequence_outer_error.<locals>._FailureResult object"
    )
    failed = u.batch([1], lambda _item: _FailureResult(), on_error="fail")
    assert failed.is_success

    class _ExplodingMeta(type):
        def __call__(cls, _value: str) -> NoReturn:
            msg = "parse exploded"
            raise ValueError(msg)

    class _ExplodingEnum(metaclass=_ExplodingMeta):
        pass

    parsed = u.parse_sequence(cast("type[_Color]", _ExplodingEnum), ["x"])
    assert parsed.is_failure


def test_is_general_value_list_accepts_list_subclass() -> None:
    value = _ListSubclass([1, 2, 3])

    assert u._is_general_value_list(cast("t.NormalizedValue", value)) is False
