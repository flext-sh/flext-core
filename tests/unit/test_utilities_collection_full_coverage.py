"""Tests for Collection utilities mapping and edge cases."""

from __future__ import annotations

from collections import UserDict, UserList
from collections.abc import Iterator, Mapping, MutableSequence, Sequence
from enum import StrEnum, unique
from typing import NoReturn, cast, override

import pytest
from flext_tests import tm

from flext_core import r
from tests import c, m, t, u


class TestUtilitiesCollectionFullCoverage:
    @unique
    class _Color(StrEnum):
        RED = "red"
        BLUE = "blue"

    class _BadMapping(Mapping[str, str]):
        @override
        def __getitem__(self, _key: str) -> str:
            msg = "mapping get failed"
            raise TypeError(msg)

        @override
        def __iter__(self) -> Iterator[str]:
            msg = "mapping iter failed"
            raise TypeError(msg)

        @override
        def __len__(self) -> int:
            return 1

    class _BadSequence:
        def __iter__(self) -> Iterator[str]:
            msg = "iter failed"
            raise TypeError(msg)

    class _BadCopyDict(UserDict[str, t.NormalizedValue]):
        @override
        def copy(self) -> TestUtilitiesCollectionFullCoverage._BadCopyDict:
            msg = "copy failed"
            raise TypeError(msg)

    class _ListSubclass(UserList[int]):
        pass

    class _FailureResult:
        is_success = False
        value: None = None
        error = "boom"

    class _ExplodingMeta(type):
        def __call__(cls, _value: str) -> NoReturn:
            msg = "parse exploded"
            raise ValueError(msg)

    class _ExplodingEnum(metaclass=_ExplodingMeta):
        pass

    def test_find_mapping_no_match_and_merge_error_paths(self) -> None:
        assert c.UNKNOWN_ERROR
        assert isinstance(m.Categories(categories={}), m.Categories)
        assert r[int].ok(1).is_success
        assert isinstance(t.ConfigMap({"a": 1}), t.ConfigMap)
        not_found = u.find({"a": 1}, lambda value: value == 2)
        tm.fail(not_found)
        nested = u._merge_deep_single_key(
            cast(
                "t.MutableContainerMapping",
                {"x": self._BadCopyDict({"a": 1})},
            ),
            "x",
            cast("t.NormalizedValue", {"b": 2}),
        )
        tm.ok(nested)
        deep = u.merge_mappings(
            cast("t.ContainerMapping", {"x": self._BadCopyDict({"a": 1})}),
            cast("t.ContainerMapping", {"x": {"b": 2}}),
            strategy="deep",
        )
        tm.ok(deep)
        with pytest.raises(TypeError, match="iterable"):
            _ = u.merge_mappings(
                cast("t.ContainerMapping", None), {"x": 1}, strategy="deep"
            )

    def test_batch_fail_collect_flatten_and_progress(self) -> None:
        def _success_list(_item: int) -> Sequence[int]:
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
        tm.ok(flattened)
        flat_value = flattened.value
        assert flat_value.results == [1, 2]
        collected = u.batch(
            [1],
            _failure_result,
            on_error="collect",
        )
        tm.ok(collected)
        collected_value = collected.value
        tm.that(len(collected_value.errors), eq=1)
        tm.that(collected_value.errors[0][1], has="err")
        failed = u.batch([1], _hard_failure, on_error="fail")
        tm.fail(failed)
        failed_exc = u.batch(
            [1],
            _raise_value_error,
        )
        tm.fail(failed_exc)
        progress_calls: MutableSequence[tuple[int, int]] = []
        ok = u.batch(
            [1, 2],
            _identity,
            progress=lambda processed, total: progress_calls.append((processed, total)),
        )
        tm.ok(ok)
        assert progress_calls[-1] == (2, 2)

    def test_process_outer_exception_and_coercion_branches(self) -> None:
        with pytest.raises(TypeError, match="iter failed"):
            _ = u.process(
                cast("t.StrSequence", self._BadSequence()),
                lambda x: x,
            )
        value = u._coerce_value_to_float(1.5)
        tm.that(abs(value - 1.5), lt=1e-9)
        assert u._coerce_value_to_bool(True) is True
        enum_dict = u.coerce_dict_to_enum(self._Color)({"a": self._Color.RED})
        assert enum_dict["a"] is self._Color.RED
        enum_list = u.coerce_list_to_enum(self._Color)([self._Color.BLUE])
        tm.that(enum_list, eq=[self._Color.BLUE])
        assert u.first([], default=9).value == 9
        assert u.last([], default=8).value == 8

    def test_parse_mapping_outer_exception(self) -> None:
        result = u.parse_mapping(
            self._Color,
            cast(
                "Mapping[str, str | TestUtilitiesCollectionFullCoverage._Color]",
                self._BadMapping(),
            ),
        )
        tm.fail(result)
        assert result.error is not None and "Parse mapping failed" in result.error

    def test_collection_batch_failure_error_capture_and_parse_sequence_outer_error(
        self,
    ) -> None:
        collected = u.batch(
            [1],
            lambda _item: str(self._FailureResult()),
            on_error="collect",
        )
        tm.ok(collected)
        collected_value = collected.value
        assert collected_value.errors == []
        tm.that(str(collected_value.results[0]), has="_FailureResult")
        failed = u.batch([1], lambda _item: str(self._FailureResult()), on_error="fail")
        tm.ok(failed)
        parsed = u.parse_sequence(
            cast(
                "type[TestUtilitiesCollectionFullCoverage._Color]", self._ExplodingEnum
            ),
            ["x"],
        )
        tm.fail(parsed)

    def test_is_general_value_list_accepts_list_subclass(self) -> None:
        value = self._ListSubclass([1, 2, 3])
        assert u._is_general_value_list(cast("t.NormalizedValue", value)) is False
