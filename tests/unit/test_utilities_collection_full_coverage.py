"""Tests for Collection utilities mapping and edge cases."""

from __future__ import annotations

from collections import UserDict, UserList
from collections.abc import Iterator, Mapping, Sequence
from enum import StrEnum, unique
from typing import NoReturn, overload, override

import pytest

from flext_tests import r, tm
from tests import c, t, u


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

    class _BadSequence(Sequence[str]):
        @override
        def __iter__(self) -> Iterator[str]:
            msg = "iter failed"
            raise TypeError(msg)

        @overload
        def __getitem__(self, index: int) -> str: ...

        @overload
        def __getitem__(self, index: slice) -> Sequence[str]: ...

        @override
        def __getitem__(self, index: int | slice) -> str | Sequence[str]:
            _ = index
            msg = "iter failed"
            raise TypeError(msg)

        @override
        def __len__(self) -> int:
            return 1

    class _BadCopyDict(UserDict[str, t.RecursiveContainer]):
        @override
        def copy(self) -> TestUtilitiesCollectionFullCoverage._BadCopyDict:
            msg = "copy failed"
            raise TypeError(msg)

    class _ListSubclass(UserList[int]):
        pass

    class _FailureResult:
        success = False
        value: None = None
        error = "boom"

    class _ExplodingMeta(type):
        def __call__(cls, _value: str) -> NoReturn:
            msg = "parse exploded"
            raise ValueError(msg)

    class _ExplodingEnum(metaclass=_ExplodingMeta):
        pass

    def test_find_mapping_no_match_and_merge_error_paths(self) -> None:
        assert c.ErrorCode.UNKNOWN_ERROR
        assert r[int].ok(1).success
        assert isinstance(t.ConfigMap({"a": 1}), t.ConfigMap)
        not_found = u.find({"a": 1}, lambda value: value == 2)
        tm.fail(not_found)
        target: t.MutableRecursiveContainerMapping = {"x": self._BadCopyDict({"a": 1})}
        nested = u._merge_deep_single_key(target, "x", {"b": 2})
        tm.ok(nested)
        deep = u.merge_mappings(
            {"x": self._BadCopyDict({"a": 1})},
            {"x": {"b": 2}},
            strategy="deep",
        )
        tm.ok(deep)
        with pytest.raises(TypeError, match="iterable"):
            _ = u.merge_mappings(
                None,
                {"x": 1},
                strategy="deep",
            )

    def test_process_outer_exception_and_coercion_branches(self) -> None:
        bad_seq: t.StrSequence = self._BadSequence()
        with pytest.raises(TypeError, match="iter failed"):
            _ = u.process(bad_seq, lambda x: x)
