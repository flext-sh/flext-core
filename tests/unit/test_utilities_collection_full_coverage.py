"""Tests for Collection utilities mapping and edge cases."""

from __future__ import annotations

from collections import UserDict, UserList
from collections.abc import Iterator
from enum import StrEnum, unique
from typing import NoReturn, cast, override

import pytest

from flext_core import r
from flext_tests import tm
from tests import c, m, t, u


class TestUtilitiesCollectionFullCoverage:
    @unique
    class _Color(StrEnum):
        RED = "red"
        BLUE = "blue"

    class _BadMapping(t.StrMapping):
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
                cast("t.ContainerMapping", None),
                {"x": 1},
                strategy="deep",
            )

    def test_process_outer_exception_and_coercion_branches(self) -> None:
        with pytest.raises(TypeError, match="iter failed"):
            _ = u.process(
                cast("t.StrSequence", self._BadSequence()),
                lambda x: x,
            )

    def test_parse_sequence_outer_error(self) -> None:
        parsed = u.parse_sequence(
            cast(
                "type[TestUtilitiesCollectionFullCoverage._Color]",
                self._ExplodingEnum,
            ),
            ["x"],
        )
        tm.fail(parsed)
