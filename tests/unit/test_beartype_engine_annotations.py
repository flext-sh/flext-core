"""Behavioral tests for FlextUtilitiesBeartypeEngine annotation inspection.

These assert the OBSERVABLE CONTRACT of the public static methods only:
their return values for representative type hints. No private attribute or
method of the engine is accessed; the engine is exercised exclusively through
its public surface (``contains_any``, ``has_forbidden_collection_origin``,
``count_union_members``, ``matches_str_none_union``, ``alias_contains_any``).
"""

from __future__ import annotations

import typing
from collections.abc import Mapping

import pytest

from flext_core.utilities import FlextUtilitiesBeartypeEngine as be
from tests.typings import t
from tests.unit._beartype_engine_support import (
    AnyAlias,
    CleanAlias,
    NestedAnyAlias,
    TestsFlextBeartypeEngine,
)


class TestsFlextBeartypeEngineAnnotations(TestsFlextBeartypeEngine):
    """Contract of the beartype annotation-inspection engine."""

    @pytest.mark.parametrize(
        ("hint", "expected"),
        [
            (typing.Any, True),
            (Mapping[str, typing.Any], True),
            (Mapping[str, t.SequenceOf[typing.Any]], True),
            (str | typing.Any, True),
            (typing.Any | None, True),
            (str, False),
            (int, False),
            (str | None, False),
            (t.IntMapping, False),
            (t.StrSequence, False),
            (Mapping[str, t.SequenceOf[int]], False),
            (None, False),
        ],
    )
    def test_contains_any_reports_presence_of_any(
        self, hint: t.TypeHintSpecifier | None, *, expected: bool
    ) -> None:
        """contains_any is True iff typing.Any appears at any nesting depth."""
        assert be.contains_any(hint) is expected

    @pytest.mark.parametrize(
        ("hint", "expected"),
        [
            (dict[str, int], (True, "dict")),
            (list[str], (True, "list")),
            (set[int], (True, "set")),
            (t.IntMapping, (False, "")),
            (t.StrSequence, (False, "")),
            (str, (False, "")),
            (None, (False, "")),
        ],
    )
    def test_has_forbidden_collection_origin_names_offending_origin(
        self, hint: t.TypeHintSpecifier | None, expected: tuple[bool, str]
    ) -> None:
        """Bare mutable collection origins are flagged with their name."""
        assert be.has_forbidden_collection_origin(hint, self.FORBIDDEN) == expected

    @pytest.mark.parametrize(
        ("hint", "expected"),
        [
            (str | int, 2),
            (str | int | float, 3),
            (str | None, 1),
            (str | int | float | None, 3),
            (str, 0),
            (None, 0),
        ],
    )
    def test_count_union_members_excludes_none(
        self, hint: t.TypeHintSpecifier | None, *, expected: int
    ) -> None:
        """count_union_members counts non-None members; 0 for non-unions."""
        assert be.count_union_members(hint) == expected

    @pytest.mark.parametrize(
        ("hint", "expected"),
        [
            (str | None, True),
            (str | int | None, True),
            (str, False),
            (int | None, False),
            (str | int, False),
            (None, False),
        ],
    )
    def test_matches_str_none_union_requires_str_and_none(
        self, hint: t.TypeHintSpecifier | None, *, expected: bool
    ) -> None:
        """Union matches iff both str and None are members."""
        assert be.matches_str_none_union(hint) is expected

    @pytest.mark.parametrize(
        ("alias_value", "expected"),
        [
            (AnyAlias.__value__, True),
            (NestedAnyAlias.__value__, True),
            (CleanAlias.__value__, False),
            (None, False),
        ],
    )
    def test_alias_contains_any_unwraps_alias_values(
        self, alias_value: t.TypeHintSpecifier | None, *, expected: bool
    ) -> None:
        """alias_contains_any detects Any inside a resolved type-alias value."""
        assert be.alias_contains_any(alias_value) is expected
