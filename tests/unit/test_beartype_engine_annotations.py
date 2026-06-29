"""Beartype annotation inspection tests."""

from __future__ import annotations

import typing
from collections.abc import Mapping

from flext_core._utilities.beartype_engine import FlextUtilitiesBeartypeEngine as be
from tests import t
from tests.unit._beartype_engine_support import (
    AnyAlias,
    CleanAlias,
    NestedAnyAlias,
    TestsFlextBeartypeEngine,
)


class TestsFlextBeartypeEngineAnnotations(TestsFlextBeartypeEngine):
    def test_direct_any(self) -> None:
        """typing.Any is detected."""
        assert be.contains_any(typing.Any) is True

    def test_str_clean(self) -> None:
        """Plain str has no Any."""
        assert be.contains_any(str) is False

    def test_int_clean(self) -> None:
        """Plain int has no Any."""
        assert be.contains_any(int) is False

    def test_mapping_with_any_value(self) -> None:
        """Mapping[str, Any] detected at 1 level."""
        assert be.contains_any(Mapping[str, typing.Any]) is True

    def test_deep_nested_any(self) -> None:
        """Mapping[str, t.SequenceOf[Any]] detected at 2 levels."""
        assert be.contains_any(Mapping[str, t.SequenceOf[typing.Any]]) is True

    def test_clean_mapping(self) -> None:
        """t.IntMapping has no Any."""
        assert be.contains_any(t.IntMapping) is False

    def test_clean_sequence(self) -> None:
        """t.StrSequence has no Any."""
        assert be.contains_any(t.StrSequence) is False

    def test_union_with_any(self) -> None:
        """Str | Any detected."""
        assert be.contains_any(str | typing.Any) is True

    def test_optional_any(self) -> None:
        """Any | None detected."""
        assert be.contains_any(typing.Any | None) is True

    def test_clean_optional(self) -> None:
        """Str | None has no Any."""
        assert be.contains_any(str | None) is False

    def test_complex_clean_type(self) -> None:
        """Mapping[str, t.SequenceOf[int]] has no Any."""
        assert be.contains_any(Mapping[str, t.SequenceOf[int]]) is False

    def test_bare_dict(self) -> None:
        """dict[str, int] is forbidden."""
        result = be.has_forbidden_collection_origin(dict[str, int], self.FORBIDDEN)
        assert result == (True, "dict")

    def test_bare_list(self) -> None:
        """list[str] is forbidden."""
        result = be.has_forbidden_collection_origin(list[str], self.FORBIDDEN)
        assert result == (True, "list")

    def test_bare_set(self) -> None:
        """set[int] is forbidden."""
        result = be.has_forbidden_collection_origin(set[int], self.FORBIDDEN)
        assert result == (True, "set")

    def test_mapping_ok(self) -> None:
        """t.IntMapping is not forbidden."""
        result = be.has_forbidden_collection_origin(t.IntMapping, self.FORBIDDEN)
        assert result == (False, "")

    def test_sequence_ok(self) -> None:
        """t.StrSequence is not forbidden."""
        result = be.has_forbidden_collection_origin(t.StrSequence, self.FORBIDDEN)
        assert result == (False, "")

    def test_plain_str_ok(self) -> None:
        """Plain str has no origin."""
        result = be.has_forbidden_collection_origin(str, self.FORBIDDEN)
        assert result == (False, "")

    def test_simple_union(self) -> None:
        """Str | int has 2 members."""
        assert be.count_union_members(str | int) == 2

    def test_optional(self) -> None:
        """Str | None has 1 non-None member."""
        assert be.count_union_members(str | None) == 1

    def test_complex_union(self) -> None:
        """Str | int | float | None has 3 non-None members."""
        assert be.count_union_members(str | int | float | None) == 3

    def test_non_union(self) -> None:
        """Plain str returns 0."""
        assert be.count_union_members(str) == 0

    def test_triple_union(self) -> None:
        """Str | int | float has 3 members."""
        assert be.count_union_members(str | int | float) == 3

    def test_str_none(self) -> None:
        """Str | None detected."""
        assert be.matches_str_none_union(str | None) is True

    def test_plain_str(self) -> None:
        """Plain str is not str | None."""
        assert be.matches_str_none_union(str) is False

    def test_int_none(self) -> None:
        """Int | None is NOT str | None."""
        assert be.matches_str_none_union(int | None) is False

    def test_str_int(self) -> None:
        """Str | int is NOT str | None."""
        assert be.matches_str_none_union(str | int) is False

    def test_str_int_none(self) -> None:
        """Str | int | None IS str | None (str and None both present)."""
        assert be.matches_str_none_union(str | int | None) is True

    def test_alias_with_any(self) -> None:
        """Type alias containing Any is detected."""
        assert be.alias_contains_any(AnyAlias.__value__) is True

    def test_clean_alias(self) -> None:
        """Type alias without Any passes."""
        assert be.alias_contains_any(CleanAlias.__value__) is False

    def test_nested_any_in_alias(self) -> None:
        """Nested Any in alias is detected."""
        assert be.alias_contains_any(NestedAnyAlias.__value__) is True
