"""Tests for Enum utilities full coverage."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum, unique
from typing import cast, override

import pytest
from flext_tests import tm, u

from tests import t


class TestUtilitiesEnumFullCoverage:
    @unique
    class Status(StrEnum):
        """Test Status Enum."""

        ACTIVE = "active"
        PENDING = "pending"
        INACTIVE = "inactive"

    @unique
    class Priority(StrEnum):
        """Test Priority Enum."""

        LOW = "low"
        HIGH = "high"

    class TextLike:
        """Test string-like class implementation."""

        @override
        def __str__(self) -> str:
            return "active"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (Status.ACTIVE, True),
            ("active", True),
            ("unknown", False),
            (123, False),
        ],
    )
    def test_private_is_member_by_value(
        self,
        value: str | float | bool | Status,
        expected: bool,
    ) -> None:
        tm.that(u._is_member_by_value(value, self.Status), eq=expected)

    def test_private_is_member_by_name(self) -> None:
        tm.that(u._is_member_by_name("ACTIVE", self.Status), eq=True)
        tm.that(not u._is_member_by_name("MISSING", self.Status), eq=True)

    def test_private_parse_success_and_failure(self) -> None:
        parsed_enum = u._parse(self.Status, self.Status.PENDING)
        tm.ok(parsed_enum)
        tm.that(parsed_enum.value, eq=self.Status.PENDING)
        parsed_value = u._parse(self.Status, "active")
        tm.ok(parsed_value)
        tm.that(parsed_value.value, eq=self.Status.ACTIVE)
        parsed_invalid = u._parse(self.Status, "invalid")
        tm.fail(parsed_invalid)
        tm.that(parsed_invalid.error, none=False)
        tm.fail(parsed_invalid, has="Invalid Status")
        tm.fail(parsed_invalid, has="active")
        tm.fail(parsed_invalid, has="pending")
        tm.fail(parsed_invalid, has="inactive")

    def test_private_coerce_with_enum_and_string(self) -> None:
        tm.that(u._coerce(self.Status, self.Status.ACTIVE), eq=self.Status.ACTIVE)
        tm.that(u._coerce(self.Status, "pending"), eq=self.Status.PENDING)

    def test_names_uses_cache_on_second_call(self) -> None:
        u._names_cache.clear()
        first = u.names(self.Status)
        second = u.names(self.Status)
        assert first == frozenset({"ACTIVE", "PENDING", "INACTIVE"})
        assert second == first

    def test_members_uses_cache_on_second_call(self) -> None:
        u._members_cache.clear()
        first = u.members(self.Status)
        second = u.members(self.Status)
        assert first == frozenset({
            self.Status.ACTIVE,
            self.Status.PENDING,
            self.Status.INACTIVE,
        })
        assert second == first

    def test_get_enum_values_returns_immutable_sequence(self) -> None:
        tm.that(
            u.get_enum_values(self.Status),
            eq=("active", "pending", "inactive"),
        )

    def test_create_discriminated_union_multiple_enums(self) -> None:
        union_map = u.create_discriminated_union("kind", self.Status, self.Priority)
        tm.that(union_map["active"] is self.Status, eq=True)
        tm.that(union_map["pending"] is self.Status, eq=True)
        tm.that(union_map["inactive"] is self.Status, eq=True)
        tm.that(union_map["low"] is self.Priority, eq=True)
        tm.that(union_map["high"] is self.Priority, eq=True)

    def test_auto_value_lowercases_input(self) -> None:
        tm.that(u.auto_value("MIXED_Name"), eq="mixed_name")

    def test_bi_map_returns_forward_copy_and_inverse(self) -> None:
        source = {"one": "1", "two": "2"}
        forward, inverse = u.bi_map(source)
        tm.that(forward, eq=source)
        tm.that(forward is not source, eq=True)
        tm.that(inverse, eq={"1": "one", "2": "two"})

    def test_create_enum_executes_factory_path(self) -> None:
        dynamic_status = u.create_enum("DynamicStatus", {"OK": "ok", "ERR": "err"})
        ok_member = dynamic_status.__members__["OK"]
        err_member = dynamic_status.__members__["ERR"]
        tm.that(ok_member.value, eq="ok")
        tm.that(err_member.value, eq="err")
        tm.that(dynamic_status.__members__.values(), has=ok_member)

    def test_shortcuts_delegate_to_primary_methods(self) -> None:
        tm.that(u.is_member(self.Status, "active"), eq=True)
        parsed = u.parse("inactive", self.Status)
        tm.ok(parsed)
        tm.that(parsed.value, eq=self.Status.INACTIVE)

    def test_dispatch_is_member_by_name_and_by_value(self) -> None:
        tm.that(
            u.dispatch("ACTIVE", self.Status, mode="is_member", by_name=True),
            eq=True,
        )
        tm.that(u.dispatch(self.Status.PENDING, self.Status, mode="is_member"), eq=True)
        tm.that(not u.dispatch("bad", self.Status, mode="is_member"), eq=True)

    def test_dispatch_is_name_mode(self) -> None:
        tm.that(u.dispatch("ACTIVE", self.Status, mode="is_name"), eq=True)
        tm.that(not u.dispatch("missing", self.Status, mode="is_name"), eq=True)

    def test_dispatch_parse_mode_with_enum_string_and_other_object(self) -> None:
        parsed_from_enum = u.dispatch(self.Status.ACTIVE, self.Status, mode="parse")
        tm.ok(parsed_from_enum)
        tm.that(parsed_from_enum.value, eq=self.Status.ACTIVE)
        parsed_from_string = u.dispatch("pending", self.Status, mode="parse")
        tm.ok(parsed_from_string)
        tm.that(parsed_from_string.value, eq=self.Status.PENDING)
        parsed_from_other = u.dispatch(str(self.TextLike()), self.Status, mode="parse")
        tm.ok(parsed_from_other)
        tm.that(parsed_from_other.value, eq=self.Status.ACTIVE)

    def test_dispatch_coerce_mode_with_enum_string_and_other_object(self) -> None:
        from_enum = u.dispatch(self.Status.INACTIVE, self.Status, mode="coerce")
        tm.that(from_enum, eq=self.Status.INACTIVE)
        from_string = u.dispatch("active", self.Status, mode="coerce")
        tm.that(from_string, eq=self.Status.ACTIVE)
        from_other = u.dispatch(str(self.TextLike()), self.Status, mode="coerce")
        tm.that(from_other, eq=self.Status.ACTIVE)

    def test_dispatch_unknown_mode_raises(self) -> None:
        bad_mode = cast("str", "not-a-mode")
        dispatch_any = cast("Callable[..., t.NormalizedValue]", u.dispatch)
        with pytest.raises(ValueError, match="Unknown mode"):
            _ = dispatch_any("active", self.Status, mode=bad_mode)
