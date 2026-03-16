"""Tests for Enum utilities full coverage."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import cast, override

import pytest
from flext_tests import tm

from tests import u


class Status(StrEnum):
    """Test Status Enum."""

    ACTIVE = "active"
    PENDING = "pending"
    INACTIVE = "inactive"


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
    [(Status.ACTIVE, True), ("active", True), ("unknown", False), (123, False)],
)
def test_private_is_member_by_value(
    value: str | float | bool | Status,
    expected: bool,
) -> None:
    tm.that(u._is_member_by_value(value, Status), eq=expected)


def test_private_is_member_by_name() -> None:
    tm.that(u._is_member_by_name("ACTIVE", Status), eq=True)
    tm.that(u._is_member_by_name("MISSING", Status), eq=False)


def test_private_parse_success_and_failure() -> None:
    parsed_enum = u._parse(Status, Status.PENDING)
    tm.ok(parsed_enum)
    tm.that(parsed_enum.value, eq=Status.PENDING)
    parsed_value = u._parse(Status, "active")
    tm.ok(parsed_value)
    tm.that(parsed_value.value, eq=Status.ACTIVE)
    parsed_invalid = u._parse(Status, "invalid")
    tm.fail(parsed_invalid)
    tm.that(parsed_invalid.error, none=False)
    tm.fail(parsed_invalid, has="Invalid Status")
    tm.fail(parsed_invalid, has="active")
    tm.fail(parsed_invalid, has="pending")
    tm.fail(parsed_invalid, has="inactive")


def test_private_coerce_with_enum_and_string() -> None:
    tm.that(u._coerce(Status, Status.ACTIVE), eq=Status.ACTIVE)
    tm.that(u._coerce(Status, "pending"), eq=Status.PENDING)


def test_names_uses_cache_on_second_call() -> None:
    u._names_cache.clear()
    first = u.names(Status)
    second = u.names(Status)
    tm.that(first, eq=frozenset({"ACTIVE", "PENDING", "INACTIVE"}))
    tm.that(second, eq=first)


def test_members_uses_cache_on_second_call() -> None:
    u._members_cache.clear()
    first = u.members(Status)
    second = u.members(Status)
    tm.that(first, eq=frozenset({Status.ACTIVE, Status.PENDING, Status.INACTIVE}))
    tm.that(second, eq=first)


def test_get_enum_values_returns_immutable_sequence() -> None:
    tm.that(u.get_enum_values(Status), eq=("active", "pending", "inactive"))


def test_create_discriminated_union_multiple_enums() -> None:
    union_map = u.create_discriminated_union("kind", Status, Priority)
    tm.that(union_map["active"] is Status, eq=True)
    tm.that(union_map["pending"] is Status, eq=True)
    tm.that(union_map["inactive"] is Status, eq=True)
    tm.that(union_map["low"] is Priority, eq=True)
    tm.that(union_map["high"] is Priority, eq=True)


def test_auto_value_lowercases_input() -> None:
    tm.that(u.auto_value("MIXED_Name"), eq="mixed_name")


def test_bi_map_returns_forward_copy_and_inverse() -> None:
    source = {"one": "1", "two": "2"}
    forward, inverse = u.bi_map(source)
    tm.that(forward, eq=source)
    tm.that(forward is not source, eq=True)
    tm.that(inverse, eq={"1": "one", "2": "two"})


def test_create_enum_executes_factory_path() -> None:
    dynamic_status = u.create_enum("DynamicStatus", {"OK": "ok", "ERR": "err"})
    tm.that(dynamic_status.OK.value, eq="ok")
    tm.that(dynamic_status.ERR.value, eq="err")
    tm.that(dynamic_status.OK in dynamic_status.__members__.values(), eq=True)


def test_shortcuts_delegate_to_primary_methods() -> None:
    tm.that(u.is_member(Status, "active"), eq=True)
    parsed = u.parse("inactive", Status)
    tm.ok(parsed)
    tm.that(parsed.value, eq=Status.INACTIVE)


def test_dispatch_is_member_by_name_and_by_value() -> None:
    tm.that(u.dispatch("ACTIVE", Status, mode="is_member", by_name=True), eq=True)
    tm.that(u.dispatch(Status.PENDING, Status, mode="is_member"), eq=True)
    tm.that(u.dispatch("bad", Status, mode="is_member"), eq=False)


def test_dispatch_is_name_mode() -> None:
    tm.that(u.dispatch("ACTIVE", Status, mode="is_name"), eq=True)
    tm.that(u.dispatch("missing", Status, mode="is_name"), eq=False)


def test_dispatch_parse_mode_with_enum_string_and_other_object() -> None:
    parsed_from_enum = u.dispatch(Status.ACTIVE, Status, mode="parse")
    tm.ok(parsed_from_enum)
    tm.that(parsed_from_enum.value, eq=Status.ACTIVE)
    parsed_from_string = u.dispatch("pending", Status, mode="parse")
    tm.ok(parsed_from_string)
    tm.that(parsed_from_string.value, eq=Status.PENDING)
    parsed_from_other = u.dispatch(str(TextLike()), Status, mode="parse")
    tm.ok(parsed_from_other)
    tm.that(parsed_from_other.value, eq=Status.ACTIVE)


def test_dispatch_coerce_mode_with_enum_string_and_other_object() -> None:
    from_enum = u.dispatch(Status.INACTIVE, Status, mode="coerce")
    tm.that(from_enum, eq=Status.INACTIVE)
    from_string = u.dispatch("active", Status, mode="coerce")
    tm.that(from_string, eq=Status.ACTIVE)
    from_other = u.dispatch(str(TextLike()), Status, mode="coerce")
    tm.that(from_other, eq=Status.ACTIVE)


def test_dispatch_unknown_mode_raises() -> None:
    bad_mode = cast("str", "not-a-mode")
    dispatch_any = cast("Callable[..., object]", u.dispatch)
    with pytest.raises(ValueError, match="Unknown mode"):
        _ = dispatch_any("active", Status, mode=bad_mode)
