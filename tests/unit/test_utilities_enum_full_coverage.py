from __future__ import annotations
# pyright: reportMissingImports=false, reportPrivateUsage=false, reportUnknownMemberType=false, reportUntypedFunctionDecorator=false, reportUnusedCallResult=false, reportUnknownVariableType=false, reportCallIssue=false, reportArgumentType=false

from enum import StrEnum
from collections.abc import Callable
from typing import cast, override

import pytest

from flext_core import u


class Status(StrEnum):
    ACTIVE = "active"
    PENDING = "pending"
    INACTIVE = "inactive"


class Priority(StrEnum):
    LOW = "low"
    HIGH = "high"


class TextLike:
    @override
    def __str__(self) -> str:
        return "active"


def test_check_direct_access_warns_from_non_approved_module() -> None:
    mod_globals: dict[str, object] = {
        "__name__": "external.module",
        "target": u.Enum._check_direct_access,
    }
    exec(
        "def outer():\n    inner()\n\ndef inner():\n    target()\n",
        mod_globals,
    )
    outer = mod_globals["outer"]
    assert callable(outer)

    with pytest.warns(DeprecationWarning, match="Direct import from _utilities.enum"):
        outer()


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
    value: str | int | float | bool | Status | None,
    expected: bool,
) -> None:
    assert u.Enum._is_member_by_value(value, Status) is expected


def test_private_is_member_by_name() -> None:
    assert u.Enum._is_member_by_name("ACTIVE", Status) is True
    assert u.Enum._is_member_by_name("MISSING", Status) is False


def test_private_parse_success_and_failure() -> None:
    parsed_enum = u.Enum._parse(Status, Status.PENDING)
    assert parsed_enum.is_success
    assert parsed_enum.value == Status.PENDING

    parsed_value = u.Enum._parse(Status, "active")
    assert parsed_value.is_success
    assert parsed_value.value == Status.ACTIVE

    parsed_invalid = u.Enum._parse(Status, "invalid")
    assert parsed_invalid.is_failure
    assert parsed_invalid.error is not None
    assert "Invalid Status" in parsed_invalid.error
    assert "active" in parsed_invalid.error
    assert "pending" in parsed_invalid.error
    assert "inactive" in parsed_invalid.error


def test_private_coerce_with_enum_and_string() -> None:
    assert u.Enum._coerce(Status, Status.ACTIVE) == Status.ACTIVE
    assert u.Enum._coerce(Status, "pending") == Status.PENDING


def test_names_uses_cache_on_second_call() -> None:
    u.Enum._names_cache.clear()
    first = u.Enum.names(Status)
    second = u.Enum.names(Status)

    assert first == frozenset({"ACTIVE", "PENDING", "INACTIVE"})
    assert second == first


def test_members_uses_cache_on_second_call() -> None:
    u.Enum._members_cache.clear()
    first = u.Enum.members(Status)
    second = u.Enum.members(Status)

    assert first == frozenset({Status.ACTIVE, Status.PENDING, Status.INACTIVE})
    assert second == first


def test_get_enum_values_returns_immutable_sequence() -> None:
    assert u.Enum.get_enum_values(Status) == ("active", "pending", "inactive")


def test_create_discriminated_union_multiple_enums() -> None:
    union_map = u.Enum.create_discriminated_union("kind", Status, Priority)

    assert union_map["active"] is Status
    assert union_map["pending"] is Status
    assert union_map["inactive"] is Status
    assert union_map["low"] is Priority
    assert union_map["high"] is Priority


def test_auto_value_lowercases_input() -> None:
    assert u.Enum.auto_value("MIXED_Name") == "mixed_name"


def test_bi_map_returns_forward_copy_and_inverse() -> None:
    source = {"one": "1", "two": "2"}
    forward, inverse = u.Enum.bi_map(source)

    assert forward == source
    assert forward is not source
    assert inverse == {"1": "one", "2": "two"}


def test_create_enum_executes_factory_path() -> None:
    expected_errors = cast("tuple[type[Exception], ...]", (TypeError, AttributeError))
    with pytest.raises(expected_errors):
        _ = u.Enum.create_enum("DynamicStatus", {"OK": "ok", "ERR": "err"})


def test_shortcuts_delegate_to_primary_methods() -> None:
    assert u.Enum.is_enum_member("active", Status) is True

    parsed = u.Enum.parse_enum(Status, "inactive")
    assert parsed.is_success
    assert parsed.value == Status.INACTIVE


def test_dispatch_is_member_by_name_and_by_value() -> None:
    assert u.Enum.dispatch("ACTIVE", Status, mode="is_member", by_name=True) is True
    assert u.Enum.dispatch(Status.PENDING, Status, mode="is_member") is True
    assert u.Enum.dispatch("bad", Status, mode="is_member") is False


def test_dispatch_is_name_mode() -> None:
    assert u.Enum.dispatch("ACTIVE", Status, mode="is_name") is True
    assert u.Enum.dispatch("missing", Status, mode="is_name") is False


def test_dispatch_parse_mode_with_enum_string_and_other_object() -> None:
    parsed_from_enum = u.Enum.dispatch(Status.ACTIVE, Status, mode="parse")
    assert parsed_from_enum.is_success
    assert parsed_from_enum.value == Status.ACTIVE

    parsed_from_string = u.Enum.dispatch("pending", Status, mode="parse")
    assert parsed_from_string.is_success
    assert parsed_from_string.value == Status.PENDING

    parsed_from_other = u.Enum.dispatch(str(TextLike()), Status, mode="parse")
    assert parsed_from_other.is_success
    assert parsed_from_other.value == Status.ACTIVE


def test_dispatch_coerce_mode_with_enum_string_and_other_object() -> None:
    from_enum = u.Enum.dispatch(Status.INACTIVE, Status, mode="coerce")
    assert from_enum == Status.INACTIVE

    from_string = u.Enum.dispatch("active", Status, mode="coerce")
    assert from_string == Status.ACTIVE

    from_other = u.Enum.dispatch(str(TextLike()), Status, mode="coerce")
    assert from_other == Status.ACTIVE


def test_dispatch_unknown_mode_raises() -> None:
    bad_mode = cast(str, "not-a-mode")
    dispatch_any = cast("Callable[..., object]", u.Enum.dispatch)
    with pytest.raises(ValueError, match="Unknown mode"):
        _ = dispatch_any("active", Status, mode=bad_mode)
