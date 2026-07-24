"""Behavioral tests for the public enum utilities surface (u.parse / u.enum_values).

Module: flext_core
Scope: public contract of ``u.enum_values`` and ``u.parse`` for StrEnum targets.

These tests assert only observable public behavior — return values, the ``r[T]``
outcome of the fallible ``parse`` operation, and the immutability/completeness
invariants of ``enum_values``. No private attributes, internal caches, or
collaborators are inspected.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import TYPE_CHECKING, Annotated, ClassVar

import pytest

from flext_tests import tm
from tests.models import m
from tests.utilities import u

if TYPE_CHECKING:
    from collections.abc import Sequence


class TestsFlextCoreEnumUtilities:
    """Behavioral contract of the public enum utilities."""

    @unique
    class Status(StrEnum):
        """Test StrEnum with lowercase values."""

        ACTIVE = "active"
        PENDING = "pending"
        INACTIVE = "inactive"

    @unique
    class Priority(StrEnum):
        """Independent StrEnum to prove per-type isolation."""

        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"

    class ParseScenario(m.BaseModel):
        """Declarative parse case for the public ``u.parse`` contract."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)
        name: Annotated[str, m.Field(description="Parse scenario name")]
        value: Annotated[str | StrEnum, m.Field(description="Input value to parse")]
        expected_success: Annotated[
            bool, m.Field(description="Whether parse should succeed")
        ]
        expected_status: Annotated[
            StrEnum | None, m.Field(description="Expected parsed enum member")
        ] = None
        expected_error: Annotated[
            str | None, m.Field(description="Expected error message fragment")
        ] = None

    PARSE: ClassVar[Sequence[ParseScenario]] = [
        ParseScenario(
            name="valid_string_resolves_to_member",
            value="active",
            expected_success=True,
            expected_status=Status.ACTIVE,
        ),
        ParseScenario(
            name="enum_instance_is_returned_unchanged",
            value=Status.PENDING,
            expected_success=True,
            expected_status=Status.PENDING,
        ),
        ParseScenario(
            name="unknown_string_fails",
            value="invalid",
            expected_success=False,
            expected_error="Cannot parse",
        ),
        ParseScenario(
            name="wrong_case_string_fails",
            value="ACTIVE",
            expected_success=False,
            expected_error="Cannot parse",
        ),
        ParseScenario(
            name="empty_string_fails",
            value="",
            expected_success=False,
            expected_error="Cannot parse",
        ),
    ]

    @pytest.mark.parametrize("scenario", PARSE, ids=lambda s: s.name)
    def test_parse_returns_expected_outcome(self, scenario: ParseScenario) -> None:
        """Parse yields success with the member, or a failure carrying the reason."""
        result = u.parse(scenario.value, self.Status)

        if scenario.expected_success:
            tm.ok(result)
            tm.that(result.value, eq=scenario.expected_status)
        else:
            tm.fail(result)
            tm.that(scenario.expected_error, none=False)
            tm.that(result.error, none=False)
            assert scenario.expected_error in result.error

    def test_parsed_member_round_trips_into_enum_values(self) -> None:
        """A successfully parsed member's value is present in enum_values."""
        result = u.parse("pending", self.Status)

        tm.ok(result)
        assert result.value.value in u.enum_values(self.Status)

    def test_enum_values_returns_complete_value_set(self) -> None:
        """enum_values exposes exactly the declared member string values."""
        values = u.enum_values(self.Status)

        tm.that(values, eq=frozenset({"active", "pending", "inactive"}))

    def test_enum_values_is_immutable_frozenset(self) -> None:
        """The returned value set is a frozenset and cannot be mutated."""
        values = u.enum_values(self.Status)

        assert isinstance(values, frozenset)
        assert not hasattr(values, "add")

    def test_enum_values_is_idempotent(self) -> None:
        """Repeated calls return equal value sets regardless of internal caching."""
        first = u.enum_values(self.Status)
        second = u.enum_values(self.Status)

        tm.that(first, eq=second)

    def test_enum_values_isolates_distinct_enum_types(self) -> None:
        """Different StrEnum types produce their own, non-overlapping value sets."""
        status_values = u.enum_values(self.Status)
        priority_values = u.enum_values(self.Priority)

        tm.that(priority_values, eq=frozenset({"low", "medium", "high"}))
        assert status_values.isdisjoint(priority_values)
