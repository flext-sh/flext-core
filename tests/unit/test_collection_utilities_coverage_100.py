"""Real tests to achieve 100% collection utilities coverage - no mocks.

Module: flext_core._utilities.collection
Scope: FlextUtilitiesCollection - all methods and edge cases

This module provides comprehensive real tests (no mocks, patches, or bypasses)
to cover all remaining lines in _utilities/collection.py.

Uses Python 3.13 patterns, u, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum, unique
from typing import Annotated, ClassVar

import pytest
from flext_tests import tm, u
from pydantic import BaseModel, ConfigDict, Field

from tests import t


class TestCollectionUtilitiesCoverage:
    @unique
    class Status(StrEnum):
        """Status enumeration."""

        ACTIVE = "active"
        PENDING = "pending"
        INACTIVE = "inactive"

    @unique
    class Priority(StrEnum):
        """Priority enumeration."""

        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"

    class ParseSequenceScenario(BaseModel):
        """Scenario for sequence parsing."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
        name: Annotated[str, Field(description="Parse sequence scenario name")]
        values: Annotated[
            Sequence[str | TestCollectionUtilitiesCoverage.Status],
            Field(description="Input sequence values"),
        ]
        expected_success: Annotated[
            bool,
            Field(description="Whether parsing should succeed"),
        ]
        expected_count: Annotated[
            int | None,
            Field(default=None, description="Expected parsed item count"),
        ] = None
        expected_error: Annotated[
            str | None,
            Field(default=None, description="Expected error message fragment"),
        ] = None

    class CoerceListValidatorScenario(BaseModel):
        """Scenario for list coercion."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
        name: Annotated[str, Field(description="Coerce list scenario name")]
        value: Annotated[
            t.NormalizedValue,
            Field(description="Input value for list coercion"),
        ]
        expected_success: Annotated[
            bool,
            Field(description="Whether coercion should succeed"),
        ]
        expected_error: Annotated[
            str | None,
            Field(default=None, description="Expected error message fragment"),
        ] = None

    class ParseMappingScenario(BaseModel):
        """Scenario for mapping parsing."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
        name: Annotated[str, Field(description="Parse mapping scenario name")]
        mapping: Annotated[
            Mapping[str, str | TestCollectionUtilitiesCoverage.Status],
            Field(description="Input mapping values"),
        ]
        expected_success: Annotated[
            bool,
            Field(description="Whether parsing should succeed"),
        ]
        expected_count: Annotated[
            int | None,
            Field(default=None, description="Expected parsed entry count"),
        ] = None
        expected_error: Annotated[
            str | None,
            Field(default=None, description="Expected error message fragment"),
        ] = None

    def _parse_sequence_scenarios(
        self,
    ) -> Sequence[TestCollectionUtilitiesCoverage.ParseSequenceScenario]:
        return [
            self.ParseSequenceScenario(
                name="valid_strings",
                values=["active", "pending"],
                expected_success=True,
                expected_count=2,
            ),
            self.ParseSequenceScenario(
                name="valid_enums",
                values=[self.Status.ACTIVE, self.Status.PENDING],
                expected_success=True,
                expected_count=2,
            ),
            self.ParseSequenceScenario(
                name="mixed_strings_and_enums",
                values=["active", self.Status.PENDING],
                expected_success=True,
                expected_count=2,
            ),
            self.ParseSequenceScenario(
                name="invalid_string",
                values=["invalid"],
                expected_success=False,
                expected_error="Invalid Status values",
            ),
            self.ParseSequenceScenario(
                name="multiple_invalid",
                values=["active", "invalid1", "invalid2"],
                expected_success=False,
                expected_error="Invalid Status values",
            ),
            self.ParseSequenceScenario(
                name="empty_sequence",
                values=[],
                expected_success=True,
                expected_count=0,
            ),
        ]

    def _coerce_list_validator_scenarios(
        self,
    ) -> Sequence[TestCollectionUtilitiesCoverage.CoerceListValidatorScenario]:
        return [
            self.CoerceListValidatorScenario(
                name="valid_list_strings",
                value=["active", "pending"],
                expected_success=True,
            ),
            self.CoerceListValidatorScenario(
                name="valid_tuple_strings",
                value=("active", "pending"),
                expected_success=True,
            ),
            self.CoerceListValidatorScenario(
                name="valid_set_strings",
                value=list({"active", "pending"}),
                expected_success=True,
            ),
            self.CoerceListValidatorScenario(
                name="valid_frozenset_strings",
                value=list(frozenset({"active", "pending"})),
                expected_success=True,
            ),
            self.CoerceListValidatorScenario(
                name="valid_list_enums",
                value=[self.Status.ACTIVE, self.Status.PENDING],
                expected_success=True,
            ),
            self.CoerceListValidatorScenario(
                name="invalid_not_sequence",
                value="not a sequence",
                expected_success=False,
                expected_error="Expected sequence",
            ),
            self.CoerceListValidatorScenario(
                name="invalid_string_in_list",
                value=["active", "invalid"],
                expected_success=False,
                expected_error="Invalid Status",
            ),
            self.CoerceListValidatorScenario(
                name="invalid_type_in_list",
                value=["active", 123],
                expected_success=False,
                expected_error="Expected str",
            ),
        ]

    def _parse_mapping_scenarios(
        self,
    ) -> Sequence[TestCollectionUtilitiesCoverage.ParseMappingScenario]:
        return [
            self.ParseMappingScenario(
                name="valid_strings",
                mapping={"user1": "active", "user2": "pending"},
                expected_success=True,
                expected_count=2,
            ),
            self.ParseMappingScenario(
                name="valid_enums",
                mapping={"user1": self.Status.ACTIVE, "user2": self.Status.PENDING},
                expected_success=True,
                expected_count=2,
            ),
            self.ParseMappingScenario(
                name="mixed_strings_and_enums",
                mapping={"user1": "active", "user2": self.Status.PENDING},
                expected_success=True,
                expected_count=2,
            ),
            self.ParseMappingScenario(
                name="invalid_string",
                mapping={"user1": "invalid"},
                expected_success=False,
                expected_error="Invalid Status values",
            ),
            self.ParseMappingScenario(
                name="empty_mapping",
                mapping={},
                expected_success=True,
                expected_count=0,
            ),
        ]

    def test_parse_sequence(self) -> None:
        for scenario in self._parse_sequence_scenarios():
            result = u.parse_sequence(self.Status, scenario.values)
            if scenario.expected_success:
                _ = u.Tests.Result.assert_success(result)
                parsed = result.value
                tm.that(len(parsed), eq=scenario.expected_count)
                tm.that(parsed, is_=(list, tuple))
            else:
                _ = u.Tests.Result.assert_failure(result)
                error_msg = result.error
                assert error_msg is not None and scenario.expected_error is not None
                tm.that(error_msg, has=scenario.expected_error)

    def test_coerce_list_validator(self) -> None:
        validator = u.coerce_list_validator(self.Status)
        for scenario in self._coerce_list_validator_scenarios():
            if scenario.expected_success:
                result = validator(scenario.value)
                tm.that(result, is_=list)
                tm.that(all(item in self.Status for item in result), eq=True)
            else:
                with pytest.raises(Exception) as exc_info:
                    validator(scenario.value)
                expected_error = scenario.expected_error
                assert expected_error is not None
                tm.that(exc_info.value, is_=(TypeError, ValueError))
                tm.that(str(exc_info.value), has=expected_error)

    def test_parse_mapping(self) -> None:
        for scenario in self._parse_mapping_scenarios():
            result = u.parse_mapping(self.Status, scenario.mapping)
            if scenario.expected_success:
                _ = u.Tests.Result.assert_success(result)
                parsed = result.value
                tm.that(len(parsed), eq=scenario.expected_count)
                tm.that(parsed, is_=dict)
                tm.that(all(v in self.Status for v in parsed.values()), eq=True)
            else:
                _ = u.Tests.Result.assert_failure(result)
                error_msg = result.error
                assert error_msg is not None and scenario.expected_error is not None
                tm.that(error_msg, has=scenario.expected_error)
