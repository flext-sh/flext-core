"""Type system foundation for flext-core tests.

Provides TestsFlextTypes, extending FlextTestsTypes with flext-core-specific types.
All generic test types come from flext_tests, only flext-core-specific additions here.

Architecture:
- FlextTestsTypes (flext_tests) = Generic types for all FLEXT projects
- TestsFlextTypes (tests/) = flext-core-specific types extending FlextTestsTypes

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypedDict

from flext_core.typings import T, T_co, T_contra, t as core_t
from flext_tests.typings import (
    FlextTestsTypes,
    TTestModel,
    TTestResult,
    TTestService,
)


class TestsFlextTypes(FlextTestsTypes):
    """Type system foundation for flext-core tests - extends FlextTestsTypes.

    Architecture: Extends FlextTestsTypes with flext-core-specific type definitions.
    All generic types from FlextTestsTypes are available through inheritance.

    Rules:
    - NEVER redeclare types from FlextTestsTypes
    - Only flext-core-specific types allowed (not generic for other projects)
    - All generic types come from FlextTestsTypes
    """

    class Core(FlextTestsTypes.Core):
        """Flext-core-specific type definitions for testing.

        Uses composition of core_t for type safety and consistency.
        Only defines types that are truly flext-core-specific.
        """

        type ServiceConfigMapping = Mapping[
            str,
            core_t.GeneralValueType | Sequence[str] | Mapping[str, str | int] | None,
        ]
        """Service configuration mapping specific to flext-core services."""

        type HandlerConfigMapping = Mapping[
            str,
            core_t.GeneralValueType | Sequence[str] | Mapping[str, str] | None,
        ]
        """Handler configuration mapping specific to flext-core handlers."""

    class Fixtures:
        """TypedDict definitions for test fixtures."""

        class GenericFieldsDict(TypedDict, total=False):
            """Generic dictionary for flexible test data and configurations."""

        class GenericTestCaseDict(TypedDict, total=False):
            """Generic test case dictionary for parameterized tests."""

        class BddPhaseDict(TypedDict, total=False):
            """BDD phase (given/when/then) configuration."""

            description: str

        class BddPhaseData(TypedDict, total=False):
            """BDD phase data (given/when/then)."""

            description: str
            assertions: list[str]
            setup_steps: list[str]

        class MockScenarioData(TypedDict, total=False):
            """Mock scenario test data."""

            given: dict[str, str | int | bool]
            when: dict[str, str | int | bool]
            then: dict[str, str | int | bool]
            tags: list[str]
            priority: str

        class NestedDataDict(TypedDict, total=False):
            """Nested test data."""

            key: str
            value: str | int | bool
            metadata: str

        class FixtureDataDict(TypedDict, total=False):
            """Test data for FlextTestBuilder."""

            id: str
            correlation_id: str
            created_at: str
            updated_at: str
            name: str
            email: str
            environment: str
            version: str
            nested_data: dict[str, TestsFlextTypes.Fixtures.NestedDataDict]

        class FixtureCaseDict(TypedDict, total=False):
            """Individual test case configuration."""

            email: str
            input: str

        class SuccessCaseDict(TypedDict, total=False):
            """Success test case."""

            email: str
            input: str

        class FailureCaseDict(TypedDict, total=False):
            """Failure test case."""

            email: str
            input: str

        class SetupDataDict(TypedDict, total=False):
            """Setup data for test suite."""

            initialization_step: str
            configuration_key: str
            configuration_value: str
            environment: str

        class FixtureSuiteDict(TypedDict):
            """Test suite configuration."""

            suite_name: str
            scenario_count: int
            tags: list[str]
            setup_data: dict[str, TestsFlextTypes.Fixtures.SetupDataDict]

        class UserDataFixtureDict(TypedDict, total=False):
            """User fixture data."""

            username: str
            email: str
            status: str

        class RequestDataFixtureDict(TypedDict, total=False):
            """Request fixture data."""

            method: str
            path: str
            headers: dict[str, str]

        class FixtureFixturesDict(TypedDict, total=False):
            """Test fixtures configuration."""

            user: dict[str, TestsFlextTypes.Fixtures.UserDataFixtureDict]
            request: dict[str, TestsFlextTypes.Fixtures.RequestDataFixtureDict]

        class UserProfileDict(TypedDict):
            """User profile for property-based testing."""

            id: str
            name: str
            email: str

        class ConfigTestCaseDict(TypedDict, total=False):
            """Configuration test case."""

            domain: str
            port: int
            timeout: float
            debug: bool

        class PerformanceMetricsDict(TypedDict):
            """Performance metrics from testing."""

            total_operations: int
            time_elapsed: float
            ops_per_second: float
            memory_peak_mb: float

        class StressTestResultDict(TypedDict):
            """Result from stress testing."""

            iterations: int
            success_count: int
            failure_count: int
            average_time_ms: float

        class AsyncPayloadDict(TypedDict, total=False):
            """Async event payload."""

            data: str
            status: str

        class AsyncTestDataDict(TypedDict, total=False):
            """Async test data."""

            event_type: str
            timestamp: str
            payload: dict[str, TestsFlextTypes.Fixtures.AsyncPayloadDict]

        class UserPayloadDict(TypedDict, total=False):
            """User command payload."""

            username: str
            email: str

        class UpdateFieldDict(TypedDict, total=False):
            """Individual update field."""

            field_name: str
            new_value: str | int | bool

        class UpdatePayloadDict(TypedDict):
            """Update command payload."""

            target_user_id: str
            updates: dict[str, TestsFlextTypes.Fixtures.UpdateFieldDict]

        class UserDataDict(TypedDict, total=False):
            """User data response."""

            id: str
            username: str
            email: str
            status: str

        class UpdateResultDict(TypedDict, total=False):
            """Update operation result."""

            user_id: str
            updated_fields: list[str]
            update_count: int

        class CommandPayloadDict(TypedDict, total=False):
            """Generic command payload."""

            id: str
            username: str
            email: str


__all__ = [
    "T",
    "TTestModel",
    "TTestResult",
    "TTestService",
    "T_co",
    "T_contra",
    "TestsFlextTypes",
]
