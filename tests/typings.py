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

from flext_core import T, T_co, T_contra, t
from flext_tests import FlextTestsTypes


class TestsFlextTypes(FlextTestsTypes):
    """Type system foundation for flext-core tests - extends FlextTestsTypes.

    Architecture: Extends FlextTestsTypes with flext-core-specific type definitions.
    All generic types from FlextTestsTypes are available through inheritance.

    Rules:
    - NEVER redeclare types from FlextTestsTypes
    - Only flext-core-specific types allowed (not generic for other projects)
    - All generic types come from FlextTestsTypes
    """

    class Core:
        """Flext-core-specific type definitions for testing.

        Uses composition of t for type safety and consistency.
        Only defines types that are truly flext-core-specific.
        """

        type ServiceConfigMapping = Mapping[
            str, t.ContainerValue | Sequence[str] | Mapping[str, str | int] | None
        ]
        "Service configuration mapping specific to flext-core services."
        type HandlerConfigMapping = Mapping[
            str, t.ContainerValue | Sequence[str] | Mapping[str, str] | None
        ]
        "Handler configuration mapping specific to flext-core handlers."

    class Fixtures:
        """TypedDict definitions for test fixtures."""

        class GenericFieldsDict:
            """Generic dictionary for flexible test data and configurations."""

        class GenericTestCaseDict:
            """Generic test case dictionary for parameterized tests."""

        class BddPhaseDict:
            """BDD phase (given/when/then) configuration."""

            description: str

        class BddPhaseData:
            """BDD phase data (given/when/then)."""

            description: str
            assertions: list[str]
            setup_steps: list[str]

        class MockScenarioData:
            """Mock scenario test data."""

            given: dict[str, str | int | bool]
            when: dict[str, str | int | bool]
            then: dict[str, str | int | bool]
            tags: list[str]
            priority: str

        class NestedDataDict:
            """Nested test data."""

            key: str
            value: str | int | bool
            metadata: str

        class FixtureDataDict:
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

        class FixtureCaseDict:
            """Individual test case configuration."""

            email: str
            input: str

        class SuccessCaseDict:
            """Success test case."""

            email: str
            input: str

        class FailureCaseDict:
            """Failure test case."""

            email: str
            input: str

        class SetupDataDict:
            """Setup data for test suite."""

            initialization_step: str
            configuration_key: str
            configuration_value: str
            environment: str

        class FixtureSuiteDict:
            """Test suite configuration."""

            suite_name: str
            scenario_count: int
            tags: list[str]
            setup_data: dict[str, TestsFlextTypes.Fixtures.SetupDataDict]

        class UserDataFixtureDict:
            """User fixture data."""

            username: str
            email: str
            status: str

        class RequestDataFixtureDict:
            """Request fixture data."""

            method: str
            path: str
            headers: dict[str, str]

        class FixtureFixturesDict:
            """Test fixtures configuration."""

            user: dict[str, TestsFlextTypes.Fixtures.UserDataFixtureDict]
            request: dict[str, TestsFlextTypes.Fixtures.RequestDataFixtureDict]

        class UserProfileDict:
            """User profile for property-based testing."""

            id: str
            name: str
            email: str

        class ConfigTestCaseDict:
            """Configuration test case."""

            domain: str
            port: int
            timeout: float
            debug: bool

        class PerformanceMetricsDict:
            """Performance metrics from testing."""

            total_operations: int
            time_elapsed: float
            ops_per_second: float
            memory_peak_mb: float

        class StressTestResultDict:
            """Result from stress testing."""

            iterations: int
            success_count: int
            failure_count: int
            average_time_ms: float

        class AsyncPayloadDict:
            """Async event payload."""

            data: str
            status: str

        class AsyncTestDataDict:
            """Async test data."""

            event_type: str
            timestamp: str
            payload: dict[str, TestsFlextTypes.Fixtures.AsyncPayloadDict]

        class UserPayloadDict:
            """User command payload."""

            username: str
            email: str

        class UpdateFieldDict:
            """Individual update field."""

            field_name: str
            new_value: str | int | bool

        class UpdatePayloadDict:
            """Update command payload."""

            target_user_id: str
            updates: dict[str, TestsFlextTypes.Fixtures.UpdateFieldDict]

        class UserDataDict:
            """User data response."""

            id: str
            username: str
            email: str
            status: str

        class UpdateResultDict:
            """Update operation result."""

            user_id: str
            updated_fields: list[str]
            update_count: int

        class CommandPayloadDict:
            """Generic command payload."""

            id: str
            username: str
            email: str


__all__ = ["T", "T_co", "T_contra", "TestsFlextTypes"]
