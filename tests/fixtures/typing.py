"""Type definitions for flext-core test fixtures using Python 3.13 patterns.

Module functionality: Centralized TypedDict definitions for all test fixtures.
Provides type-safe configuration dictionaries replacing generic dict[str, object].

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TypedDict


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


class TestDataDict(TypedDict, total=False):
    """Test data for FlextTestBuilder."""


    id: str
    correlation_id: str
    created_at: str
    updated_at: str
    name: str
    email: str
    environment: str
    version: str
    nested_data: dict[str, NestedDataDict]


class TestCaseDict(TypedDict, total=False):
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


class TestSuiteDict(TypedDict):
    """Test suite configuration."""


    suite_name: str
    scenario_count: int
    tags: list[str]
    setup_data: dict[str, SetupDataDict]


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


class TestFixturesDict(TypedDict, total=False):
    """Test fixtures configuration."""


    user: dict[str, UserDataFixtureDict]
    request: dict[str, RequestDataFixtureDict]


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
    payload: dict[str, AsyncPayloadDict]


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
    updates: dict[str, UpdateFieldDict]


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
    "AsyncPayloadDict",
    "AsyncTestDataDict",
    "BddPhaseData",
    "BddPhaseDict",
    "CommandPayloadDict",
    "ConfigTestCaseDict",
    "FailureCaseDict",
    "GenericFieldsDict",
    "GenericTestCaseDict",
    "MockScenarioData",
    "NestedDataDict",
    "PerformanceMetricsDict",
    "RequestDataFixtureDict",
    "SetupDataDict",
    "StressTestResultDict",
    "SuccessCaseDict",
    "TestCaseDict",
    "TestDataDict",
    "TestFixturesDict",
    "TestSuiteDict",
    "UpdateFieldDict",
    "UpdatePayloadDict",
    "UpdateResultDict",
    "UserDataDict",
    "UserDataFixtureDict",
    "UserPayloadDict",
    "UserProfileDict",
]
