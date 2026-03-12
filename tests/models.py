"""Models for flext-core tests.

Provides TestsFlextModels using composition with FlextTestsModels and FlextModels.
All generic test models come from flext_tests.

Architecture:
- FlextTestsModels (flext_tests) = Generic models for all FLEXT projects
- FlextModels (flext_core) = Core domain models
- TestsFlextModels (tests/) = flext-core-specific models using composition

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import override

from pydantic import BaseModel, ConfigDict, Field

from flext_core import FlextModels
from flext_infra import FlextInfraModels
from flext_tests import FlextTestsModels


class TestsFlextModels(FlextTestsModels, FlextInfraModels):
    """Models for flext-core tests - uses composition with FlextTestsModels.

    Architecture: Uses composition (not inheritance) with FlextTestsModels and FlextModels
    for flext-core-specific model definitions.

    Access patterns:
    - TestsFlextModels.FlextTestsModels.Tests.* = flext_tests test models (via composition)
    - TestsFlextModels.Tests.* = flext-core-specific test models
    - TestsFlextModels.FlextModels.Entity, .FlextModels.Value, etc. = FlextModels domain models (via composition)

    Rules:
    - Use composition, not inheritance (FlextTestsModels deprecates subclassing)
    - flext-core-specific models go in Core namespace
    - Generic models accessed via FlextTestsModels.Tests namespace
    """

    class Tests(FlextTestsModels.Tests):
        """flext-core-specific test models namespace."""

        class ServiceTestType(StrEnum):
            """Service test type enum for test scenarios."""

            GET_USER = "get_user"
            VALIDATE = "validate"
            FAIL = "fail"

        class User(FlextModels.Entity):
            """Shared user model for tests."""

            model_config = ConfigDict(frozen=False)

            user_id: str
            name: str
            email: str
            is_active: bool = True

        class ServiceTestCase(BaseModel):
            """Service execution case model for tests."""

            model_config = ConfigDict(frozen=True)

            service_type: str
            input_value: str
            expected_success: bool = True
            expected_error: str | None = None
            extra_param: int = 3
            description: str = ""

        class DomainTestEntity(FlextModels.Entity):
            """Test entity for domain tests."""

            model_config = ConfigDict(frozen=False)

            name: str
            value: object

        class DomainTestValue(FlextModels.Value):
            """Test value object for domain tests."""

            model_config = ConfigDict(frozen=True)

            data: str = ""
            count: int

        class CustomEntity:
            """Custom entity with configurable ID attribute."""

            def __init__(self, custom_id: str | None = None) -> None:
                """Initialize custom entity with ID."""
                self.custom_id = custom_id

        class SimpleValue:
            """Simple value object without model_dump."""

            def __init__(self, data: str) -> None:
                """Initialize simple value object."""
                self.data = data

        class ComplexValue:
            """FlextModels.Value object with non-hashable attributes."""

            def __init__(self, data: str, items: list[str]) -> None:
                """Initialize complex value with non-hashable items."""
                self.data = data
                self.items = items  # list is not hashable

        class NoDict:
            """Object without __dict__, using __slots__."""

            __slots__ = ("value",)

            def __init__(self, value: int) -> None:
                """Initialize object without __dict__."""
                object.__setattr__(self, "value", value)

            @override
            def __repr__(self) -> str:
                """Return string representation."""
                return f"NoDict({getattr(self, 'value', None)})"

        class MutableObj:
            """Mutable object for immutability testing."""

            def __init__(self, value: int) -> None:
                """Initialize mutable object."""
                self.value = value

        class ImmutableObj:
            """Immutable object with custom __setattr__."""

            _frozen: bool = True

            def __init__(self, value: int) -> None:
                """Initialize immutable object."""
                object.__setattr__(self, "value", value)

            @override
            def __setattr__(self, name: str, value: object) -> None:
                """Prevent attribute setting if frozen."""
                if self._frozen:
                    msg = "Object is frozen"
                    raise AttributeError(msg)
                object.__setattr__(self, name, value)

        class NoConfigNoSetattr:
            """Object without model_config or __setattr__."""

        class NoSetattr:
            """Object without __setattr__."""

        # ParseOptions reference for string parser tests
        # ParseOptions reference for string parser tests
        class ParseOptions(FlextModels.CollectionsParseOptions):
            """Parse options - real inheritance."""

        class ParseDelimitedCase(BaseModel):
            """Test case for parse_delimited method."""

            model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

            text: str
            delimiter: str
            expected: list[str] | None = None
            expected_error: str | None = None
            options: FlextModels.CollectionsParseOptions | None = None
            strip: bool = True
            remove_empty: bool = True
            validator: Callable[[str], bool] | None = None
            use_legacy: bool = False
            description: str = Field(default="", exclude=True)

        class SplitEscapeCase(BaseModel):
            """Test case for split_on_char_with_escape method."""

            model_config = ConfigDict(frozen=True)

            text: str
            split_char: str
            escape_char: str = "\\"
            expected: list[str] | None = None
            expected_error: str | None = None
            description: str = Field(default="", exclude=True)

        class NormalizeWhitespaceCase(BaseModel):
            """Test case for normalize_whitespace method."""

            model_config = ConfigDict(frozen=True)

            text: str
            pattern: str = r"\s+"
            replacement: str = " "
            expected: str | None = None
            expected_error: str | None = None
            description: str = Field(default="", exclude=True)

        class RegexPipelineCase(BaseModel):
            """Test case for apply_regex_pipeline method."""

            model_config = ConfigDict(frozen=True)

            text: str
            patterns: list[tuple[str, str] | tuple[str, str, int]]
            expected: str | None = None
            expected_error: str | None = None
            description: str = Field(default="", exclude=True)

        class ObjectKeyCase(BaseModel):
            """Test case for get_object_key method."""

            model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

            obj: object
            expected_contains: list[str] | None = None
            expected_exact: str | None = None
            description: str = Field(default="", exclude=True)

        class AutomatedTestScenario(BaseModel):
            """Pydantic v2 model for automated test scenarios."""

            model_config = ConfigDict(frozen=True)

            description: str
            input: object
            expected_success: bool

        class StandardTestCaseModel(BaseModel):
            """Standard operation case model for shared test utilities."""

            description: str
            input_data: object
            expected_result: object
            expected_success: bool = True
            error_contains: str | None = None

        class UtilityEntityModel(FlextModels.Entity):
            """Shared entity model for generic test fixtures."""

            model_config = ConfigDict(frozen=False)

            name: str
            value: object

        class UtilityValueModel(FlextModels.Value):
            """Shared value model for generic test fixtures."""

            model_config = ConfigDict(frozen=True)

            value: object

        class BddPhaseDict(BaseModel):
            """BDD phase (given/when/then) configuration."""

            model_config = ConfigDict(frozen=True)

            description: str

        class BddPhaseData(BaseModel):
            """BDD phase data (given/when/then)."""

            model_config = ConfigDict(frozen=True)

            description: str
            assertions: list[str]
            setup_steps: list[str]

        class MockScenarioData(BaseModel):
            """Mock scenario test data."""

            model_config = ConfigDict(frozen=True)

            given: dict[str, str | int | bool]
            when: dict[str, str | int | bool]
            then: dict[str, str | int | bool]
            tags: list[str]
            priority: str

        class NestedDataDict(BaseModel):
            """Nested test data."""

            model_config = ConfigDict(frozen=True)

            key: str
            value: str | int | bool
            metadata: str

        class FixtureDataDict(BaseModel):
            """Test data for FlextTestBuilder."""

            model_config = ConfigDict(frozen=True)

            id: str
            correlation_id: str
            created_at: str
            updated_at: str
            name: str
            email: str
            environment: str
            version: str
            nested_data: dict[str, TestsFlextModels.Tests.NestedDataDict]

        class FixtureCaseDict(BaseModel):
            """Individual test case configuration."""

            model_config = ConfigDict(frozen=True)

            email: str
            input: str

        class SuccessCaseDict(BaseModel):
            """Success test case."""

            model_config = ConfigDict(frozen=True)

            email: str
            input: str

        class FailureCaseDict(BaseModel):
            """Failure test case."""

            model_config = ConfigDict(frozen=True)

            email: str
            input: str

        class SetupDataDict(BaseModel):
            """Setup data for test suite."""

            model_config = ConfigDict(frozen=True)

            initialization_step: str
            configuration_key: str
            configuration_value: str
            environment: str

        class FixtureSuiteDict(BaseModel):
            """Test suite configuration."""

            model_config = ConfigDict(frozen=True)

            suite_name: str
            scenario_count: int
            tags: list[str]
            setup_data: dict[str, TestsFlextModels.Tests.SetupDataDict]

        class UserDataFixtureDict(BaseModel):
            """User fixture data."""

            model_config = ConfigDict(frozen=True)

            username: str
            email: str
            status: str

        class RequestDataFixtureDict(BaseModel):
            """Request fixture data."""

            model_config = ConfigDict(frozen=True)

            method: str
            path: str
            headers: dict[str, str]

        class FixtureFixturesDict(BaseModel):
            """Test fixtures configuration."""

            model_config = ConfigDict(frozen=True)

            user: dict[str, TestsFlextModels.Tests.UserDataFixtureDict]
            request: dict[str, TestsFlextModels.Tests.RequestDataFixtureDict]

        class UserProfileDict(BaseModel):
            """User profile for property-based testing."""

            model_config = ConfigDict(frozen=True)

            id: str
            name: str
            email: str

        class ConfigTestCaseDict(BaseModel):
            """Configuration test case."""

            model_config = ConfigDict(frozen=True)

            domain: str
            port: int
            timeout: float
            debug: bool

        class PerformanceMetricsDict(BaseModel):
            """Performance metrics from testing."""

            model_config = ConfigDict(frozen=True)

            total_operations: int
            time_elapsed: float
            ops_per_second: float
            memory_peak_mb: float

        class StressTestResultDict(BaseModel):
            """Result from stress testing."""

            model_config = ConfigDict(frozen=True)

            iterations: int
            success_count: int
            failure_count: int
            average_time_ms: float

        class AsyncPayloadDict(BaseModel):
            """Async event payload."""

            model_config = ConfigDict(frozen=True)

            data: str
            status: str

        class AsyncTestDataDict(BaseModel):
            """Async test data."""

            model_config = ConfigDict(frozen=True)

            event_type: str
            timestamp: str
            payload: dict[str, TestsFlextModels.Tests.AsyncPayloadDict]

        class UserPayloadDict(BaseModel):
            """User command payload."""

            model_config = ConfigDict(frozen=True)

            username: str
            email: str

        class UpdateFieldDict(BaseModel):
            """Individual update field."""

            model_config = ConfigDict(frozen=True)

            field_name: str
            new_value: str | int | bool

        class UpdatePayloadDict(BaseModel):
            """Update command payload."""

            model_config = ConfigDict(frozen=True)

            target_user_id: str
            updates: dict[str, TestsFlextModels.Tests.UpdateFieldDict]

        class UserDataDict(BaseModel):
            """User data response."""

            model_config = ConfigDict(frozen=True)

            id: str
            username: str
            email: str
            status: str

        class UpdateResultDict(BaseModel):
            """Update operation result."""

            model_config = ConfigDict(frozen=True)

            user_id: str
            updated_fields: list[str]
            update_count: int

        class CommandPayloadDict(BaseModel):
            """Generic command payload."""

            model_config = ConfigDict(frozen=True)

            id: str = ""
            username: str = ""
            email: str = ""


m = TestsFlextModels

__all__ = [
    "TestsFlextModels",
    "m",
]
