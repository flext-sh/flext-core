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

from collections.abc import Callable, Mapping
from typing import override

from pydantic import BaseModel, ConfigDict, Field

from flext_core import FlextModels, p, t


class TestsFlextModels:
    """Models for flext-core tests - uses composition with FlextTestsModels.

    Architecture: Uses composition (not inheritance) with FlextTestsModels and FlextModels
    for flext-core-specific model definitions.

    Access patterns:
    - TestsFlextModels.FlextTestsModels.Tests.* = flext_tests test models (via composition)
    - TestsFlextModels.Core.* = flext-core-specific test models
    - TestsFlextModels.FlextModels.Entity, .FlextModels.Value, etc. = FlextModels domain models (via composition)

    Rules:
    - Use composition, not inheritance (FlextTestsModels deprecates subclassing)
    - flext-core-specific models go in Core namespace
    - Generic models accessed via FlextTestsModels.Tests namespace
    """

    AggregateRoot = FlextModels.AggregateRoot
    DomainEvent = FlextModels.DomainEvent

    # Type aliases for domain test input
    type DomainInputValue = t.ContainerValue | p.HasModelDump
    type DomainInputMapping = Mapping[str, TestsFlextModels.DomainInputValue]
    type DomainExpectedResult = t.ContainerValue | type[t.ContainerValue]

    class Core:
        """flext-core-specific test models namespace."""

        class DomainTestEntity:
            """Test entity for domain tests."""

            def __init__(self, name: str, value: t.ContainerValue) -> None:
                """Initialize test entity with name and value."""
                self.name = name
                self.value = value
                self.unique_id = f"test-{name}-{value}"

        class DomainTestValue:
            """Test value object for domain tests."""

            _frozen = False

            def __init__(self, data: str = "", count: int = 0) -> None:
                """Initialize test value object with optional data and count."""
                self._frozen = False
                self.data = data
                self.count = count
                self._frozen = True

            @override
            def __setattr__(self, name: str, value: object) -> None:
                """Set attribute with frozen state validation."""
                if getattr(self, "_frozen", False) and name != "_frozen":
                    raise AttributeError(
                        f"{type(self).__name__} object attribute '{name}' is read-only",
                    )
                super().__setattr__(name, value)

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

        obj: t.ContainerValue
        expected_contains: list[str] | None = None
        expected_exact: str | None = None
        description: str = Field(default="", exclude=True)

    # =========================================================================
    # Fixture models (Pydantic v2 replacements for TypedDicts from typings.py)
    # =========================================================================

    class Fixtures:
        """Pydantic v2 fixture models — replaces all TypedDicts from typings.py."""

        class GenericFieldsModel(BaseModel):
            """Generic model for flexible test data and configurations."""

            model_config = ConfigDict(extra="allow")

        class GenericTestCaseModel(BaseModel):
            """Generic test case model for parameterized tests."""

            model_config = ConfigDict(extra="allow")

        class BddPhaseModel(BaseModel):
            """BDD phase (given/when/then) configuration."""

            description: str | None = None

        class BddPhaseData(BaseModel):
            """BDD phase data (given/when/then)."""

            description: str | None = None
            assertions: list[str] | None = None
            setup_steps: list[str] | None = None

        class MockScenarioData(BaseModel):
            """Mock scenario test data."""

            given: dict[str, str | int | bool] | None = None
            when: dict[str, str | int | bool] | None = None
            then: dict[str, str | int | bool] | None = None
            tags: list[str] | None = None
            priority: str | None = None

        class NestedDataModel(BaseModel):
            """Nested test data."""

            key: str | None = None
            value: str | int | bool | None = None
            metadata: str | None = None

        class FixtureDataModel(BaseModel):
            """Test data for FlextTestBuilder."""

            id: str | None = None
            correlation_id: str | None = None
            created_at: str | None = None
            updated_at: str | None = None
            name: str | None = None
            email: str | None = None
            environment: str | None = None
            version: str | None = None
            nested_data: Mapping[str, t.ContainerValue] | None = None

        class FixtureCaseModel(BaseModel):
            """Individual test case configuration."""

            email: str | None = None
            input: str | None = None

        class SuccessCaseModel(BaseModel):
            """Success test case."""

            email: str | None = None
            input: str | None = None

        class FailureCaseModel(BaseModel):
            """Failure test case."""

            email: str | None = None
            input: str | None = None

        class SetupDataModel(BaseModel):
            """Setup data for test suite."""

            initialization_step: str | None = None
            configuration_key: str | None = None
            configuration_value: str | None = None
            environment: str | None = None

        class FixtureSuiteModel(BaseModel):
            """Test suite configuration."""

            suite_name: str
            scenario_count: int
            tags: list[str]
            setup_data: Mapping[str, t.ContainerValue] | None = None

        class UserDataFixtureModel(BaseModel):
            """User fixture data."""

            username: str | None = None
            email: str | None = None
            status: str | None = None

        class RequestDataFixtureModel(BaseModel):
            """Request fixture data."""

            method: str | None = None
            path: str | None = None
            headers: dict[str, str] | None = None

        class FixtureFixturesModel(BaseModel):
            """Test fixtures configuration."""

            user: Mapping[str, t.ContainerValue] | None = None
            request: Mapping[str, t.ContainerValue] | None = None

        class UserProfileModel(BaseModel):
            """User profile for property-based testing."""

            id: str
            name: str
            email: str

        class ConfigTestCaseModel(BaseModel):
            """Configuration test case."""

            domain: str | None = None
            port: int | None = None
            timeout: float | None = None
            debug: bool | None = None

        class PerformanceMetricsModel(BaseModel):
            """Performance metrics from testing."""

            total_operations: int
            time_elapsed: float
            ops_per_second: float
            memory_peak_mb: float

        class StressTestResultModel(BaseModel):
            """Result from stress testing."""

            iterations: int
            success_count: int
            failure_count: int
            average_time_ms: float

        class AsyncPayloadModel(BaseModel):
            """Async event payload."""

            data: str | None = None
            status: str | None = None

        class AsyncTestDataModel(BaseModel):
            """Async test data."""

            event_type: str | None = None
            timestamp: str | None = None
            payload: Mapping[str, t.ContainerValue] | None = None

        class UserPayloadModel(BaseModel):
            """User command payload."""

            username: str | None = None
            email: str | None = None

        class UpdateFieldModel(BaseModel):
            """Individual update field."""

            field_name: str | None = None
            new_value: str | int | bool | None = None

        class UpdatePayloadModel(BaseModel):
            """Update command payload."""

            target_user_id: str
            updates: Mapping[str, t.ConfigurationMapping] | None = None

        class UserDataModel(BaseModel):
            """User data response."""

            id: str | None = None
            username: str | None = None
            email: str | None = None
            status: str | None = None

        class UpdateResultModel(BaseModel):
            """Update operation result."""

            user_id: str | None = None
            updated_fields: list[str] | None = None
            update_count: int | None = None

        class CommandPayloadModel(BaseModel):
            """Generic command payload."""

            id: str | None = None
            username: str | None = None
            email: str | None = None


class AutomatedTestScenario(BaseModel):
    """Pydantic v2 model for automated test scenarios."""

    description: str
    input: dict[str, t.ContainerValue]
    expected_success: bool


__all__ = [
    "AutomatedTestScenario",
    "TestsFlextModels",
]
