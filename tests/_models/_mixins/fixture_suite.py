"""Fixture suite model helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, ClassVar

from flext_core import m

if TYPE_CHECKING:
    from tests import p, t


class TestsFlextModelsFixtureSuiteMixin:
    """Fixture suite model helpers."""

    class AutomatedTestScenario(m.BaseModel):
        """Pydantic v2 model for automated test scenarios."""

        model_config: ClassVar[t.ConfigDict] = m.ConfigDict(frozen=True)

        description: str
        input: t.JsonValue
        expected_success: bool

    class StandardTestCaseModel(m.BaseModel):
        """Standard operation case model for shared test utilities."""

        description: str
        input_data: t.JsonValue
        expected_result: t.JsonValue
        expected_success: bool = True
        error_contains: str | None = None

    class UtilityEntityModel(m.Entity):
        """Shared entity model for generic test fixtures."""

        model_config: ClassVar[t.ConfigDict] = m.ConfigDict(frozen=False)

        name: Annotated[str, m.Field(description="Fixture entity name.")]
        value: Annotated[
            t.JsonValue,
            m.Field(description="Fixture entity payload."),
        ]

    class UtilityValueModel(m.Value):
        """Shared value model for generic test fixtures."""

        model_config: ClassVar[t.ConfigDict] = m.ConfigDict(frozen=True)

        value: Annotated[
            t.JsonValue,
            m.Field(description="Fixture value payload."),
        ]

    class MockScenarioData(m.BaseModel):
        """Mock scenario test data."""

        model_config: ClassVar[t.ConfigDict] = m.ConfigDict(frozen=True)

        given: t.MappingKV[str, t.Primitives]
        when: t.MappingKV[str, t.Primitives]
        then: t.MappingKV[str, t.Primitives]
        tags: t.StrSequence
        priority: str

    class NestedDataDict(m.BaseModel):
        """Nested test data."""

        model_config: ClassVar[t.ConfigDict] = m.ConfigDict(frozen=True)

        key: str
        value: t.Primitives
        metadata: str

    class FixtureDataDict(m.BaseModel):
        """Test data for FlextTestBuilder."""

        model_config: ClassVar[t.ConfigDict] = m.ConfigDict(frozen=True)

        id: str
        correlation_id: str
        created_at: str
        updated_at: str
        name: str
        email: str
        environment: str
        version: str
        nested_data: t.MappingKV[
            str,
            TestsFlextModelsFixtureSuiteMixin.NestedDataDict,
        ]

    class FixtureCaseDict(m.BaseModel):
        """Individual test case configuration."""

        model_config: ClassVar[t.ConfigDict] = m.ConfigDict(frozen=True)

        email: str
        input: str

    class SuccessCaseDict(m.BaseModel):
        """Success test case."""

        model_config: ClassVar[t.ConfigDict] = m.ConfigDict(frozen=True)

        email: str
        input: str

    class FailureCaseDict(m.BaseModel):
        """Failure test case."""

        model_config: ClassVar[t.ConfigDict] = m.ConfigDict(frozen=True)

        email: str
        input: str

    class SetupDataDict(m.BaseModel):
        """Setup data for test suite."""

        model_config: ClassVar[t.ConfigDict] = m.ConfigDict(frozen=True)

        initialization_step: str
        configuration_key: str
        configuration_value: str
        environment: str

    class FixtureSuiteDict(m.BaseModel):
        """Test suite configuration."""

        model_config: ClassVar[t.ConfigDict] = m.ConfigDict(frozen=True)

        suite_name: str
        scenario_count: int
        tags: t.StrSequence
        setup_data: t.MappingKV[
            str,
            TestsFlextModelsFixtureSuiteMixin.SetupDataDict,
        ]

    class UserDataFixtureDict(m.BaseModel):
        """User fixture data."""

        model_config: ClassVar[t.ConfigDict] = m.ConfigDict(frozen=True)

        username: str
        email: str
        status: str

    class RequestDataFixtureDict(m.BaseModel):
        """Request fixture data."""

        model_config: ClassVar[t.ConfigDict] = m.ConfigDict(frozen=True)

        method: str
        path: str
        headers: t.StrMapping

    class FixtureFixturesDict(m.BaseModel):
        """Test fixtures configuration."""

        model_config: ClassVar[t.ConfigDict] = m.ConfigDict(frozen=True)

        user: t.MappingKV[
            str,
            TestsFlextModelsFixtureSuiteMixin.UserDataFixtureDict,
        ]
        request: t.MappingKV[
            str,
            TestsFlextModelsFixtureSuiteMixin.RequestDataFixtureDict,
        ]


__all__: list[str] = ["TestsFlextModelsFixtureSuiteMixin"]
