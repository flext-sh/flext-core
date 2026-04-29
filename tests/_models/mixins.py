from __future__ import annotations

from collections import UserDict, UserList
from collections.abc import (
    Callable,
    ItemsView,
    Iterator,
    Mapping,
    MutableSequence,
    Sequence,
)
from typing import Annotated, ClassVar, Never, Self, override

from flext_core import m
from tests import c, t


class TestsFlextModelsMixins:
    """flext-core test models namespace."""

    class BadDict(UserDict[str, t.Tests.TestobjectSerializable]):
        """Dict that raises on get()."""

        @override
        def __getitem__(self, key: str) -> Never:
            """Raise error on get attempt."""
            _ = key
            msg = c.Tests.TestErrors.BAD_DICT_GET
            raise RuntimeError(msg)

    class BadList(UserList[t.Tests.TestobjectSerializable]):
        """List that raises on iteration."""

        @override
        def __iter__(self) -> Iterator[t.Tests.TestobjectSerializable]:
            """Raise error on iteration."""
            msg = c.Tests.TestErrors.BAD_LIST_ITERATION
            raise RuntimeError(msg)

    class BadModelDump:
        """Object with model_dump that raises."""

        model_dump: Callable[[], Mapping[str, t.Tests.TestobjectSerializable]] = (
            staticmethod(
                lambda: (_ for _ in ()).throw(RuntimeError("Bad model_dump")),
            )
        )

    class AttrObject(m.BaseModel):
        """Simple model with name/value attributes for mapper tests."""

        name: Annotated[
            str, m.Field(description="Attribute recursive container name")
        ] = "name"
        value: Annotated[
            int, m.Field(description="Attribute recursive container value")
        ] = 1

    class BadMapping(UserDict[str, t.JsonValue]):
        """Mapping that raises on access — used for error-path testing."""

        @override
        def __getitem__(self, key: str) -> t.JsonValue:
            _ = key
            msg = "get exploded"
            raise TypeError(msg)

        @override
        def __iter__(self) -> Iterator[str]:
            msg = "iter exploded"
            raise TypeError(msg)

        @override
        def __len__(self) -> int:
            return 1

    class _ValidationLikeError(ValueError):
        """Validation-like error for tests."""

        def errors(self) -> Sequence[Mapping[str, t.JsonValue]]:
            return [{"loc": ["value"], "msg": "bad value"}]

    type TestCaseMap = Mapping[str, t.Tests.TestobjectSerializable]
    type InputPayloadMap = Mapping[str, t.Tests.TestobjectSerializable]

    class _MsgWithCommandId(m.BaseModel):
        command_id: str = "cmd-1"

    class _MsgWithMessageId(m.BaseModel):
        message_id: str = "msg-1"

    class SampleModel(m.BaseModel):
        """Sample model for testing."""

        name: str
        value: int

    class _SvcModel(m.BaseModel):
        value: str

    class _BrokenDumpModel:
        """Test fake whose ``model_dump`` returns wrong type.

        Triggers ``TypeError`` in mapping-shaped validators (e.g.,
        ``Metadata.attributes``). Intentionally NOT a ``m.BaseModel`` subclass:
        avoids ``__getattribute__`` override (forbidden outside flext-core
        src/) while still presenting the duck-typed ``model_dump`` callable
        Pydantic runtime probes.
        """

        value: int = 1

        @staticmethod
        def model_dump() -> bool:
            return True

    class _ErrorsModel(m.BaseModel):
        value: int

        @classmethod
        @override
        def model_validate(
            cls,
            obj: t.JsonValue,
            *,
            strict: bool | None = None,
            extra: str | None = None,
            from_attributes: bool | None = None,
            context: t.JsonMapping | None = None,
            by_alias: bool | None = None,
            by_name: bool | None = None,
        ) -> Never:
            _ = strict, extra, from_attributes, context, by_alias, by_name
            _ = obj
            raise TestsFlextModelsMixins._ValidationLikeError

    class _PlainErrorModel(m.BaseModel):
        value: int

        @classmethod
        @override
        def model_validate(
            cls,
            obj: t.JsonValue,
            *,
            strict: bool | None = None,
            extra: str | None = None,
            from_attributes: bool | None = None,
            context: t.JsonMapping | None = None,
            by_alias: bool | None = None,
            by_name: bool | None = None,
        ) -> Never:
            _ = strict, extra, from_attributes, context, by_alias, by_name
            _ = obj
            msg = c.Tests.TestErrors.PLAIN_BOOM
            raise RuntimeError(msg)

    class _TargetModel(m.BaseModel):
        value: int

    class CacheTestModel(m.BaseModel):
        """Test model for cache key generation."""

        name: str
        value: int
        tags: MutableSequence[str] = []
        meta: t.MutableStrMapping = {}

    class NestedModel(m.BaseModel):
        """Nested Pydantic model for cache testing."""

        inner: TestsFlextModelsMixins.CacheTestModel
        count: int

    class SettingsModelForTest(m.BaseModel):
        """Test configuration model (mutable for set_parameter tests)."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(
            validate_assignment=True,
            extra="forbid",
        )

        name: str = "default_settings"
        timeout: Annotated[int, m.Field(ge=0)] = 30
        enabled: bool = True

    class InvalidModelForTest(m.BaseModel):
        """Model with invalid model_dump."""

        value: str = "test"

    class SingletonClassForTest(m.BaseModel):
        """Test singleton class with Pydantic validation."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(
            validate_assignment=True,
            extra="forbid",
        )

        _instance: ClassVar[TestsFlextModelsMixins.SingletonClassForTest | None] = None

        name: str = "default"
        timeout: int = 30

        @classmethod
        def fetch_global(
            cls,
        ) -> TestsFlextModelsMixins.SingletonClassForTest:
            """Get global singleton instance."""
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

        @classmethod
        def reset_instance(cls) -> None:
            """Reset singleton instance for test isolation."""
            cls._instance = None

    class BadSettingsForTest(m.BaseModel):
        """Settings that fails to instantiate."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(
            validate_assignment=True,
        )

        def __init__(self, **kwargs: t.Scalar) -> None:
            """Raise error on init."""
            super().__init__(**kwargs)
            msg = c.Tests.TestErrors.CANNOT_INSTANTIATE
            raise ValueError(msg)

    class _DumpErrorModel(m.BaseModel):
        value: int = 1

    class _Opts(m.BaseModel):
        value: int = 1

    class _FakeSettings(m.BaseModel):
        """Fake settings with model_copy support."""

        timeout: int = 10

        @property
        def data(self) -> t.JsonMapping:
            return {"timeout": self.timeout}

    class _Model(m.BaseModel):
        value: int

    class _SampleEntity(m.BaseModel):
        """Test entity for domain utility tests."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=False)

        unique_id: str = "test-123"
        name: str = "test"

    class _FrozenEntity(m.BaseModel):
        """Frozen entity for immutability tests."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        unique_id: str = "frozen-1"

    class _GoodModel(m.BaseModel):
        value: int = 7

    class ComplexModel(m.BaseModel):
        """Complex test model."""

        id: int
        data: t.JsonMapping
        items: t.StrSequence

    class _Cfg(m.BaseModel):
        x: int = 0
        y: str = "a"

    class _BadCopyModel(m.BaseModel):
        x: int = 1

    class EmailResponse(m.BaseModel):
        """Shared email response model for tests."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        status: str
        message_id: str

    class DomainTestEntity(m.Entity):
        """Test entity for domain tests."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=False)

        name: Annotated[str, m.Field(description="Entity display name.")]
        value: Annotated[
            t.JsonValue,
            m.Field(description="Entity payload value."),
        ]

    class DomainTestValue(m.Value):
        """Test value object for domain tests."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        data: Annotated[str, m.Field(description="Value payload string.")] = ""
        count: Annotated[int, m.Field(description="Occurrence counter.")]

    class CustomEntity(m.BaseModel):
        """Custom entity with configurable ID attribute."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=False)

        custom_id: str | None = None

        def __init__(self, custom_id: str | None = None, **kwargs: t.Scalar) -> None:
            """Initialize custom entity with ID."""
            super().__init__(custom_id=custom_id, **kwargs)

    class SimpleValue(m.BaseModel):
        """Simple value object — tests behavior when model_dump is absent at runtime."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=False)

        data: str = ""

        def __init__(self, data: str = "", **kwargs: t.Scalar) -> None:
            """Initialize simple value object."""
            super().__init__(data=data, **kwargs)

    class ComplexValue(m.BaseModel):
        """Value object with non-hashable attributes."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=False)

        data: str = ""
        items: t.StrSequence = []

        def __init__(
            self, data: str = "", items: t.StrSequence | None = None, **kwargs: t.Scalar
        ) -> None:
            """Initialize complex value with non-hashable items."""
            super().__init__(data=data, items=items or [], **kwargs)

    class NoDict(m.BaseModel):
        """Model for testing value-comparison fallback paths in domain utilities."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=False)

        value: int = 0

        def __init__(self, value: int = 0, **kwargs: t.Scalar) -> None:
            """Initialize model for domain utility edge-case testing."""
            super().__init__(value=value, **kwargs)

        @override
        def __repr__(self) -> str:
            """Return string representation."""
            return f"NoDict(value={self.value})"

    class MutableObj:
        """Mutable t.JsonValue for immutability testing."""

        def __init__(self, value: int) -> None:
            """Initialize mutable t.JsonValue."""
            self.value = value

    class NoSettingsNoSetattr:
        """Object without model_config or __setattr__."""

    class NoSetattr:
        """Object without __setattr__."""

    class ParseOptions(m.BaseModel):
        """Test-local parse options after production model removal."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        strip: bool = True
        remove_empty: bool = True
        validator: Callable[[str], bool] | None = None

    class ParseDelimitedCase(m.BaseModel):
        """Test case for parse_delimited method."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(
            frozen=True,
            arbitrary_types_allowed=True,
        )

        text: str
        delimiter: str
        expected: t.StrSequence | None = None
        expected_error: str | None = None
        options: m.BaseModel | None = None
        strip: bool = True
        remove_empty: bool = True
        validator: Callable[[str], bool] | None = None
        use_legacy: bool = False
        description: Annotated[str, m.Field(exclude=True)] = ""

    class SplitEscapeCase(m.BaseModel):
        """Test case for split_on_char_with_escape method."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        text: str
        split_char: str
        escape_char: str = "\\"
        expected: t.StrSequence | None = None
        expected_error: str | None = None
        description: Annotated[str, m.Field(exclude=True)] = ""

    class NormalizeWhitespaceCase(m.BaseModel):
        """Test case for normalize_whitespace method."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        text: str
        pattern: str = r"\s+"
        replacement: str = " "
        expected: str | None = None
        expected_error: str | None = None
        description: Annotated[str, m.Field(exclude=True)] = ""

    class RegexPipelineCase(m.BaseModel):
        """Test case for apply_regex_pipeline method."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        text: str
        patterns: Sequence[tuple[str, str] | tuple[str, str, int]]
        expected: str | None = None
        expected_error: str | None = None
        description: Annotated[str, m.Field(exclude=True)] = ""

    class ObjectKeyCase(m.BaseModel):
        """Test case for get_object_key method."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(
            frozen=True,
            arbitrary_types_allowed=True,
        )

        obj: t.JsonValue
        expected_contains: t.StrSequence | None = None
        expected_exact: str | None = None
        description: Annotated[str, m.Field(exclude=True)] = ""

    class AutomatedTestScenario(m.BaseModel):
        """Pydantic v2 model for automated test scenarios."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

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

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=False)

        name: Annotated[str, m.Field(description="Fixture entity name.")]
        value: Annotated[
            t.JsonValue,
            m.Field(description="Fixture entity payload."),
        ]

    class UtilityValueModel(m.Value):
        """Shared value model for generic test fixtures."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        value: Annotated[
            t.JsonValue,
            m.Field(description="Fixture value payload."),
        ]

    class MockScenarioData(m.BaseModel):
        """Mock scenario test data."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        given: Mapping[str, t.Primitives]
        when: Mapping[str, t.Primitives]
        then: Mapping[str, t.Primitives]
        tags: t.StrSequence
        priority: str

    class NestedDataDict(m.BaseModel):
        """Nested test data."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        key: str
        value: t.Primitives
        metadata: str

    class FixtureDataDict(m.BaseModel):
        """Test data for FlextTestBuilder."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        id: str
        correlation_id: str
        created_at: str
        updated_at: str
        name: str
        email: str
        environment: str
        version: str
        nested_data: Mapping[
            str,
            TestsFlextModelsMixins.NestedDataDict,
        ]

    class FixtureCaseDict(m.BaseModel):
        """Individual test case configuration."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        email: str
        input: str

    class SuccessCaseDict(m.BaseModel):
        """Success test case."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        email: str
        input: str

    class FailureCaseDict(m.BaseModel):
        """Failure test case."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        email: str
        input: str

    class SetupDataDict(m.BaseModel):
        """Setup data for test suite."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        initialization_step: str
        configuration_key: str
        configuration_value: str
        environment: str

    class FixtureSuiteDict(m.BaseModel):
        """Test suite configuration."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        suite_name: str
        scenario_count: int
        tags: t.StrSequence
        setup_data: Mapping[
            str,
            TestsFlextModelsMixins.SetupDataDict,
        ]

    class UserDataFixtureDict(m.BaseModel):
        """User fixture data."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        username: str
        email: str
        status: str

    class RequestDataFixtureDict(m.BaseModel):
        """Request fixture data."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        method: str
        path: str
        headers: t.StrMapping

    class FixtureFixturesDict(m.BaseModel):
        """Test fixtures configuration."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        user: Mapping[
            str,
            TestsFlextModelsMixins.UserDataFixtureDict,
        ]
        request: Mapping[
            str,
            TestsFlextModelsMixins.RequestDataFixtureDict,
        ]

    class UserProfileDict(m.BaseModel):
        """User profile for property-based testing."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        id: str
        name: str
        email: str

    class SettingsTestCaseDict(m.BaseModel):
        """Configuration test case."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        domain: str
        port: int
        timeout: float
        debug: bool

    class PerformanceMetricsDict(m.BaseModel):
        """Performance metrics from testing."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        total_operations: int
        time_elapsed: float
        ops_per_second: float
        memory_peak_mb: float

    class StressTestResultDict(m.BaseModel):
        """Result from stress testing."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        iterations: int
        success_count: int
        failure_count: int
        average_time_ms: float

    class AsyncPayloadDict(m.BaseModel):
        """Async event payload."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        data: str
        status: str

    class AsyncTestDataDict(m.BaseModel):
        """Async test data."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        event_type: str
        timestamp: str
        payload: Mapping[
            str,
            TestsFlextModelsMixins.AsyncPayloadDict,
        ]

    class UserPayloadDict(m.BaseModel):
        """User command payload."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        username: str
        email: str

    class UpdateFieldDict(m.BaseModel):
        """Individual update field."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        field_name: str
        new_value: t.Primitives

    class UpdatePayloadDict(m.BaseModel):
        """Update command payload."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        target_user_id: str
        updates: Mapping[
            str,
            TestsFlextModelsMixins.UpdateFieldDict,
        ]

    class UserDataDict(m.BaseModel):
        """User data response."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        id: str
        username: str
        email: str
        status: str

    class UpdateResultDict(m.BaseModel):
        """Update operation result."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        user_id: str
        updated_fields: t.StrSequence
        update_count: int

    class CommandPayloadDict(m.BaseModel):
        """Generic command payload."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        id: str = ""
        username: str = ""
        email: str = ""

    class ServiceTestCase(m.BaseModel):
        """Test case for service."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        service_type: Annotated[
            str | None,
            m.Field(
                description="Service type for factory-driven tests",
            ),
        ] = None
        input_value: Annotated[
            str | None, m.Field(description="Primary service input")
        ] = None
        user_id: Annotated[
            str | None, m.Field(description="User identifier for documented tests")
        ] = None
        expected_success: Annotated[
            bool,
            m.Field(
                description="Whether service call is expected to succeed",
            ),
        ] = True
        expected_error: Annotated[
            str | None,
            m.Field(
                description="Expected error substring for failure cases",
            ),
        ] = None
        description: Annotated[
            str, m.Field(description="Human-readable test case description")
        ] = ""
        extra_param: Annotated[
            int, m.Field(description="Auxiliary numeric parameter")
        ] = 3

    class RailwayTestCase(m.BaseModel):
        """Test case for railway pattern."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        user_ids: Annotated[
            t.StrSequence,
            m.Field(description="User identifiers used in pipeline"),
        ]
        operations: Annotated[
            t.StrSequence,
            m.Field(description="Pipeline operations to execute"),
        ] = m.Field(default_factory=tuple)
        expected_pipeline_length: Annotated[
            int, m.Field(description="Expected number of pipeline stages")
        ] = 1
        should_fail_at: Annotated[
            int | None,
            m.Field(
                description="Optional pipeline step expected to fail",
            ),
        ] = None
        description: Annotated[
            str,
            m.Field(
                description="Human-readable railway test case description",
            ),
        ] = ""

    class ValidationScenario(m.BaseModel):
        """Single scenario for validation testing."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        name: Annotated[str, m.Field(description="Unique scenario name")]
        validator_type: Annotated[
            str, m.Field(description="Validator category under test")
        ]
        input_value: Annotated[
            t.JsonValue | None,
            m.Field(description="Input value passed to validator"),
        ]
        input_params: Annotated[
            t.JsonPayload | None,
            m.Field(
                description="Optional validator parameters for scenario execution",
            ),
        ] = None
        should_succeed: Annotated[
            bool,
            m.Field(
                description="Whether scenario expects validation success",
            ),
        ] = True
        expected_value: Annotated[
            t.JsonValue | None,
            m.Field(
                description="Expected normalized value when validation succeeds",
            ),
        ] = None
        expected_error_contains: Annotated[
            str | None,
            m.Field(
                description="Expected error substring when validation fails",
            ),
        ] = None
        description: Annotated[
            str | None, m.Field(description="Human-readable scenario description")
        ] = None

    class Operation(m.BaseModel):
        """Generic operation progress model used by tests utilities."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        success_count: Annotated[int, m.Field(description="Successful operations")]
        failure_count: Annotated[int, m.Field(description="Failed operations")]
        skipped_count: Annotated[int, m.Field(description="Skipped operations")]
        metadata: Annotated[
            t.JsonMapping,
            m.Field(description="Additional operation metadata"),
        ] = m.Field(default_factory=dict)

    class Conversion(m.BaseModel):
        """Generic conversion progress model used by tests utilities."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        converted: Annotated[
            t.JsonList,
            m.Field(description="Converted records"),
        ] = m.Field(default_factory=list)
        errors: Annotated[
            t.StrSequence,
            m.Field(description="Conversion errors"),
        ] = m.Field(default_factory=tuple)
        warnings: Annotated[
            t.StrSequence,
            m.Field(description="Conversion warnings"),
        ] = m.Field(default_factory=tuple)
        skipped: Annotated[
            t.JsonList,
            m.Field(description="Skipped records"),
        ] = m.Field(default_factory=list)
        metadata: Annotated[
            t.JsonMapping,
            m.Field(description="Additional conversion metadata"),
        ] = m.Field(default_factory=dict)

    class ParserScenario(m.BaseModel):
        """Single scenario for parser testing."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        name: Annotated[str, m.Field(description="Unique parser scenario name")]
        parser_method: Annotated[str, m.Field(description="Parser method to execute")]
        input_data: Annotated[str, m.Field(description="Raw parser input data")]
        expected_output: Annotated[
            t.JsonValue | None,
            m.Field(
                description="Expected parsed output for successful scenarios",
            ),
        ] = None
        should_succeed: Annotated[
            bool,
            m.Field(
                description="Whether parser scenario expects success",
            ),
        ] = True
        error_contains: Annotated[
            str | None, m.Field(description="Expected parser error substring")
        ] = None
        description: Annotated[
            str | None, m.Field(description="Human-readable scenario description")
        ] = None

    class PublicParseCase(m.BaseModel):
        """Data-driven public parser contract scenario."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(
            frozen=True,
            arbitrary_types_allowed=True,
        )

        name: Annotated[str, m.Field(description="Unique scenario name")]
        input_value: Annotated[
            t.JsonPayload | None,
            m.Field(description="Public value passed to u.parse()"),
        ]
        target: Annotated[
            type[object],
            m.Field(description="Public target type passed to u.parse()"),
        ]
        options: Annotated[
            m.BaseModel | None, m.Field(description="Optional ParseOptions instance")
        ] = None
        should_succeed: Annotated[
            bool, m.Field(description="Whether parsing should succeed")
        ] = True
        expected_value: Annotated[
            t.JsonPayload | None,
            m.Field(description="Expected parsed scalar or enum value"),
        ] = None
        expected_data: Annotated[
            t.JsonMapping | None,
            m.Field(description="Expected parsed model_dump payload"),
        ] = None
        error_contains: Annotated[
            str | None, m.Field(description="Expected failure error substring")
        ] = None
        description: Annotated[
            str | None, m.Field(description="Human-readable scenario description")
        ] = None

    class ReliabilityScenario(m.BaseModel):
        """Single scenario for reliability testing (circuit breaker, retry)."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        name: Annotated[str, m.Field(description="Unique reliability scenario name")]
        strategy: Annotated[str, m.Field(description="Reliability strategy under test")]
        settings: Annotated[
            m.ConfigMap,
            m.Field(description="Reliability configuration payload"),
        ]
        simulate_failures: Annotated[
            int,
            m.Field(description="Number of failures to simulate"),
        ]
        expected_state: Annotated[
            str,
            m.Field(description="Expected strategy terminal state"),
        ]
        should_succeed: Annotated[
            bool,
            m.Field(
                description="Whether scenario expects successful outcome",
            ),
        ] = True
        description: Annotated[
            str | None, m.Field(description="Human-readable scenario description")
        ] = None

    class FalseSettings:
        app_name: str = "app"
        version: str = "1.0.0"
        enable_caching: bool = False
        timeout_seconds: float = 1.0
        dispatcher_auto_context: bool = False
        dispatcher_enable_logging: bool = False

        @classmethod
        def fetch_global(
            cls,
            *,
            overrides: t.ScalarMapping | None = None,
        ) -> Self:
            """Return a new instance for testing."""
            _ = overrides
            return cls()

        def model_copy(
            self,
            *,
            update: t.JsonMapping | None = None,
            deep: bool = False,
        ) -> Self:
            return self

        def model_dump(self) -> t.ScalarMapping:
            return dict[str, t.Scalar]()

    class Identifiers(m.BaseModel):
        """Test identifiers and IDs."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        user_id: Annotated[
            str,
            m.Field(
                description="Default test user identifier",
            ),
        ] = "test_user_123"
        session_id: Annotated[
            str,
            m.Field(
                description="Default test session identifier",
            ),
        ] = "test_session_123"
        service_name: Annotated[
            str,
            m.Field(
                description="Default test service name",
            ),
        ] = "test_service"
        operation_id: Annotated[
            str,
            m.Field(
                description="Default test operation identifier",
            ),
        ] = "test_operation"
        request_id: Annotated[
            str,
            m.Field(
                description="Default test request identifier",
            ),
        ] = "test-request-456"
        correlation_id: Annotated[
            str,
            m.Field(
                description="Default test correlation identifier",
            ),
        ] = "test-corr-123"

    class Names(m.BaseModel):
        """Test module and component names."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        module_name: Annotated[
            str,
            m.Field(
                description="Default test module name",
            ),
        ] = "test_module"
        handler_name: Annotated[
            str,
            m.Field(
                description="Default test handler name",
            ),
        ] = "test_handler"
        chain_name: Annotated[str, m.Field(description="Default test chain name")] = (
            "test_chain"
        )
        command_type: Annotated[
            str,
            m.Field(
                description="Default test command type",
            ),
        ] = "test_command"
        query_type: Annotated[str, m.Field(description="Default test query type")] = (
            "test_query"
        )
        logger_name: Annotated[
            str,
            m.Field(
                description="Default test logger name",
            ),
        ] = "test_logger"
        app_name: Annotated[
            str,
            m.Field(
                description="Default test application name",
            ),
        ] = "test-app"
        validation_app: Annotated[
            str,
            m.Field(
                description="Default validation test application name",
            ),
        ] = "validation-test"
        source_service: Annotated[
            str,
            m.Field(
                description="Default source service name",
            ),
        ] = "test_service"

    class ErrorData(m.BaseModel):
        """Test error codes and messages."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        error_code: Annotated[
            str,
            m.Field(
                description="Default test error code",
            ),
        ] = "TEST_ERROR_001"
        validation_error: Annotated[
            str,
            m.Field(
                description="Default validation error message",
            ),
        ] = "test_error"
        operation_error: Annotated[
            str,
            m.Field(
                description="Default operation error message",
            ),
        ] = "Op failed"
        settings_error: Annotated[
            str,
            m.Field(
                description="Default configuration error message",
            ),
        ] = "Settings failed"
        timeout_error: Annotated[
            str,
            m.Field(
                description="Default timeout error message",
            ),
        ] = "Operation timeout"

    class Data(m.BaseModel):
        """Test field names and data values."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        field_name: Annotated[str, m.Field(description="Default test field name")] = (
            "test_field"
        )
        config_key: Annotated[str, m.Field(description="Default test settings key")] = (
            "test_key"
        )
        username: Annotated[str, m.Field(description="Default test username")] = (
            "test_user"
        )
        email: Annotated[str, m.Field(description="Default test email")] = (
            "test@example.com"
        )
        password: Annotated[str, m.Field(description="Default test password")] = (
            "test_pass"
        )
        string_value: Annotated[
            str,
            m.Field(
                description="Default test string value",
            ),
        ] = "test_value"
        input_data: Annotated[str, m.Field(description="Default test input data")] = (
            "test_input"
        )
        request_data: Annotated[
            str,
            m.Field(
                description="Default test request data",
            ),
        ] = "test_request"
        result_data: Annotated[
            str,
            m.Field(
                description="Default test result data",
            ),
        ] = "test_result"
        message: Annotated[str, m.Field(description="Default test message")] = (
            "test_message"
        )

    class PatternData(m.BaseModel):
        """Test patterns and formats."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        slug_input: Annotated[
            str,
            m.Field(
                description="Input value for slug conversion tests",
            ),
        ] = "Test_String"
        slug_expected: Annotated[
            str,
            m.Field(
                description="Expected slug conversion output",
            ),
        ] = "test_string"
        uuid_format: Annotated[
            str,
            m.Field(
                description="Sample UUID format for tests",
            ),
        ] = "550e8400-e29b-41d4-a716-446655440000"

    class NumericValues(m.BaseModel):
        """Test port and numeric values."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        port: Annotated[int, m.Field(description="Default test port")] = 8080
        timeout: Annotated[int, m.Field(description="Default timeout in seconds")] = 30
        retry_count: Annotated[int, m.Field(description="Default retry count")] = 3
        batch_size: Annotated[int, m.Field(description="Default test batch size")] = 100

    # --- from test_container.py ---

    class ServiceScenario(m.BaseModel):
        """Test scenario for service registration and retrieval."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(
            frozen=True,
            arbitrary_types_allowed=True,
        )
        name: Annotated[str, m.Field(description="Service scenario name")]
        service: Annotated[
            t.Primitives, m.Field(description="Service value to register")
        ]
        description: Annotated[str, m.Field(description="Scenario description")] = ""

    class TypedRetrievalScenario(m.BaseModel):
        """Test scenario for typed service retrieval."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(
            frozen=True,
            arbitrary_types_allowed=True,
        )
        name: Annotated[str, m.Field(description="Typed retrieval scenario name")]
        service: Annotated[
            t.Primitives, m.Field(description="Registered service value")
        ]
        expected_type: Annotated[type, m.Field(description="Expected service type")]
        should_pass: Annotated[
            bool,
            m.Field(description="Whether typed retrieval should succeed"),
        ]
        description: Annotated[str, m.Field(description="Scenario description")] = ""

    class ContainerScenarios:
        """Centralized container test scenarios using c."""

        SERVICE_SCENARIOS: ClassVar[
            Sequence[TestsFlextModelsMixins.ServiceScenario]
        ] = []  # populated after class definition
        TYPED_RETRIEVAL_SCENARIOS: ClassVar[
            Sequence[TestsFlextModelsMixins.TypedRetrievalScenario]
        ] = []  # populated after class definition
        CONFIG_SCENARIOS: ClassVar[Sequence[t.ScalarMapping]] = [
            {"enable_singleton": False, "max_services": 8},
            {"invalid_key": "value", "another_invalid": 42},
            {},
        ]

    # --- from test_utilities_guards.py and test_utilities_guards_full_coverage.py ---

    class GuardSampleModel(m.BaseModel):
        """Sample model for guard testing."""

        name: str = "test"

    class NoModelDump:
        """Object without model_dump — should fail is_pydantic_model."""

    class LoggerLike(m.BaseModel):
        """Partial logger-like object for testing rejection by logger protocol check.

        Extends BaseModel to satisfy GuardInput typing. Intentionally omits
        required Logger protocol methods (name, bind, new, unbind, etc.) so that
        matches_type(instance, 'logger') returns False.
        """

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(
            arbitrary_types_allowed=True
        )

        def debug(self, *args: t.Scalar, **kwargs: t.Scalar) -> None:
            return None

        def info(self, *args: t.Scalar, **kwargs: t.Scalar) -> None:
            return None

        def warning(self, *args: t.Scalar, **kwargs: t.Scalar) -> None:
            return None

        def error(self, *args: t.Scalar, **kwargs: t.Scalar) -> None:
            return None

        def exception(self, *args: t.Scalar, **kwargs: t.Scalar) -> None:
            return None

    # --- from test_models_context_full_coverage.py ---

    class ModelWithNoCallableDump:
        """Model with non-callable model_dump attribute."""

        model_dump = "bad"

    # --- from test_utilities_mapper_full_coverage.py ---

    class PortModel(m.BaseModel):
        """Model with port/nested for mapper take/extract tests."""

        port: int = 0
        nested: Annotated[
            t.JsonMapping,
            m.Field(default_factory=dict),
        ]

    class MaybeModel(m.BaseModel):
        """Model with optional field for take tests."""

        x: str | None = None

    class GroupModel(m.BaseModel):
        """Model with optional kind for group tests."""

        kind: str | None = None

    class BadItems(UserDict[str, t.JsonValue]):
        """UserDict that explodes on items() for error-path testing."""

        @override
        def items(self) -> ItemsView[str, t.JsonValue]:
            """Items method."""
            msg = "bad items"
            raise RuntimeError(msg)

    class BadIter(UserList[str]):
        """UserList that explodes on __iter__ for error-path testing."""

        @override
        def __iter__(self) -> Iterator[str]:
            """__iter__ method."""
            msg = "bad iter"
            raise RuntimeError(msg)

    # --- from test_architectural_patterns.py ---

    class UserCreatedEvent(m.DomainEvent):
        """Domain event for user creation using FlextModels foundation."""

        user_id: Annotated[str, m.Field(description="Identifier of the created user.")]
        user_name: Annotated[str, m.Field(description="Name assigned to the new user.")]
        timestamp: Annotated[
            float,
            m.Field(description="POSIX timestamp when the event fired."),
        ]

    class UserUpdatedEvent(m.DomainEvent):
        """Domain event for user updates."""

        user_id: Annotated[str, m.Field(description="Identifier of the updated user.")]
        old_name: Annotated[str, m.Field(description="Previous user name.")]
        new_name: Annotated[str, m.Field(description="Updated user name.")]
        timestamp: Annotated[
            float,
            m.Field(description="POSIX timestamp when the event fired."),
        ]


# Populate ContainerScenarios after class is fully defined to allow forward references
_svc_scenarios: Sequence[TestsFlextModelsMixins.ServiceScenario] = [
    TestsFlextModelsMixins.ServiceScenario(
        name="test_service",
        service="test_service_value",
        description="Simple string service",
    ),
    TestsFlextModelsMixins.ServiceScenario(
        name="service_instance",
        service=42,
        description="Integer service instance",
    ),
    TestsFlextModelsMixins.ServiceScenario(
        name="string_service",
        service="test_value",
        description="String service",
    ),
]
TestsFlextModelsMixins.ContainerScenarios.SERVICE_SCENARIOS = _svc_scenarios
_typed_scenarios: Sequence[TestsFlextModelsMixins.TypedRetrievalScenario] = [
    TestsFlextModelsMixins.TypedRetrievalScenario(
        name="dict_service",
        service="test_dict_service",
        expected_type=str,
        should_pass=True,
        description="String service",
    ),
    TestsFlextModelsMixins.TypedRetrievalScenario(
        name="string_service",
        service="test_string",
        expected_type=str,
        should_pass=True,
        description="String service",
    ),
    TestsFlextModelsMixins.TypedRetrievalScenario(
        name="list_service",
        service=123,
        expected_type=int,
        should_pass=True,
        description="Integer service for typed retrieval",
    ),
]
TestsFlextModelsMixins.ContainerScenarios.TYPED_RETRIEVAL_SCENARIOS = _typed_scenarios
