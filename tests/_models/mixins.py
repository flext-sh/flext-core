from __future__ import annotations

from collections import UserDict, UserList
from collections.abc import Callable, Iterator, Mapping, MutableSequence, Sequence
from typing import Annotated, ClassVar, Never, Self, override

from pydantic import BaseModel, ConfigDict, Field

from flext_tests import m
from tests import c, t

# Use `flext_tests` generic model base classes here to avoid a runtime circular import
# through the tests package. `t` comes from tests.typings, which extends the generic test
# typing definitions for flext-core.


class TestsFlextCoreModelsMixins:
    """flext-core test models namespace."""

    class BadDict(UserDict[str, t.Core.Tests.TestobjectSerializable]):
        """Dict that raises on get()."""

        @override
        def __getitem__(self, key: str) -> Never:
            """Raise error on get attempt."""
            _ = key
            msg = c.Core.Tests.TestErrors.BAD_DICT_GET
            raise RuntimeError(msg)

    class BadList(UserList[t.Core.Tests.TestobjectSerializable]):
        """List that raises on iteration."""

        @override
        def __iter__(self) -> Iterator[t.Core.Tests.TestobjectSerializable]:
            """Raise error on iteration."""
            msg = c.Core.Tests.TestErrors.BAD_LIST_ITERATION
            raise RuntimeError(msg)

    class BadModelDump:
        """Object with model_dump that raises."""

        model_dump: Callable[[], Mapping[str, t.Core.Tests.TestobjectSerializable]] = (
            staticmethod(
                lambda: (_ for _ in ()).throw(RuntimeError("Bad model_dump")),
            )
        )

    class _ValidationLikeError(Exception):
        """Validation-like error for tests."""

        def errors(self) -> Sequence[t.RecursiveContainerMapping]:
            return [{"loc": ["value"], "msg": "bad value"}]

    type TestCaseMap = Mapping[str, t.Core.Tests.TestobjectSerializable]
    type InputPayloadMap = Mapping[str, t.Core.Tests.TestobjectSerializable]

    class _MsgWithCommandId(BaseModel):
        command_id: str = "cmd-1"

    class _MsgWithMessageId(BaseModel):
        message_id: str = "msg-1"

    class SampleModel(BaseModel):
        """Sample model for testing."""

        name: str
        value: int

    class _SvcModel(BaseModel):
        value: str

    class _BrokenDumpModel(BaseModel):
        value: int = 1

        @override
        def __getattribute__(
            self,
            name: str,
        ) -> t.Core.Tests.PredicateSpec | None:
            if name == "model_dump":

                def _broken_dump(
                    _value: t.Core.Tests.Testobject = None,
                ) -> bool:
                    return True

                return _broken_dump
            return super().__getattribute__(name)

    class _ErrorsModel(BaseModel):
        value: int

        @classmethod
        @override
        def model_validate(
            cls,
            obj: t.RecursiveContainer,
            *,
            strict: bool | None = None,
            extra: str | None = None,
            from_attributes: bool | None = None,
            context: t.RecursiveContainerMapping | None = None,
            by_alias: bool | None = None,
            by_name: bool | None = None,
        ) -> Never:
            _ = strict, extra, from_attributes, context, by_alias, by_name
            _ = obj
            raise TestsFlextCoreModelsMixins._ValidationLikeError

    class _PlainErrorModel(BaseModel):
        value: int

        @classmethod
        @override
        def model_validate(
            cls,
            obj: t.RecursiveContainer,
            *,
            strict: bool | None = None,
            extra: str | None = None,
            from_attributes: bool | None = None,
            context: t.RecursiveContainerMapping | None = None,
            by_alias: bool | None = None,
            by_name: bool | None = None,
        ) -> Never:
            _ = strict, extra, from_attributes, context, by_alias, by_name
            _ = obj
            msg = c.Core.Tests.TestErrors.PLAIN_BOOM
            raise RuntimeError(msg)

    class _TargetModel(BaseModel):
        value: int

    class CacheTestModel(BaseModel):
        """Test model for cache key generation."""

        name: str
        value: int
        tags: MutableSequence[str] = []
        meta: t.MutableStrMapping = {}

    class NestedModel(BaseModel):
        """Nested Pydantic model for cache testing."""

        inner: TestsFlextCoreModelsMixins.CacheTestModel
        count: int

    class SettingsModelForTest(BaseModel):
        """Test configuration model (mutable for set_parameter tests)."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            validate_assignment=True,
            extra="forbid",
        )

        name: str = "default_settings"
        timeout: Annotated[int, Field(default=30, ge=0)] = 30
        enabled: bool = True

    class InvalidModelForTest(BaseModel):
        """Model with invalid model_dump."""

        value: str = "test"

    class SingletonClassForTest(BaseModel):
        """Test singleton class with Pydantic validation."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            validate_assignment=True,
            extra="forbid",
        )

        _instance: ClassVar[TestsFlextCoreModelsMixins.SingletonClassForTest | None] = (
            None
        )

        name: str = "default"
        timeout: int = 30

        @classmethod
        def fetch_global(
            cls,
        ) -> TestsFlextCoreModelsMixins.SingletonClassForTest:
            """Get global singleton instance."""
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

        @classmethod
        def reset_instance(cls) -> None:
            """Reset singleton instance for test isolation."""
            cls._instance = None

    class BadSettingsForTest(BaseModel):
        """Settings that fails to instantiate."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            validate_assignment=True,
        )

        def __init__(self, **kwargs: t.Scalar) -> None:
            """Raise error on init."""
            super().__init__(**kwargs)
            msg = c.Core.Tests.TestErrors.CANNOT_INSTANTIATE
            raise ValueError(msg)

    class _DumpErrorModel(BaseModel):
        value: int = 1

    class _Opts(BaseModel):
        value: int = 1

    class _FakeSettings(BaseModel):
        """Fake settings with model_copy support."""

        timeout: int = 10

        @property
        def data(self) -> t.RecursiveContainerMapping:
            return {"timeout": self.timeout}

    class _Model(BaseModel):
        value: int

    class _SampleEntity(BaseModel):
        """Test entity for domain utility tests."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=False)

        unique_id: str = "test-123"
        name: str = "test"

    class _FrozenEntity(BaseModel):
        """Frozen entity for immutability tests."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        unique_id: str = "frozen-1"

    class _GoodModel(BaseModel):
        value: int = 7

    class ComplexModel(BaseModel):
        """Complex test model."""

        id: int
        data: t.RecursiveContainerMapping
        items: t.StrSequence

    class _Cfg(BaseModel):
        x: int = 0
        y: str = "a"

    class _BadCopyModel(BaseModel):
        x: int = 1

    class User(BaseModel):
        """Shared user model for tests."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=False)

        id: str | None = None
        unique_id: str | None = None
        name: str
        email: str
        active: bool = True

    class EmailResponse(BaseModel):
        """Shared email response model for tests."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        status: str
        message_id: str

    class DomainTestEntity(m.Entity):
        """Test entity for domain tests."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=False)

        name: str
        value: t.ContainerValue

    class DomainTestValue(m.Value):
        """Test value t.RecursiveContainer for domain tests."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        data: str = ""
        count: int

    class CustomEntity:
        """Custom entity with settingsurable ID attribute."""

        def __init__(self, custom_id: str | None = None) -> None:
            """Initialize custom entity with ID."""
            self.custom_id = custom_id

    class SimpleValue:
        """Simple value t.RecursiveContainer without model_dump."""

        def __init__(self, data: str) -> None:
            """Initialize simple value t.RecursiveContainer."""
            self.data = data

    class ComplexValue:
        """TestsFlextModels.Value t.RecursiveContainer with non-hashable attributes."""

        def __init__(self, data: str, items: t.StrSequence) -> None:
            """Initialize complex value with non-hashable items."""
            self.data = data
            self.items = items  # list is not hashable

    class NoDict:
        """Object without __dict__, using __slots__."""

        __slots__ = ("value",)
        value: int

        def __init__(self, value: int) -> None:
            """Initialize t.RecursiveContainer without __dict__."""
            object.__setattr__(self, "value", value)

        @override
        def __repr__(self) -> str:
            """Return string representation."""
            return f"NoDict(value={self.value})"

    class MutableObj:
        """Mutable t.RecursiveContainer for immutability testing."""

        def __init__(self, value: int) -> None:
            """Initialize mutable t.RecursiveContainer."""
            self.value = value

    class ImmutableObj:
        """Immutable t.RecursiveContainer with custom __setattr__."""

        _frozen: bool = True

        def __init__(self, value: int) -> None:
            """Initialize immutable t.RecursiveContainer."""
            object.__setattr__(self, "value", value)

        @override
        def __setattr__(self, name: str, value: t.ContainerValue) -> None:
            """Prevent attribute setting if frozen."""
            if self._frozen:
                msg = c.Core.Tests.TestErrors.OBJECT_IS_FROZEN
                raise AttributeError(msg)
            object.__setattr__(self, name, value)

    class NoSettingsNoSetattr:
        """Object without model_config or __setattr__."""

    class NoSetattr:
        """Object without __setattr__."""

    class ParseOptions(BaseModel):
        """Test-local parse options after production model removal."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        strip: bool = True
        remove_empty: bool = True
        validator: Callable[[str], bool] | None = None

    class ParseDelimitedCase(BaseModel):
        """Test case for parse_delimited method."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            frozen=True,
            arbitrary_types_allowed=True,
        )

        text: str
        delimiter: str
        expected: t.StrSequence | None = None
        expected_error: str | None = None
        options: BaseModel | None = None
        strip: bool = True
        remove_empty: bool = True
        validator: Callable[[str], bool] | None = None
        use_legacy: bool = False
        description: Annotated[str, Field(default="", exclude=True)] = ""

    class SplitEscapeCase(BaseModel):
        """Test case for split_on_char_with_escape method."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        text: str
        split_char: str
        escape_char: str = "\\"
        expected: t.StrSequence | None = None
        expected_error: str | None = None
        description: Annotated[str, Field(default="", exclude=True)] = ""

    class NormalizeWhitespaceCase(BaseModel):
        """Test case for normalize_whitespace method."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        text: str
        pattern: str = r"\s+"
        replacement: str = " "
        expected: str | None = None
        expected_error: str | None = None
        description: Annotated[str, Field(default="", exclude=True)] = ""

    class RegexPipelineCase(BaseModel):
        """Test case for apply_regex_pipeline method."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        text: str
        patterns: Sequence[tuple[str, str] | tuple[str, str, int]]
        expected: str | None = None
        expected_error: str | None = None
        description: Annotated[str, Field(default="", exclude=True)] = ""

    class ObjectKeyCase(BaseModel):
        """Test case for get_object_key method."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            frozen=True,
            arbitrary_types_allowed=True,
        )

        obj: t.ContainerValue
        expected_contains: t.StrSequence | None = None
        expected_exact: str | None = None
        description: Annotated[str, Field(default="", exclude=True)] = ""

    class AutomatedTestScenario(BaseModel):
        """Pydantic v2 model for automated test scenarios."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        description: str
        input: t.ContainerValue
        expected_success: bool

    class StandardTestCaseModel(BaseModel):
        """Standard operation case model for shared test utilities."""

        description: str
        input_data: t.ContainerValue
        expected_result: t.ContainerValue
        expected_success: bool = True
        error_contains: str | None = None

    class UtilityEntityModel(m.Entity):
        """Shared entity model for generic test fixtures."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=False)

        name: str
        value: t.ContainerValue

    class UtilityValueModel(m.Value):
        """Shared value model for generic test fixtures."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        value: t.ContainerValue

    class MockScenarioData(BaseModel):
        """Mock scenario test data."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        given: Mapping[str, t.Primitives]
        when: Mapping[str, t.Primitives]
        then: Mapping[str, t.Primitives]
        tags: t.StrSequence
        priority: str

    class NestedDataDict(BaseModel):
        """Nested test data."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        key: str
        value: t.Primitives
        metadata: str

    class FixtureDataDict(BaseModel):
        """Test data for FlextTestBuilder."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

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
            TestsFlextCoreModelsMixins.NestedDataDict,
        ]

    class FixtureCaseDict(BaseModel):
        """Individual test case configuration."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        email: str
        input: str

    class SuccessCaseDict(BaseModel):
        """Success test case."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        email: str
        input: str

    class FailureCaseDict(BaseModel):
        """Failure test case."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        email: str
        input: str

    class SetupDataDict(BaseModel):
        """Setup data for test suite."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        initialization_step: str
        configuration_key: str
        configuration_value: str
        environment: str

    class FixtureSuiteDict(BaseModel):
        """Test suite configuration."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        suite_name: str
        scenario_count: int
        tags: t.StrSequence
        setup_data: Mapping[
            str,
            TestsFlextCoreModelsMixins.SetupDataDict,
        ]

    class UserDataFixtureDict(BaseModel):
        """User fixture data."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        username: str
        email: str
        status: str

    class RequestDataFixtureDict(BaseModel):
        """Request fixture data."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        method: str
        path: str
        headers: t.StrMapping

    class FixtureFixturesDict(BaseModel):
        """Test fixtures configuration."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        user: Mapping[
            str,
            TestsFlextCoreModelsMixins.UserDataFixtureDict,
        ]
        request: Mapping[
            str,
            TestsFlextCoreModelsMixins.RequestDataFixtureDict,
        ]

    class UserProfileDict(BaseModel):
        """User profile for property-based testing."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        id: str
        name: str
        email: str

    class SettingsTestCaseDict(BaseModel):
        """Configuration test case."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        domain: str
        port: int
        timeout: float
        debug: bool

    class PerformanceMetricsDict(BaseModel):
        """Performance metrics from testing."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        total_operations: int
        time_elapsed: float
        ops_per_second: float
        memory_peak_mb: float

    class StressTestResultDict(BaseModel):
        """Result from stress testing."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        iterations: int
        success_count: int
        failure_count: int
        average_time_ms: float

    class AsyncPayloadDict(BaseModel):
        """Async event payload."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        data: str
        status: str

    class AsyncTestDataDict(BaseModel):
        """Async test data."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        event_type: str
        timestamp: str
        payload: Mapping[
            str,
            TestsFlextCoreModelsMixins.AsyncPayloadDict,
        ]

    class UserPayloadDict(BaseModel):
        """User command payload."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        username: str
        email: str

    class UpdateFieldDict(BaseModel):
        """Individual update field."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        field_name: str
        new_value: t.Primitives

    class UpdatePayloadDict(BaseModel):
        """Update command payload."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        target_user_id: str
        updates: Mapping[
            str,
            TestsFlextCoreModelsMixins.UpdateFieldDict,
        ]

    class UserDataDict(BaseModel):
        """User data response."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        id: str
        username: str
        email: str
        status: str

    class UpdateResultDict(BaseModel):
        """Update operation result."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        user_id: str
        updated_fields: t.StrSequence
        update_count: int

    class CommandPayloadDict(BaseModel):
        """Generic command payload."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        id: str = ""
        username: str = ""
        email: str = ""

    class ServiceTestCase(BaseModel):
        """Test case for service."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        service_type: Annotated[
            str | None,
            Field(
                default=None,
                description="Service type for factory-driven tests",
            ),
        ] = None
        input_value: Annotated[
            str | None,
            Field(default=None, description="Primary service input"),
        ] = None
        user_id: Annotated[
            str | None,
            Field(default=None, description="User identifier for documented tests"),
        ] = None
        expected_success: Annotated[
            bool,
            Field(
                default=True,
                description="Whether service call is expected to succeed",
            ),
        ] = True
        expected_error: Annotated[
            str | None,
            Field(
                default=None,
                description="Expected error substring for failure cases",
            ),
        ] = None
        description: Annotated[
            str,
            Field(default="", description="Human-readable test case description"),
        ] = ""
        extra_param: Annotated[
            int,
            Field(default=3, description="Auxiliary numeric parameter"),
        ] = 3

    class RailwayTestCase(BaseModel):
        """Test case for railway pattern."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        user_ids: Annotated[
            t.StrSequence,
            Field(description="User identifiers used in pipeline"),
        ]
        operations: Annotated[
            t.StrSequence,
            Field(description="Pipeline operations to execute"),
        ] = Field(default_factory=list)
        expected_pipeline_length: Annotated[
            int,
            Field(default=1, description="Expected number of pipeline stages"),
        ] = 1
        should_fail_at: Annotated[
            int | None,
            Field(
                default=None,
                description="Optional pipeline step expected to fail",
            ),
        ] = None
        description: Annotated[
            str,
            Field(
                default="",
                description="Human-readable railway test case description",
            ),
        ] = ""

    class ValidationScenario(BaseModel):
        """Single scenario for validation testing."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        name: Annotated[str, Field(description="Unique scenario name")]
        validator_type: Annotated[
            str, Field(description="Validator category under test")
        ]
        input_value: Annotated[
            t.RecursiveContainer,
            Field(description="Input value passed to validator"),
        ]
        input_params: Annotated[
            t.RecursiveContainer | None,
            Field(
                default=None,
                description="Optional validator parameters for scenario execution",
            ),
        ] = None
        should_succeed: Annotated[
            bool,
            Field(
                default=True,
                description="Whether scenario expects validation success",
            ),
        ] = True
        expected_value: Annotated[
            t.RecursiveContainer | None,
            Field(
                default=None,
                description="Expected normalized value when validation succeeds",
            ),
        ] = None
        expected_error_contains: Annotated[
            str | None,
            Field(
                default=None,
                description="Expected error substring when validation fails",
            ),
        ] = None
        description: Annotated[
            str | None,
            Field(default=None, description="Human-readable scenario description"),
        ] = None

    class ParserScenario(BaseModel):
        """Single scenario for parser testing."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        name: Annotated[str, Field(description="Unique parser scenario name")]
        parser_method: Annotated[str, Field(description="Parser method to execute")]
        input_data: Annotated[str, Field(description="Raw parser input data")]
        expected_output: Annotated[
            t.RecursiveContainer | None,
            Field(
                default=None,
                description="Expected parsed output for successful scenarios",
            ),
        ] = None
        should_succeed: Annotated[
            bool,
            Field(
                default=True,
                description="Whether parser scenario expects success",
            ),
        ] = True
        error_contains: Annotated[
            str | None,
            Field(default=None, description="Expected parser error substring"),
        ] = None
        description: Annotated[
            str | None,
            Field(default=None, description="Human-readable scenario description"),
        ] = None

    class ReliabilityScenario(BaseModel):
        """Single scenario for reliability testing (circuit breaker, retry)."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        name: Annotated[str, Field(description="Unique reliability scenario name")]
        strategy: Annotated[str, Field(description="Reliability strategy under test")]
        settings: Annotated[
            t.ConfigMap,
            Field(description="Reliability configuration payload"),
        ]
        simulate_failures: Annotated[
            int,
            Field(description="Number of failures to simulate"),
        ]
        expected_state: Annotated[
            str,
            Field(description="Expected strategy terminal state"),
        ]
        should_succeed: Annotated[
            bool,
            Field(
                default=True,
                description="Whether scenario expects successful outcome",
            ),
        ] = True
        description: Annotated[
            str | None,
            Field(default=None, description="Human-readable scenario description"),
        ] = None

    class FalseSettings:
        app_name: str = "app"
        version: str = "1.0.0"
        enable_caching: bool = False
        timeout_seconds: float = 1.0
        dispatcher_auto_context: bool = False
        dispatcher_enable_logging: bool = False

        def model_copy(
            self,
            *,
            update: Mapping[str, t.Container] | None = None,
            deep: bool = False,
        ) -> Self:
            return self

        def model_dump(self) -> t.ScalarMapping:
            return dict[str, t.Scalar]()

    class Identifiers(BaseModel):
        """Test identifiers and IDs."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        user_id: Annotated[
            str,
            Field(
                default="test_user_123",
                description="Default test user identifier",
            ),
        ] = "test_user_123"
        session_id: Annotated[
            str,
            Field(
                default="test_session_123",
                description="Default test session identifier",
            ),
        ] = "test_session_123"
        service_name: Annotated[
            str,
            Field(
                default="test_service",
                description="Default test service name",
            ),
        ] = "test_service"
        operation_id: Annotated[
            str,
            Field(
                default="test_operation",
                description="Default test operation identifier",
            ),
        ] = "test_operation"
        request_id: Annotated[
            str,
            Field(
                default="test-request-456",
                description="Default test request identifier",
            ),
        ] = "test-request-456"
        correlation_id: Annotated[
            str,
            Field(
                default="test-corr-123",
                description="Default test correlation identifier",
            ),
        ] = "test-corr-123"

    class Names(BaseModel):
        """Test module and component names."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        module_name: Annotated[
            str,
            Field(
                default="test_module",
                description="Default test module name",
            ),
        ] = "test_module"
        handler_name: Annotated[
            str,
            Field(
                default="test_handler",
                description="Default test handler name",
            ),
        ] = "test_handler"
        chain_name: Annotated[
            str,
            Field(default="test_chain", description="Default test chain name"),
        ] = "test_chain"
        command_type: Annotated[
            str,
            Field(
                default="test_command",
                description="Default test command type",
            ),
        ] = "test_command"
        query_type: Annotated[
            str,
            Field(default="test_query", description="Default test query type"),
        ] = "test_query"
        logger_name: Annotated[
            str,
            Field(
                default="test_logger",
                description="Default test logger name",
            ),
        ] = "test_logger"
        app_name: Annotated[
            str,
            Field(
                default="test-app",
                description="Default test application name",
            ),
        ] = "test-app"
        validation_app: Annotated[
            str,
            Field(
                default="validation-test",
                description="Default validation test application name",
            ),
        ] = "validation-test"
        source_service: Annotated[
            str,
            Field(
                default="test_service",
                description="Default source service name",
            ),
        ] = "test_service"

    class ErrorData(BaseModel):
        """Test error codes and messages."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        error_code: Annotated[
            str,
            Field(
                default="TEST_ERROR_001",
                description="Default test error code",
            ),
        ] = "TEST_ERROR_001"
        validation_error: Annotated[
            str,
            Field(
                default="test_error",
                description="Default validation error message",
            ),
        ] = "test_error"
        operation_error: Annotated[
            str,
            Field(
                default="Op failed",
                description="Default operation error message",
            ),
        ] = "Op failed"
        settings_error: Annotated[
            str,
            Field(
                default="Settings failed",
                description="Default configuration error message",
            ),
        ] = "Settings failed"
        timeout_error: Annotated[
            str,
            Field(
                default="Operation timeout",
                description="Default timeout error message",
            ),
        ] = "Operation timeout"

    class Data(BaseModel):
        """Test field names and data values."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        field_name: Annotated[
            str,
            Field(default="test_field", description="Default test field name"),
        ] = "test_field"
        config_key: Annotated[
            str,
            Field(default="test_key", description="Default test settings key"),
        ] = "test_key"
        username: Annotated[
            str,
            Field(default="test_user", description="Default test username"),
        ] = "test_user"
        email: Annotated[
            str,
            Field(default="test@example.com", description="Default test email"),
        ] = "test@example.com"
        password: Annotated[
            str,
            Field(default="test_pass", description="Default test password"),
        ] = "test_pass"
        string_value: Annotated[
            str,
            Field(
                default="test_value",
                description="Default test string value",
            ),
        ] = "test_value"
        input_data: Annotated[
            str,
            Field(default="test_input", description="Default test input data"),
        ] = "test_input"
        request_data: Annotated[
            str,
            Field(
                default="test_request",
                description="Default test request data",
            ),
        ] = "test_request"
        result_data: Annotated[
            str,
            Field(
                default="test_result",
                description="Default test result data",
            ),
        ] = "test_result"
        message: Annotated[
            str,
            Field(default="test_message", description="Default test message"),
        ] = "test_message"

    class PatternData(BaseModel):
        """Test patterns and formats."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        slug_input: Annotated[
            str,
            Field(
                default="Test_String",
                description="Input value for slug conversion tests",
            ),
        ] = "Test_String"
        slug_expected: Annotated[
            str,
            Field(
                default="test_string",
                description="Expected slug conversion output",
            ),
        ] = "test_string"
        uuid_format: Annotated[
            str,
            Field(
                default="550e8400-e29b-41d4-a716-446655440000",
                description="Sample UUID format for tests",
            ),
        ] = "550e8400-e29b-41d4-a716-446655440000"

    class NumericValues(BaseModel):
        """Test port and numeric values."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        port: Annotated[
            int,
            Field(default=8080, description="Default test port"),
        ] = 8080
        timeout: Annotated[
            int,
            Field(default=30, description="Default timeout in seconds"),
        ] = 30
        retry_count: Annotated[
            int,
            Field(default=3, description="Default retry count"),
        ] = 3
        batch_size: Annotated[
            int,
            Field(default=100, description="Default test batch size"),
        ] = 100
