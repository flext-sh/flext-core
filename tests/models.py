"""Models for flext-core tests.

Provides TestsFlextCoreModels using composition with TestsFlextModels and TestsFlextModels.
All generic test models come from flext_tests.

Architecture:
- TestsFlextModels (flext_tests) = Generic models for all FLEXT projects
- TestsFlextModels (flext_core) = Core domain models
- TestsFlextCoreModels (tests/) = flext-core-specific models using composition

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections import UserDict, UserList
from collections.abc import Callable, Iterator, Mapping, MutableSequence, Sequence
from typing import Annotated, ClassVar, Never, override

from pydantic import BaseModel, ConfigDict, Field

from flext_tests import m
from tests import e, r, t, u


class TestsFlextCoreModels(m):
    """Models for flext-core tests - uses composition with TestsFlextModels.

    Architecture: Uses composition (not inheritance) with TestsFlextModels and TestsFlextModels
    for flext-core-specific model definitions.

    Access patterns:
    - TestsFlextCoreModels.Tests.* = flext_tests test models (via inheritance)
    - TestsFlextCoreModels.Core.Tests.* = flext-core-specific test models
    - TestsFlextCoreModels.Entity, .Value, etc. = TestsFlextModels domain models (via inheritance)

    Rules:
    - flext-core-specific models go in Core namespace
    - Generic models accessed via TestsFlextModels.Tests namespace
    """

    class Core:
        """flext-core-specific test models namespace."""

        class Tests:
            """flext-core test models namespace."""

            class BadDict(UserDict[str, t.Core.Tests.TestobjectSerializable]):
                """Dict that raises on get()."""

                @override
                def __getitem__(self, key: str) -> Never:
                    """Raise error on get attempt."""
                    _ = key
                    msg = "Bad dict get"
                    raise RuntimeError(msg)

            class BadList(UserList[t.Core.Tests.TestobjectSerializable]):
                """List that raises on iteration."""

                @override
                def __iter__(self) -> Iterator[t.Core.Tests.TestobjectSerializable]:
                    """Raise error on iteration."""
                    msg = "Bad list iteration"
                    raise RuntimeError(msg)

            class BadModelDump:
                """Object with model_dump that raises."""

                model_dump: Callable[
                    [], Mapping[str, t.Core.Tests.TestobjectSerializable]
                ] = staticmethod(
                    lambda: (_ for _ in ()).throw(RuntimeError("Bad model_dump")),
                )

            class _ValidationLikeError(Exception):
                """Validation-like error for tests."""

                def errors(self) -> Sequence[t.ContainerMapping]:
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
                    obj: t.NormalizedValue,
                    *,
                    strict: bool | None = None,
                    extra: str | None = None,
                    from_attributes: bool | None = None,
                    context: t.ContainerMapping | None = None,
                    by_alias: bool | None = None,
                    by_name: bool | None = None,
                ) -> Never:
                    _ = strict, extra, from_attributes, context, by_alias, by_name
                    _ = obj
                    raise TestsFlextCoreModels.Core.Tests._ValidationLikeError

            class _PlainErrorModel(BaseModel):
                value: int

                @classmethod
                @override
                def model_validate(
                    cls,
                    obj: t.NormalizedValue,
                    *,
                    strict: bool | None = None,
                    extra: str | None = None,
                    from_attributes: bool | None = None,
                    context: t.ContainerMapping | None = None,
                    by_alias: bool | None = None,
                    by_name: bool | None = None,
                ) -> Never:
                    _ = strict, extra, from_attributes, context, by_alias, by_name
                    _ = obj
                    msg = "plain boom"
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

                inner: TestsFlextCoreModels.Core.Tests.CacheTestModel
                count: int

            class ConfigModelForTest(BaseModel):
                """Test configuration model (mutable for set_parameter tests)."""

                model_config: ClassVar[ConfigDict] = ConfigDict(
                    validate_assignment=True,
                    extra="forbid",
                )

                name: str = "default_config"
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

                _instance: ClassVar[
                    TestsFlextCoreModels.Core.Tests.SingletonClassForTest | None
                ] = None

                name: str = "default"
                timeout: int = 30

                @classmethod
                def get_global(
                    cls,
                ) -> TestsFlextCoreModels.Core.Tests.SingletonClassForTest:
                    """Get global singleton instance."""
                    if cls._instance is None:
                        cls._instance = cls()
                    return cls._instance

                @classmethod
                def reset_instance(cls) -> None:
                    """Reset singleton instance for test isolation."""
                    cls._instance = None

            class BadConfigForTest(BaseModel):
                """Config that fails to instantiate."""

                model_config: ClassVar[ConfigDict] = ConfigDict(
                    validate_assignment=True,
                )

                def __init__(self, **kwargs: t.Scalar) -> None:
                    """Raise error on init."""
                    super().__init__(**kwargs)
                    msg = "Cannot instantiate"
                    raise ValueError(msg)

            class _DumpErrorModel(BaseModel):
                value: int = 1

            class _Opts(BaseModel):
                value: int = 1

            class _FakeConfig(BaseModel):
                """Fake config with model_copy support."""

                timeout: int = 10

                @property
                def data(self) -> t.ContainerMapping:
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
                data: t.ContainerMapping
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
                """Test value t.NormalizedValue for domain tests."""

                model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

                data: str = ""
                count: int

            class CustomEntity:
                """Custom entity with configurable ID attribute."""

                def __init__(self, custom_id: str | None = None) -> None:
                    """Initialize custom entity with ID."""
                    self.custom_id = custom_id

            class SimpleValue:
                """Simple value t.NormalizedValue without model_dump."""

                def __init__(self, data: str) -> None:
                    """Initialize simple value t.NormalizedValue."""
                    self.data = data

            class ComplexValue:
                """TestsFlextModels.Value t.NormalizedValue with non-hashable attributes."""

                def __init__(self, data: str, items: t.StrSequence) -> None:
                    """Initialize complex value with non-hashable items."""
                    self.data = data
                    self.items = items  # list is not hashable

            class NoDict:
                """Object without __dict__, using __slots__."""

                __slots__ = ("value",)

                def __init__(self, value: int) -> None:
                    """Initialize t.NormalizedValue without __dict__."""
                    object.__setattr__(self, "value", value)

                @override
                def __repr__(self) -> str:
                    """Return string representation."""
                    return f"NoDict({getattr(self, 'value', None)})"

            class MutableObj:
                """Mutable t.NormalizedValue for immutability testing."""

                def __init__(self, value: int) -> None:
                    """Initialize mutable t.NormalizedValue."""
                    self.value = value

            class ImmutableObj:
                """Immutable t.NormalizedValue with custom __setattr__."""

                _frozen: bool = True

                def __init__(self, value: int) -> None:
                    """Initialize immutable t.NormalizedValue."""
                    object.__setattr__(self, "value", value)

                @override
                def __setattr__(self, name: str, value: t.ContainerValue) -> None:
                    """Prevent attribute setting if frozen."""
                    if self._frozen:
                        msg = "Object is frozen"
                        raise AttributeError(msg)
                    object.__setattr__(self, name, value)

            class NoConfigNoSetattr:
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
                    TestsFlextCoreModels.Core.Tests.NestedDataDict,
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
                    TestsFlextCoreModels.Core.Tests.SetupDataDict,
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
                    TestsFlextCoreModels.Core.Tests.UserDataFixtureDict,
                ]
                request: Mapping[
                    str,
                    TestsFlextCoreModels.Core.Tests.RequestDataFixtureDict,
                ]

            class UserProfileDict(BaseModel):
                """User profile for property-based testing."""

                model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

                id: str
                name: str
                email: str

            class ConfigTestCaseDict(BaseModel):
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
                    TestsFlextCoreModels.Core.Tests.AsyncPayloadDict,
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
                    TestsFlextCoreModels.Core.Tests.UpdateFieldDict,
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
                    Field(
                        default=None, description="User identifier for documented tests"
                    ),
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
                    Field(
                        default="", description="Human-readable test case description"
                    ),
                ] = ""
                extra_param: Annotated[
                    int,
                    Field(default=3, description="Auxiliary numeric parameter"),
                ] = 3

                def create_user_service(self) -> u.Core.Tests.GetUserService:
                    return u.Core.Tests.make(
                        u.Core.Tests.GetUserService,
                        user_id=self.user_id or self.input_value or "",
                    )

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

                def execute_v1_pipeline(
                    self,
                ) -> r[
                    str
                    | TestsFlextCoreModels.Core.Tests.User
                    | TestsFlextCoreModels.Core.Tests.EmailResponse
                ]:
                    if not self.user_ids:
                        return r[
                            str
                            | TestsFlextCoreModels.Core.Tests.User
                            | TestsFlextCoreModels.Core.Tests.EmailResponse
                        ].fail(
                            "No user IDs provided",
                        )
                    user_result: r[TestsFlextCoreModels.Core.Tests.User] = (
                        u.Core.Tests.make(
                            u.Core.Tests.GetUserService,
                            user_id=self.user_ids[0],
                        ).execute()
                    )
                    result: r[
                        TestsFlextCoreModels.Core.Tests.User
                        | str
                        | TestsFlextCoreModels.Core.Tests.EmailResponse
                    ] = user_result.map(
                        lambda user: user,
                    )
                    for op in self.operations:
                        if op == "get_email":
                            result = result.map(
                                lambda user: (
                                    user.email
                                    if isinstance(
                                        user,
                                        TestsFlextCoreModels.Core.Tests.User,
                                    )
                                    else str(user)
                                ),
                            )
                        elif op == "send_email":
                            email_result: r[
                                TestsFlextCoreModels.Core.Tests.EmailResponse
                            ] = result.flat_map(
                                lambda email: u.Core.Tests.make(
                                    u.Core.Tests.SendEmailService,
                                    to=str(email),
                                    subject="Test",
                                ).execute(),
                            )
                            result = email_result.map(lambda response: response)
                        elif op == "get_status":
                            result = result.map(
                                lambda response: (
                                    response.status
                                    if isinstance(
                                        response,
                                        TestsFlextCoreModels.Core.Tests.EmailResponse,
                                    )
                                    else str(response)
                                ),
                            )
                    return result

                def execute_v2_pipeline(
                    self,
                ) -> TestsFlextCoreModels.Core.Tests.User | str:
                    if not self.user_ids:
                        msg = "No user IDs provided"
                        raise e.BaseError(msg)
                    user_result = u.Core.Tests.make(
                        u.Core.Tests.GetUserService,
                        user_id=self.user_ids[0],
                    ).result
                    user: TestsFlextCoreModels.Core.Tests.User | str = user_result
                    for op in self.operations:
                        if op == "get_email":
                            user = (
                                user.email
                                if isinstance(
                                    user, TestsFlextCoreModels.Core.Tests.User
                                )
                                else str(user)
                            )
                        elif op == "send_email":
                            email_to = str(user) if not isinstance(user, str) else user
                            response_obj: TestsFlextCoreModels.Core.Tests.EmailResponse = u.Core.Tests.make(
                                u.Core.Tests.SendEmailService,
                                to=email_to,
                                subject="Test",
                            ).result
                            user = response_obj.status
                    return user

            class ValidationScenario(BaseModel):
                """Single scenario for validation testing."""

                model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

                name: Annotated[str, Field(description="Unique scenario name")]
                validator_type: Annotated[
                    str, Field(description="Validator category under test")
                ]
                input_value: Annotated[
                    t.NormalizedValue,
                    Field(description="Input value passed to validator"),
                ]
                input_params: Annotated[
                    t.NormalizedValue | None,
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
                    t.NormalizedValue | None,
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
                    Field(
                        default=None, description="Human-readable scenario description"
                    ),
                ] = None

            class ParserScenario(BaseModel):
                """Single scenario for parser testing."""

                model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

                name: Annotated[str, Field(description="Unique parser scenario name")]
                parser_method: Annotated[
                    str, Field(description="Parser method to execute")
                ]
                input_data: Annotated[str, Field(description="Raw parser input data")]
                expected_output: Annotated[
                    t.NormalizedValue | None,
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
                    Field(
                        default=None, description="Human-readable scenario description"
                    ),
                ] = None

            class ReliabilityScenario(BaseModel):
                """Single scenario for reliability testing (circuit breaker, retry)."""

                model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

                name: Annotated[
                    str, Field(description="Unique reliability scenario name")
                ]
                strategy: Annotated[
                    str, Field(description="Reliability strategy under test")
                ]
                config: Annotated[
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
                    Field(
                        default=None, description="Human-readable scenario description"
                    ),
                ] = None


m = TestsFlextCoreModels

__all__ = [
    "TestsFlextCoreModels",
    "m",
]
