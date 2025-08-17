from collections.abc import Callable as Callable
from datetime import datetime
from typing import TypedDict

import pytest
from _typeshed import Incomplete

from flext_core import FlextCommands, FlextResult

from ...conftest import (
    AssertHelpers as AssertHelpers,
    PerformanceMetrics as PerformanceMetrics,
    TestCase as TestCase,
    TestScenario as TestScenario,
)

class SampleCommandKwargs(TypedDict, total=False):
    name: str
    value: int
    command_id: str
    command_type: str
    user_id: str
    correlation_id: str
    timestamp: datetime

class SampleComplexCommandKwargs(TypedDict, total=False):
    email: str
    age: int
    command_id: str
    command_type: str
    user_id: str
    correlation_id: str
    timestamp: datetime

pytestmark: Incomplete
TCorrelationId = str
TEntityId = str
TResult = object
TServiceName = str
TUserId = str

class SampleCommand(FlextCommands.Command):
    name: str
    value: int
    def validate_command(self) -> FlextResult[None]: ...

class SampleCommandWithoutValidation(FlextCommands.Command):
    description: str

class SampleComplexCommand(FlextCommands.Command):
    email: str
    age: int
    def validate_command(self) -> FlextResult[None]: ...

class TestFlextCommandsAdvanced:
    @pytest.fixture
    def command_test_cases(self) -> list[TestCase]: ...
    @pytest.mark.parametrize_advanced
    def test_command_scenarios(
        self, command_test_cases: list[TestCase], assert_helpers: AssertHelpers
    ) -> None: ...
    @pytest.fixture
    def payload_conversion_cases(self) -> list[TestCase]: ...
    @pytest.mark.parametrize_advanced
    def test_payload_conversion_scenarios(
        self, payload_conversion_cases: list[TestCase]
    ) -> None: ...

class TestFlextCommandsComplexValidation:
    @pytest.fixture
    def complex_validation_cases(self) -> list[TestCase]: ...
    @pytest.mark.parametrize_advanced
    def test_complex_validation_scenarios(
        self, complex_validation_cases: list[TestCase]
    ) -> None: ...

class TestFlextCommandsImmutability:
    def test_command_immutability(self) -> None: ...

class TestFlextCommandsWithoutValidation:
    def test_command_without_validation(self) -> None: ...

class TestFlextCommandsPerformance:
    def test_command_creation_performance(
        self, performance_monitor: Callable[[Callable[[], object]], PerformanceMetrics]
    ) -> None: ...

class TestFlextCommandsFactoryMethods:
    def test_create_command_bus_factory(self) -> None: ...
    def test_create_simple_handler_factory(self) -> None: ...
    def test_command_bus_middleware_system(self) -> None: ...
    def test_command_handler_with_validation(self) -> None: ...
    def test_command_decorators_functionality(self) -> None: ...
    def test_command_result_with_metadata(self) -> None: ...
    def test_query_functionality(self) -> None: ...
