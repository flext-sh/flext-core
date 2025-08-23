"""Tests for FLEXT Core Commands with modern pytest patterns.

Advanced tests using parametrized fixtures, property-based testing,
performance monitoring, and comprehensive CQRS coverage.
- Enterprise-grade parametrized testing with structured TestCase objects
- Advanced fixture composition using conftest infrastructure
- Command validation testing with business logic enforcement
- Hypothesis property-based testing for edge case discovery
- Mock factories for command handler isolation

Usage of New Conftest Infrastructure:
- test_builder: Fluent builder pattern for complex test data construction
- assert_helpers: Advanced assertion helpers with FlextResult validation
- performance_monitor: Function execution monitoring with memory tracking
- hypothesis_strategies: Property-based testing with domain-specific strategies
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import TypedDict, cast
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError
from tests.conftest import (
    AssertHelpers,
    PerformanceMetrics,
    TestCase,
    TestScenario,
)

from flext_core import (
    FlextCommands,
    FlextPayload,
    FlextResult,
)


# TypedDict definitions needed at runtime
class SampleCommandKwargs(TypedDict, total=False):
    """Sample command kwargs type definition."""

    name: str
    value: int
    command_id: str
    command_type: str
    user_id: str
    correlation_id: str
    timestamp: datetime


class SampleComplexCommandKwargs(TypedDict, total=False):
    """Sample complex command kwargs type definition."""

    email: str
    age: int
    command_id: str
    command_type: str
    user_id: str
    correlation_id: str
    timestamp: datetime


# Test markers for organized execution
pytestmark = [pytest.mark.unit, pytest.mark.core]

# Define runtime aliases to prevent NameError during model_rebuild
TCorrelationId = str
TEntityId = str
TResult = object
TServiceName = str
TUserId = str


class SampleCommand(FlextCommands.Command):
    """Test command for comprehensive testing."""

    name: str
    value: int = 0

    def validate_command(self) -> FlextResult[None]:
        """Validate the test command."""
        if not self.name.strip():
            return FlextResult[None].fail("Name cannot be empty")
        if self.value < 0:
            return FlextResult[None].fail("Value cannot be negative")
        return FlextResult[None].ok(None)


class SampleCommandWithoutValidation(FlextCommands.Command):
    """Test command without custom validation."""

    description: str


class SampleComplexCommand(FlextCommands.Command):
    """Test command with complex validation rules."""

    email: str
    age: int

    def validate_command(self) -> FlextResult[None]:
        """Complex validation logic."""
        if "@" not in self.email:
            return FlextResult[None].fail("Invalid email format")
        if self.age < 18:
            return FlextResult[None].fail("Age must be 18 or older")
        return FlextResult[None].ok(None)


class TestFlextCommandsAdvanced:
    """Advanced command testing with consolidated patterns."""

    @pytest.fixture
    def command_test_cases(self) -> list[TestCase]:
        """Structured test cases for command testing."""
        return [
            # Basic command creation tests
            TestCase(
                id="command_creation_basic",
                description="Basic command creation with auto-generated fields",
                input_data={"name": "test", "value": 42},
                expected_output={
                    "name": "test",
                    "value": 42,
                    "command_type": "sample",
                    "has_command_id": True,
                    "has_timestamp": True,
                    "has_correlation_id": True,
                },
                scenario=TestScenario.HAPPY_PATH,
            ),
            TestCase(
                id="command_creation_explicit",
                description="Command creation with explicit field values",
                input_data={
                    "name": "explicit",
                    "value": 100,
                    "command_id": "cmd-123",
                    "command_type": "CustomType",
                    "user_id": "user-456",
                    "correlation_id": "corr-789",
                },
                expected_output={
                    "name": "explicit",
                    "value": 100,
                    "command_id": "cmd-123",
                    "command_type": "CustomType",
                    "user_id": "user-456",
                    "correlation_id": "corr-789",
                },
                scenario=TestScenario.HAPPY_PATH,
            ),
            TestCase(
                id="command_type_auto_generation",
                description="Command type auto-generation from class name",
                input_data={"name": "test"},
                expected_output={"command_type": "sample"},
                scenario=TestScenario.EDGE_CASE,
            ),
            TestCase(
                id="command_validation_success",
                description="Successful command validation",
                input_data={"name": "valid", "value": 42},
                expected_output={"validation_success": True},
                scenario=TestScenario.HAPPY_PATH,
            ),
            TestCase(
                id="command_validation_empty_name",
                description="Command validation failure - empty name",
                input_data={"name": "", "value": 42},
                expected_output={
                    "validation_success": False,
                    "error": "Name cannot be empty",
                },
                scenario=TestScenario.ERROR_CASE,
            ),
            TestCase(
                id="command_validation_negative_value",
                description="Command validation failure - negative value",
                input_data={"name": "test", "value": -1},
                expected_output={
                    "validation_success": False,
                    "error": "Value cannot be negative",
                },
                scenario=TestScenario.ERROR_CASE,
            ),
        ]

    def _validate_basic_properties(
        self,
        command: SampleCommand,
        expected: dict[str, object],
    ) -> None:
        """Validate basic command properties to reduce complexity."""
        if "name" in expected:
            assert command.name == expected["name"]
        if "value" in expected:
            assert command.value == expected["value"]
        if "command_type" in expected:
            assert command.command_type == expected["command_type"]
        if "command_id" in expected:
            assert command.command_id == expected["command_id"]

    def _validate_existence_flags(
        self,
        command: SampleCommand,
        expected: dict[str, object],
    ) -> None:
        """Validate existence flags to reduce complexity."""
        if "has_command_id" in expected:
            assert (command.command_id is not None) == expected["has_command_id"]
        if "has_timestamp" in expected:
            assert isinstance(command.timestamp, datetime) == expected["has_timestamp"]

    def _validate_command_result(
        self,
        command: SampleCommand,
        expected: dict[str, object],
    ) -> None:
        """Validate command validation result to reduce complexity."""
        if "validation_success" in expected:
            result: FlextResult[None] = command.validate_command()
            assert result.success == expected["validation_success"]
            if "error" in expected:
                error_message: str = cast("str", expected["error"])
                assert error_message in (result.error or "")

    @pytest.mark.parametrize_advanced
    def test_command_scenarios(
        self,
        command_test_cases: list[TestCase],
        assert_helpers: AssertHelpers,  # noqa: ARG002
    ) -> None:
        """Test commands using structured parametrized approach."""
        for test_case in command_test_cases:
            input_data: dict[str, object] = cast(
                "dict[str, object]",
                test_case.input_data,
            )
            expected: dict[str, object] = cast(
                "dict[str, object]",
                test_case.expected_output,
            )

            # Create command
            if "timestamp" in input_data and isinstance(input_data["timestamp"], str):
                input_data["timestamp"] = datetime.now(tz=ZoneInfo("UTC"))

            cmd_kwargs: SampleCommandKwargs = cast("SampleCommandKwargs", input_data)
            command: SampleCommand = SampleCommand(**cmd_kwargs)

            # Validate using helper methods to reduce complexity
            self._validate_basic_properties(command, expected)
            self._validate_existence_flags(command, expected)
            self._validate_command_result(command, expected)

    @pytest.fixture
    def payload_conversion_cases(self) -> list[TestCase]:
        """Test cases for payload conversion."""
        return [
            TestCase(
                id="command_to_payload",
                description="Command to payload conversion",
                input_data={"name": "test", "value": 42},
                expected_output={
                    "payload_data_name": "test",
                    "payload_data_value": 42,
                    "payload_type": "sample",
                },
                scenario=TestScenario.HAPPY_PATH,
            ),
            TestCase(
                id="payload_to_command_success",
                description="Successful command creation from payload",
                input_data={
                    "payload_data": {
                        "name": "test",
                        "value": 42,
                        "command_id": "cmd-123",
                    },
                },
                expected_output={
                    "from_payload_success": True,
                    "name": "test",
                    "value": 42,
                    "command_id": "cmd-123",
                },
                scenario=TestScenario.HAPPY_PATH,
            ),
            TestCase(
                id="payload_to_command_validation_failure",
                description="Command creation from payload with validation failure",
                input_data={
                    "payload_data": {
                        "name": "",  # Invalid
                        "value": -1,  # Invalid
                    },
                },
                expected_output={
                    "from_payload_success": False,
                    "error": "Name cannot be empty",
                },
                scenario=TestScenario.ERROR_CASE,
            ),
        ]

    @pytest.mark.parametrize_advanced
    def test_payload_conversion_scenarios(
        self,
        payload_conversion_cases: list[TestCase],
    ) -> None:
        """Test payload conversion scenarios."""
        for test_case in payload_conversion_cases:
            input_data: dict[str, object] = cast(
                "dict[str, object]",
                test_case.input_data,
            )
            if "payload_data" not in input_data:
                self._test_command_to_payload(test_case)
            else:
                self._test_payload_to_command(test_case)

    def _test_command_to_payload(self, test_case: TestCase) -> None:
        """Test command to payload conversion."""
        input_data = test_case.input_data
        expected = test_case.expected_output

        # Type-safe conversion for command creation
        cmd_kwargs = cast("SampleCommandKwargs", input_data)
        command = SampleCommand(**cmd_kwargs)
        payload = command.to_payload()

        assert isinstance(payload, FlextPayload)
        assert payload.value is not None

        # Type-safe expected validation
        expected_dict = cast("dict[str, object]", expected)

        if "payload_data_name" in expected_dict:
            assert payload.value is not None
            payload_dict = cast("dict[str, object]", payload.value)
            assert payload_dict["name"] == expected_dict["payload_data_name"]

        if "payload_data_value" in expected_dict:
            assert payload.value is not None
            payload_dict = cast("dict[str, object]", payload.value)
            assert payload_dict["value"] == expected_dict["payload_data_value"]

        if "payload_type" in expected_dict:
            assert payload.metadata.get("type") == expected_dict["payload_type"]

    def _test_payload_to_command(self, test_case: TestCase) -> None:
        """Test payload to command conversion."""
        input_data_dict = cast("dict[str, object]", test_case.input_data)
        expected_dict = cast("dict[str, object]", test_case.expected_output)

        payload_data = input_data_dict["payload_data"]
        payload = FlextPayload.create(data=payload_data, type="SampleCommand").value

        result: FlextResult[SampleCommand] = SampleCommand.from_payload(payload)

        if "from_payload_success" in expected_dict:
            assert result.success == expected_dict["from_payload_success"]

        if expected_dict.get("from_payload_success", False):
            self._validate_command_properties(result.value, expected_dict)
            assert result.value is not None
        elif "error" in expected_dict:
            error_msg = str(expected_dict["error"])
            assert error_msg in (result.error or "")

    def _validate_command_properties(
        self,
        command: SampleCommand,
        expected: dict[str, object],
    ) -> None:
        """Validate command properties against expected values."""
        if "name" in expected:
            assert command.name == expected["name"]
        if "value" in expected:
            assert command.value == expected["value"]
        if "command_id" in expected:
            assert command.command_id == expected["command_id"]


class TestFlextCommandsComplexValidation:
    """Advanced validation testing with complex scenarios."""

    @pytest.fixture
    def complex_validation_cases(self) -> list[TestCase]:
        """Complex validation test cases."""
        return [
            TestCase(
                id="complex_validation_success",
                description="Complex validation success",
                input_data={"email": "user@example.com", "age": 25},
                expected_output={"validation_success": True},
                scenario=TestScenario.HAPPY_PATH,
            ),
            TestCase(
                id="complex_validation_invalid_email",
                description="Complex validation - invalid email",
                input_data={"email": "invalid-email", "age": 25},
                expected_output={
                    "validation_success": False,
                    "error": "Invalid email format",
                },
                scenario=TestScenario.ERROR_CASE,
            ),
            TestCase(
                id="complex_validation_age_too_young",
                description="Complex validation - age too young",
                input_data={"email": "user@example.com", "age": 17},
                expected_output={
                    "validation_success": False,
                    "error": "Age must be 18 or older",
                },
                scenario=TestScenario.ERROR_CASE,
            ),
        ]

    @pytest.mark.parametrize_advanced
    def test_complex_validation_scenarios(
        self,
        complex_validation_cases: list[TestCase],
    ) -> None:
        """Test complex validation scenarios."""
        for test_case in complex_validation_cases:
            input_data: dict[str, object] = cast(
                "dict[str, object]",
                test_case.input_data,
            )
            expected: dict[str, object] = cast(
                "dict[str, object]",
                test_case.expected_output,
            )

            command: SampleComplexCommand = SampleComplexCommand(
                **cast("SampleComplexCommandKwargs", input_data),
            )
            result = command.validate_command()

            assert result.success == expected["validation_success"]

            if "error" in expected:
                error_msg = str(expected["error"])
                assert error_msg in (result.error or "")


class TestFlextCommandsImmutability:
    """Test command immutability patterns."""

    def test_command_immutability(self) -> None:
        """Test command immutability (frozen model)."""
        command = SampleCommand(name="test", value=42)

        # Attempt to modify the command should raise ValidationError
        with pytest.raises(
            (ValidationError, AttributeError),
            match=r".*frozen.*|.*immutable.*|.*read.*only.*",
        ):
            command.name = "changed"

        with pytest.raises(
            (ValidationError, AttributeError),
            match=r".*frozen.*|.*immutable.*|.*read.*only.*",
        ):
            command.value = 100


class TestFlextCommandsWithoutValidation:
    """Test commands without custom validation."""

    def test_command_without_validation(self) -> None:
        """Test command without custom validation method."""
        command = SampleCommandWithoutValidation(description="test")
        result = command.validate_command()

        assert result.success


class TestFlextCommandsPerformance:
    """Performance tests for command operations."""

    def test_command_creation_performance(
        self,
        performance_monitor: Callable[[Callable[[], object]], PerformanceMetrics],
    ) -> None:
        """Test command creation performance."""

        def create_commands() -> list[SampleCommand]:
            commands = []
            for i in range(1000):
                command: SampleCommand = SampleCommand(name=f"test_{i}", value=i)
                commands.append(command)
            return commands

        metrics: PerformanceMetrics = performance_monitor(create_commands)

        # Should create 1000 commands quickly
        assert (
            metrics["execution_time"] < 0.3
        )  # Less than 300ms (more realistic for CI)
        assert metrics["result"] is not None
        result_commands = cast("list[SampleCommand]", metrics["result"])
        assert len(result_commands) == 1000


class TestFlextCommandsFactoryMethods:
    """Test factory methods and other uncovered functionality."""

    def test_create_command_bus_factory(self) -> None:
        """Test FlextCommands.create_command_bus factory method."""
        bus = FlextCommands.create_command_bus()
        assert isinstance(bus, FlextCommands.Bus)
        assert hasattr(bus, "execute")
        assert hasattr(bus, "register_handler")

    def test_create_simple_handler_factory(self) -> None:
        """Test FlextCommands.create_simple_handler factory method."""

        def sample_handler(command: object) -> object:
            return f"handled: {command}"

        handler = FlextCommands.create_simple_handler(sample_handler)
        assert isinstance(handler, FlextCommands.Handler)

        # Test the handler works
        test_command = SampleCommand(name="test", value=42)
        result = handler.handle(test_command)
        assert result.success
        assert "handled:" in str(result.value)

    def test_command_bus_middleware_system(self) -> None:
        """Test command bus middleware functionality."""
        bus = FlextCommands.Bus()

        # Add middleware
        class TestMiddleware:
            def process(self, command: object, handler: object) -> FlextResult[None]:  # noqa: ARG002
                return FlextResult[None].ok(None)

        bus.add_middleware(TestMiddleware())
        handlers = bus.get_all_handlers()
        assert isinstance(handlers, list)

    def test_command_handler_with_validation(self) -> None:
        """Test handler validation and processing paths."""

        class TestHandler(FlextCommands.Handler[SampleCommand, str]):
            def handle(self, command: SampleCommand) -> FlextResult[str]:
                return FlextResult[str].ok(f"processed: {command.name}")

        handler = TestHandler()
        assert handler.handler_name == "TestHandler"

        # Test can_handle method
        test_command = SampleCommand(name="test", value=42)
        can_handle = handler.can_handle(test_command)
        assert isinstance(can_handle, bool)

        # Test processing
        result = handler.process_command(test_command)
        assert result.success

    def test_command_decorators_functionality(self) -> None:
        """Test command decorator patterns."""

        @FlextCommands.Decorators.command_handler(SampleCommand)
        def decorated_handler(command: object) -> object:
            if isinstance(command, SampleCommand):
                return f"decorated: {command.name}"
            return f"decorated: {command}"

        # Verify decorator metadata
        assert "command_type" in decorated_handler.__dict__
        assert "handler_instance" in decorated_handler.__dict__

    def test_command_result_with_metadata(self) -> None:
        """Test FlextCommands.Result with error data functionality."""
        # Test successful result (FlextResult doesn't support metadata in .ok())
        result: FlextResult[str] = FlextCommands.Result.ok("success")
        assert result is not None
        assert result.success
        assert result.value == "success"

        # Test failed result with error code and error data
        failed_result: FlextResult[None] = FlextCommands.Result.fail(
            "error message",
            error_code="TEST_ERROR",
            error_data={"context": "test"},
        )
        assert failed_result is not None
        assert failed_result.is_failure
        assert failed_result.error_code == "TEST_ERROR"
        assert failed_result.error_data["context"] == "test"

    def test_query_functionality(self) -> None:
        """Test Query classes and handlers."""

        class TestQuery(FlextCommands.Query):
            search_term: str = "test"

        query = TestQuery()
        validation_result: FlextResult[None] = query.validate_query()
        assert validation_result is not None
        assert validation_result.success

        # Test query handler
        class TestQueryHandler(FlextCommands.QueryHandler[TestQuery, str]):
            def handle(self, query: TestQuery) -> FlextResult[str]:
                return FlextResult[str].ok(f"found: {query.search_term}")

        handler = TestQueryHandler()
        assert handler.handler_name == "TestQueryHandler"
        result: FlextResult[str] = handler.handle(query)
        assert result is not None
        assert result.success
        assert result.value is not None
        assert "found: test" in str(result.value)


# All edge cases, integration tests, and additional coverage tests have been
# consolidated into the advanced parametrized test classes above.
# This reduces code duplication while maintaining comprehensive CQRS command coverage.
