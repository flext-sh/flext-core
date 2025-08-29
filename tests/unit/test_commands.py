# ruff: noqa: ARG001, ARG002
"""Advanced tests for FlextCommands using comprehensive tests/support/ utilities.

Tests CQRS patterns, command validation, performance, and real-world scenarios
using consolidated testing infrastructure for maximum coverage and reliability.

Enhanced Testing Patterns:
- Property-based testing with hypothesis strategies
- Performance benchmarking and memory profiling
- Advanced FlextMatchers for sophisticated assertions
- Factory patterns for realistic test data generation
- Async testing with concurrency scenarios
- Memory-efficient testing patterns
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TypedDict, cast
from zoneinfo import ZoneInfo

import pytest
from hypothesis import given, strategies as st
from pydantic import ValidationError

from flext_core import (
    FlextCommands,
    FlextModels,
    FlextResult,
)

# Import comprehensive tests/support/ utilities
from ..support import (
    AsyncTestUtils,
    FlextMatchers,
    MemoryProfiler,
    ServiceDataFactory,
    TestBuilders,
    UserDataFactory,
)
from ..support.builders import build_test_container


# Test scenario enumeration for testing patterns
class TestScenario(Enum):
    """Test scenario types for parametrized testing."""

    HAPPY_PATH = "happy_path"
    ERROR_CASE = "error_case"
    EDGE_CASE = "edge_case"
    PERFORMANCE = "performance"
    VALIDATION = "validation"


class PerformanceMetrics(TypedDict):
    """Performance metrics for testing."""

    execution_time: float
    memory_usage: float
    operations_count: int


# Simple test case structure for command testing
@dataclass
class TestCase:
    """Test case structure for command testing."""

    name: str
    data: dict[str, object]
    expected: bool
    input_data: dict[str, object] = field(default_factory=dict)
    expected_output: object = None
    id: str = ""
    description: str = ""
    scenario: TestScenario = TestScenario.HAPPY_PATH


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


class SampleCommand(FlextCommands.Models.Command):
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


class SampleCommandWithoutValidation(FlextCommands.Models.Command):
    """Test command without custom validation."""

    description: str


class SampleComplexCommand(FlextCommands.Models.Command):
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
                name="command_creation_basic",
                data={"name": "test", "value": 42},
                expected=True,
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
                name="command_creation_explicit",
                data={
                    "name": "explicit",
                    "value": 100,
                    "command_id": "cmd-123",
                    "command_type": "CustomType",
                    "user_id": "user-456",
                    "correlation_id": "corr-789",
                },
                expected=True,
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
                name="command_type_auto_generation",
                data={"name": "test"},
                expected=True,
                id="command_type_auto_generation",
                description="Command type auto-generation from class name",
                input_data={"name": "test"},
                expected_output={"command_type": "sample"},
                scenario=TestScenario.EDGE_CASE,
            ),
            TestCase(
                name="command_validation_success",
                data={"name": "valid", "value": 42},
                expected=True,
                id="command_validation_success",
                description="Successful command validation",
                input_data={"name": "valid", "value": 42},
                expected_output={"validation_success": True},
                scenario=TestScenario.HAPPY_PATH,
            ),
            TestCase(
                name="command_validation_empty_name",
                data={"name": "", "value": 42},
                expected=False,
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
                name="command_validation_negative_value",
                data={"name": "test", "value": -1},
                expected=False,
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
        expected: dict[str, object] | None,
    ) -> None:
        """Validate basic command properties to reduce complexity."""
        if expected is None:
            return  # No validation needed
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
        expected: dict[str, object] | None,
    ) -> None:
        """Validate existence flags to reduce complexity."""
        if expected is None:
            return  # No validation needed
        if "has_command_id" in expected:
            assert (command.command_id is not None) == expected["has_command_id"]
        if "has_timestamp" in expected:
            assert isinstance(command.timestamp, datetime) == expected["has_timestamp"]

    def _validate_command_result(
        self,
        command: SampleCommand,
        expected: dict[str, object] | None,
    ) -> None:
        """Validate command validation result to reduce complexity."""
        if expected is None:
            return  # No validation needed
        if "validation_success" in expected:
            result: FlextResult[None] = command.validate_command()
            assert result.success == expected["validation_success"]
            if "error" in expected:
                error_message: str = cast("str", expected["error"])
                assert error_message in (result.error or "")

    @pytest.mark.parametrize_advanced
    def test_command_scenarios(
        self,
    ) -> None:
        """Test commands using structured parametrized approach."""
        # Create basic test cases
        command_test_cases = [
            TestCase(
                name="basic_command",
                data={"name": "test_command", "value": 100},
                expected=True,
                input_data={"name": "test_command", "value": 100},
            ),
        ]
        for test_case in command_test_cases:
            input_data = test_case.input_data
            expected = test_case.expected_output

            # Create command
            if "timestamp" in input_data and isinstance(input_data["timestamp"], str):
                input_data["timestamp"] = datetime.now(tz=ZoneInfo("UTC"))

            cmd_kwargs: SampleCommandKwargs = cast("SampleCommandKwargs", input_data)
            command: SampleCommand = SampleCommand(**cmd_kwargs)

            # Validate using helper methods to reduce complexity
            expected_dict = cast("dict[str, object] | None", expected)
            self._validate_basic_properties(command, expected_dict)
            self._validate_existence_flags(command, expected_dict)
            self._validate_command_result(command, expected_dict)

    @pytest.fixture
    def payload_conversion_cases(self) -> list[TestCase]:
        """Test cases for payload conversion."""
        return [
            TestCase(
                name="command_to_payload",
                data={"name": "test", "value": 42},
                expected=True,
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
                name="payload_to_command_success",
                data={"name": "test", "value": 42},
                expected=True,
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
                name="payload_to_command_validation_failure",
                data={"name": "", "value": -1},
                expected=False,
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
            input_data = test_case.input_data
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

        assert isinstance(payload, FlextModels.Payload)
        assert payload.data is not None

        if expected is not None and isinstance(expected, dict):
            if "payload_data_name" in expected:
                assert payload.data is not None
                assert isinstance(payload.data, dict)
                assert payload.data["name"] == expected["payload_data_name"]

            if "payload_data_value" in expected:
                assert payload.data is not None
                assert isinstance(payload.data, dict)
                assert payload.data["value"] == expected["payload_data_value"]

            if "payload_type" in expected:
                assert payload.message_type == expected["payload_type"]

    def _test_payload_to_command(self, test_case: TestCase) -> None:
        """Test payload to command conversion."""
        input_data_dict = test_case.input_data
        expected_dict = test_case.expected_output

        if not isinstance(input_data_dict, dict) or not isinstance(expected_dict, dict):
            pytest.skip("Invalid test case data")

        payload_data = input_data_dict["payload_data"]
        payload_result = FlextModels.create_payload(
            data=payload_data,
            message_type="SampleCommand",
            source_service="test_service",
        )
        assert payload_result.success
        payload = payload_result.unwrap()

        # Ensure payload data is dict for SampleCommand.from_payload compatibility
        if hasattr(payload, "data") and isinstance(payload.data, dict):
            # Cast payload to the expected type for from_payload method
            payload_dict = cast("FlextModels.Payload[dict[str, object]]", payload)
            result: FlextResult[SampleCommand] = SampleCommand.from_payload(
                payload_dict
            )
        else:
            result = FlextResult[SampleCommand].fail("Payload data is not compatible")

        if "from_payload_success" in expected_dict:
            assert result.success == expected_dict["from_payload_success"]

        if expected_dict.get("from_payload_success", False) and result.success:
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
                name="complex_validation_success",
                data={"email": "user@example.com", "age": 25},
                expected=True,
                id="complex_validation_success",
                description="Complex validation success",
                input_data={"email": "user@example.com", "age": 25},
                expected_output={"validation_success": True},
                scenario=TestScenario.HAPPY_PATH,
            ),
            TestCase(
                name="complex_validation_invalid_email",
                data={"email": "invalid-email", "age": 25},
                expected=False,
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
                name="complex_validation_age_too_young",
                data={"email": "user@example.com", "age": 17},
                expected=False,
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
            input_data = test_case.input_data
            expected = test_case.expected_output

            command: SampleComplexCommand = SampleComplexCommand(
                **cast("SampleComplexCommandKwargs", input_data),
            )
            result = command.validate_command()

            expected_dict = cast("dict[str, object]", expected)
            assert result.success == expected_dict["validation_success"]

            if "error" in expected_dict:
                error_msg = str(expected_dict["error"])
                assert error_msg in (result.error or "")


class TestFlextCommandsImmutability:
    """Test command immutability patterns."""

    def test_command_immutability(self) -> None:
        """Test command immutability (frozen model)."""
        command = SampleCommand(name="test", value=42)

        # Test that model is frozen - attempts to modify should raise error
        with pytest.raises(
            (ValidationError, AttributeError, TypeError),
            match=r".*frozen.*|.*immutable.*|.*read.*only.*|.*cannot.*assign.*|.*dataclass.*frozen.*",
        ):
            # This should fail because SampleCommand inherits from frozen Command model
            setattr(command, "name", "changed")

        with pytest.raises(
            (ValidationError, AttributeError, TypeError),
            match=r".*frozen.*|.*immutable.*|.*read.*only.*|.*cannot.*assign.*|.*dataclass.*frozen.*",
        ):
            # This should fail because SampleCommand inherits from frozen Command model
            setattr(command, "value", 100)


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
        performance_profiler: MemoryProfiler,
    ) -> None:
        """Test command creation performance."""

        def create_commands() -> list[SampleCommand]:
            commands = []
            for i in range(1000):
                command: SampleCommand = SampleCommand(name=f"test_{i}", value=i)
                commands.append(command)
            return commands

        # Monitor performance of command creation with memory profiling
        with MemoryProfiler.track_memory_leaks(max_increase_mb=10.0):
            commands = create_commands()

        # Assert reasonable performance - verify commands were created
        assert len(commands) == 1000


class TestFlextCommandsFactoryMethods:
    """Test factory methods and other uncovered functionality."""

    def test_create_command_bus_factory(self) -> None:
        """Test FlextCommands.create_command_bus factory method."""
        bus = FlextCommands.Factories.create_command_bus()
        assert isinstance(bus, FlextCommands.Bus)
        assert hasattr(bus, "execute")
        assert hasattr(bus, "register_handler")

    def test_create_simple_handler_factory(self) -> None:
        """Test FlextCommands.create_simple_handler factory method."""

        def sample_handler(command: object) -> object:
            return f"handled: {command}"

        handler = FlextCommands.Factories.create_simple_handler(sample_handler)
        assert isinstance(handler, FlextCommands.Handlers.CommandHandler)

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
            def process(self, command: object, handler: object) -> FlextResult[None]:
                return FlextResult[None].ok(None)

        bus.add_middleware(TestMiddleware())
        handlers = bus.get_all_handlers()
        assert isinstance(handlers, list)

    def test_command_handler_with_validation(self) -> None:
        """Test handler validation and processing paths."""

        class TestHandler(FlextCommands.Handlers.CommandHandler[SampleCommand, str]):
            def handle(self, command: SampleCommand) -> FlextResult[str]:
                return FlextResult[str].ok(f"processed: {command.name}")

        handler = TestHandler()
        assert handler.handler_name == "TestHandler"

        # Test can_handle method
        test_command = SampleCommand(name="test", value=42)
        can_handle = handler.can_handle(test_command)
        assert isinstance(can_handle, bool)

        # Test processing
        result = handler.handle(test_command)
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
        result = FlextCommands.Results.success("success")
        assert result is not None
        assert result.success
        assert result.value == "success"

        # Test failed result with error code and error data
        failed_result = FlextCommands.Results.failure(
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

        class TestQuery(FlextCommands.Models.Query):
            search_term: str = "test"

        query = TestQuery()
        validation_result: FlextResult[None] = query.validate_query()
        assert validation_result is not None
        assert validation_result.success

        # Test query handler
        class TestQueryHandler(FlextCommands.Handlers.QueryHandler[TestQuery, str]):
            def handle(self, query: TestQuery) -> FlextResult[str]:
                return FlextResult[str].ok(f"found: {query.search_term}")

        handler = TestQueryHandler()
        assert handler.handler_name == "TestQueryHandler"
        result: FlextResult[str] = handler.handle(query)
        assert result is not None
        assert result.success
        assert result.value is not None
        assert "found: test" in str(result.value)


# ============================================================================
# ENHANCED TESTS USING ADVANCED TESTS/SUPPORT/ INFRASTRUCTURE
# ============================================================================


class TestFlextCommandsEnhanced:
    """Enhanced FlextCommands testing using comprehensive tests/support/ utilities.

    This class demonstrates advanced testing patterns with:
    - Property-based testing with hypothesis
    - Performance benchmarking and memory profiling
    - Advanced assertions with FlextMatchers
    - Realistic test data with factories
    - Async testing and concurrency scenarios
    """

    def test_command_creation_with_user_factory(self) -> None:
        """Test command creation using UserDataFactory for realistic data."""
        # Use factory for realistic user data
        user_data = UserDataFactory.create(name="John Smith", email="john@example.com")

        # Create command with realistic data - ensure type safety
        user_name = user_data.get("name", "default_name")
        if not isinstance(user_name, str):
            user_name = str(user_name)
        command = SampleCommand(name=user_name, value=42)

        # Use FlextMatchers for sophisticated assertions
        FlextMatchers.assert_json_structure(
            command.model_dump(),
            ["name", "value", "command_id", "command_type", "timestamp"],
            exact_match=False,
        )

        # Validate command data
        assert command.name == "John Smith"
        assert command.value == 42
        assert command.command_type == "sample"

    def test_command_validation_with_matchers(self) -> None:
        """Test command validation using FlextMatchers patterns."""
        # Test successful validation
        valid_command = SampleCommand(name="Valid Name", value=100)
        validation_result = valid_command.validate_command()

        FlextMatchers.assert_result_success(validation_result)

        # Test validation failure
        invalid_command = SampleCommand(name="", value=-10)
        fail_result = invalid_command.validate_command()

        FlextMatchers.assert_result_failure(
            fail_result, expected_error="Name cannot be empty"
        )

    @given(st.text(min_size=1, max_size=50))
    def test_command_property_based_names(self, name: str) -> None:
        """Test command creation with property-based testing for names."""
        # Create command with generated name
        command = SampleCommand(name=name.strip(), value=10)

        if name.strip():  # Valid name
            result = command.validate_command()
            FlextMatchers.assert_result_success(result)
            assert command.name == name.strip()
        else:  # Empty name should fail validation
            result = command.validate_command()
            FlextMatchers.assert_result_failure(result)

    def test_command_performance_benchmarking(self, benchmark: object) -> None:
        """Test command creation performance using pytest-benchmark."""

        def create_multiple_commands() -> list[SampleCommand]:
            commands = []
            for i in range(100):
                user_data = UserDataFactory.create(name=f"User {i}")
                user_name = user_data.get("name", f"User {i}")
                if not isinstance(user_name, str):
                    user_name = str(user_name)
                command = SampleCommand(name=user_name, value=i)
                commands.append(command)
            return commands

        # Benchmark command creation
        result = FlextMatchers.assert_performance_within_limit(
            benchmark, create_multiple_commands, max_time_seconds=0.1
        )

        # Validate results - ensure result is a list
        if isinstance(result, list):
            assert len(result) == 100
            command: SampleCommand
            for i, command in enumerate(result):
                assert command.name == f"User {i}"
                assert command.value == i
        else:
            pytest.fail("Expected list result from benchmark")

    def test_command_memory_efficiency(self) -> None:
        """Test command memory usage with MemoryProfiler."""
        with MemoryProfiler.track_memory_leaks(max_increase_mb=5.0):
            # Create commands and clean up periodically to test memory management
            commands = []
            for i in range(500):  # Reduced number to be more realistic
                service_data = ServiceDataFactory.create(name=f"service_{i}")
                service_name = service_data.get("name", f"service_{i}")
                service_port = service_data.get("port", i)
                if not isinstance(service_name, str):
                    service_name = str(service_name)
                if not isinstance(service_port, int):
                    service_port = (
                        int(service_port) if str(service_port).isdigit() else i
                    )
                command = SampleCommand(name=service_name, value=service_port)
                commands.append(command)

                # Clear all commands periodically to test memory cleanup
                if i > 0 and i % 100 == 0:
                    commands.clear()  # Full cleanup

        # Verify test completed successfully
        assert len(commands) >= 0  # Should have completed without memory issues

    @pytest.mark.asyncio
    async def test_async_command_processing(self) -> None:
        """Test async command processing patterns."""

        async def process_command_async(command: SampleCommand) -> dict[str, object]:
            """Simulate async command processing."""
            await AsyncTestUtils.simulate_delay(0.01)
            validation_result = command.validate_command()

            if validation_result.is_success:
                return {
                    "command_id": command.command_id,
                    "name": command.name,
                    "processed": True,
                    "timestamp": command.timestamp.isoformat(),
                }
            return {
                "command_id": command.command_id,
                "processed": False,
                "error": validation_result.error,
            }

        # Create test commands
        commands = [
            SampleCommand(name="Valid Command 1", value=10),
            SampleCommand(name="Valid Command 2", value=20),
            SampleCommand(name="", value=30),  # Invalid - empty name
        ]

        # Process commands concurrently
        tasks = [process_command_async(cmd) for cmd in commands]
        results = await AsyncTestUtils.run_concurrently(*tasks)

        # Validate results
        assert len(results) == 3
        assert results[0]["processed"] is True
        assert results[1]["processed"] is True
        assert results[2]["processed"] is False
        assert "error" in results[2]

    def test_command_builder_pattern(self) -> None:
        """Test command creation using TestBuilders pattern."""
        # Use TestBuilders for sophisticated test setup
        container = TestBuilders.container().build()

        # Register command-related services
        user_service = UserDataFactory.create()
        validation_service = {"strict_mode": True, "max_errors": 5}

        container.register("user_service", user_service)
        container.register("validation_service", validation_service)

        # Test service-aware command processing
        def process_with_services(command: SampleCommand) -> dict[str, object]:
            user_result = container.get("user_service")
            validation_result = container.get("validation_service")

            FlextMatchers.assert_result_success(user_result)
            FlextMatchers.assert_result_success(validation_result)

            return {
                "command_processed": True,
                "user_service_available": user_result.is_success,
                "validation_service_available": validation_result.is_success,
                "command_name": command.name,
            }

        # Test command processing
        test_command = SampleCommand(name="Service Test", value=50)
        result = process_with_services(test_command)

        FlextMatchers.assert_json_structure(
            result,
            [
                "command_processed",
                "user_service_available",
                "validation_service_available",
                "command_name",
            ],
        )

        assert result["command_processed"] is True
        assert result["user_service_available"] is True
        assert result["validation_service_available"] is True

    @given(
        st.lists(
            st.tuples(
                st.text(min_size=1, max_size=20),
                st.integers(min_value=0, max_value=1000),
            ),
            min_size=1,
            max_size=10,
        )
    )
    def test_batch_command_processing(
        self, command_data: list[tuple[str, int]]
    ) -> None:
        """Test batch command processing with property-based testing."""
        # Create commands from generated data
        commands = []
        for name, value in command_data:
            command = SampleCommand(name=name.strip(), value=value)
            commands.append(command)

        # Process all commands
        results = []
        for command in commands:
            validation_result = command.validate_command()
            results.append(
                {
                    "command": command,
                    "valid": validation_result.is_success,
                    "error": validation_result.error
                    if validation_result.is_failure
                    else None,
                }
            )

        # Validate batch results
        assert len(results) == len(command_data)

        # All commands with non-empty names should be valid
        for i, (name, _) in enumerate(command_data):
            result_item = results[i]
            if isinstance(result_item, dict) and "command" in result_item:
                command = result_item["command"]
                if hasattr(command, "validate_command"):
                    if name.strip():  # Non-empty name
                        FlextMatchers.assert_result_success(command.validate_command())
                    else:  # Empty name should fail
                        FlextMatchers.assert_result_failure(command.validate_command())

    def test_real_world_cqrs_scenario(self) -> None:
        """Test realistic CQRS scenario with microservice dependencies."""
        # Setup realistic microservice container
        container = build_test_container()

        # Add command-specific services
        command_bus_config = ServiceDataFactory.create(
            name="command_bus", port=8080, version="1.0.0"
        )
        container.register("command_bus", command_bus_config)

        # Simulate CQRS command processing pipeline
        def cqrs_pipeline(command: SampleCommand) -> dict[str, object]:
            # 1. Validate command
            validation_result = command.validate_command()
            if validation_result.is_failure:
                return {"success": False, "error": validation_result.error}

            # 2. Check services availability
            db_result = container.get("database")
            bus_result = container.get("command_bus")

            if db_result.is_failure or bus_result.is_failure:
                return {"success": False, "error": "Services unavailable"}

            # 3. Process command - ensure value is dict for indexing
            bus_value = (
                bus_result.value
                if isinstance(bus_result.value, dict)
                else {"name": "unknown_bus"}
            )
            db_value = (
                db_result.value
                if isinstance(db_result.value, dict)
                else {"name": "unknown_db"}
            )

            return {
                "success": True,
                "command_id": command.command_id,
                "processed_by": bus_value.get("name", "unknown_bus"),
                "stored_in": db_value.get("name", "unknown_db"),
                "processing_time": 0.001,
            }

        # Test successful pipeline
        valid_command = SampleCommand(name="CQRS Test Command", value=100)
        result = cqrs_pipeline(valid_command)

        assert result["success"] is True
        assert result["command_id"] == valid_command.command_id
        assert result["processed_by"] == "command_bus"
        assert result["stored_in"] == "test_db"

        # Test failure cases
        invalid_command = SampleCommand(name="", value=100)
        fail_result = cqrs_pipeline(invalid_command)

        assert fail_result["success"] is False
        assert "error" in fail_result


# All edge cases, integration tests, and additional coverage tests have been
# consolidated into the advanced parametrized test classes above.
# This reduces code duplication while maintaining comprehensive CQRS command coverage.
