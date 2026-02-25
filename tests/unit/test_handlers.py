"""Comprehensive tests for h - Handler Management.

Module: flext_core.handlers
Scope: h - handler management, execution, validation, factory methods

Tests the actual h API with real functionality testing.

Uses Python 3.13 patterns, FlextTestsUtilities, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, cast

import pytest
from pydantic import BaseModel

from flext_core import (
    FlextExceptions,
    FlextResult,
    c,
    h,
    m,
    t,
    x,
)
from flext_tests import FlextTestsUtilities, u
from tests.test_utils import assertion_helpers


class ConcreteTestHandler(h[str, str]):
    """Concrete implementation of h for testing."""

    def handle(self, message: str) -> FlextResult[str]:
        """Handle the message."""
        return FlextResult[str].ok(f"processed_{message}")


class FailingTestHandler(h[str, str]):
    """Concrete implementation that fails for testing error handling."""

    def handle(self, message: str) -> FlextResult[str]:
        """Handle the message with failure."""
        return FlextResult[str].fail(f"Handler failed for: {message}")


@dataclass(frozen=True, slots=True)
class HandlerConfigScenario:
    """Handler configuration test scenario."""

    name: str
    handler_id: str
    handler_name: str
    handler_type: str | None = None
    handler_mode: str | None = None
    command_timeout: int | None = None
    max_command_retries: int | None = None
    metadata: dict[str, t.GeneralValueType] | None = None


@dataclass(frozen=True, slots=True)
class HandlerTypeScenario:
    """Handler type test scenario."""

    name: str
    handler_type: c.Cqrs.HandlerType
    handler_mode: c.Cqrs.HandlerType


class HandlerScenarios:
    """Centralized handler test scenarios using FlextConstants."""

    HANDLER_TYPES: ClassVar[list[HandlerTypeScenario]] = [
        HandlerTypeScenario(
            "command",
            c.Cqrs.HandlerType.COMMAND,
            c.Cqrs.HandlerType.COMMAND,
        ),
        HandlerTypeScenario(
            "query",
            c.Cqrs.HandlerType.QUERY,
            c.Cqrs.HandlerType.QUERY,
        ),
        HandlerTypeScenario(
            "event",
            c.Cqrs.HandlerType.EVENT,
            c.Cqrs.HandlerType.EVENT,
        ),
        HandlerTypeScenario(
            "saga",
            c.Cqrs.HandlerType.SAGA,
            c.Cqrs.HandlerType.SAGA,
        ),
    ]

    VALIDATION_TYPES: ClassVar[list[tuple[str, object]]] = [
        ("str", "test_message"),
        ("int", 42),
        ("float", math.pi),
        ("bool", True),
        ("dict", {"key": "value", "number": 42}),
    ]


class TestFlextHandlers:
    """Test suite for h handler management using FlextTestsUtilities."""

    def test_handlers_initialization(self) -> None:
        """Test handlers initialization."""
        config = FlextTestsUtilities.Tests.HandlerHelpers.create_handler_config(
            "test_handler_1",
            "Test Handler 1",
        )
        handlers = ConcreteTestHandler(config=config)
        assert handlers is not None
        assert isinstance(handlers, h)

    def test_handlers_with_custom_config(self) -> None:
        """Test handlers initialization with custom configuration."""
        config = FlextTestsUtilities.Tests.HandlerHelpers.create_handler_config(
            "test_handler_2",
            "Test Handler 2",
            handler_type=c.Cqrs.HandlerType.QUERY,
            handler_mode=c.Cqrs.HandlerType.QUERY,
        )
        handlers = ConcreteTestHandler(config=config)
        assert handlers is not None
        assert handlers._config_model.handler_type == c.Cqrs.HandlerType.QUERY

    def test_handlers_handle_success(self) -> None:
        """Test successful handler execution."""
        config = FlextTestsUtilities.Tests.HandlerHelpers.create_handler_config(
            "test_handler_3",
            "Test Handler 3",
        )
        handler = ConcreteTestHandler(config=config)
        result = handler.handle("test_message")
        u.Tests.Result.assert_success_with_value(
            result,
            "processed_test_message",
        )

    def test_handlers_handle_failure(self) -> None:
        """Test handler execution with failure."""
        config = FlextTestsUtilities.Tests.HandlerHelpers.create_handler_config(
            "test_handler_4",
            "Test Handler 4",
        )
        handler = FailingTestHandler(config=config)
        result = handler.handle("test_message")
        u.Tests.Result.assert_result_failure_with_error(
            result,
            expected_error="Handler failed for: test_message",
        )

    def test_handlers_config_access(self) -> None:
        """Test access to handler configuration."""
        config = FlextTestsUtilities.Tests.HandlerHelpers.create_handler_config(
            "test_handler_5",
            "Test Handler 5",
            handler_type=c.Cqrs.HandlerType.COMMAND,
            handler_mode=c.Cqrs.HandlerType.COMMAND,
        )
        handler = ConcreteTestHandler(config=config)
        assert handler._config_model.handler_id == "test_handler_5"
        assert handler._config_model.handler_name == "Test Handler 5"
        assert handler._config_model.handler_type == c.Cqrs.HandlerType.COMMAND

    def test_handlers_execution_context(self) -> None:
        """Test handler execution context creation."""
        config = FlextTestsUtilities.Tests.HandlerHelpers.create_handler_config(
            "test_handler_6",
            "Test Handler 6",
        )
        handler = ConcreteTestHandler(config=config)
        assert handler._execution_context is not None
        assert hasattr(handler._execution_context, "handler_name")

    def test_handlers_different_types(self) -> None:
        """Test handlers with different message and result types."""

        class IntHandler(h[int, str]):
            def handle(self, message: int) -> FlextResult[str]:
                return FlextResult[str].ok(f"processed_{message}")

        config = FlextTestsUtilities.Tests.HandlerHelpers.create_handler_config(
            "test_handler_10",
            "Test Handler 10",
        )
        handler = IntHandler(config=config)
        result = handler.handle(42)
        u.Tests.Result.assert_success_with_value(
            result,
            "processed_42",
        )

    @pytest.mark.parametrize(
        "scenario",
        HandlerScenarios.HANDLER_TYPES,
        ids=lambda s: s.name,
    )
    def test_handlers_types(self, scenario: HandlerTypeScenario) -> None:
        """Test handlers with various types."""
        config = FlextTestsUtilities.Tests.HandlerHelpers.create_handler_config(
            f"test_{scenario.name}_handler",
            f"Test {scenario.name.title()} Handler",
            handler_type=scenario.handler_type,
            handler_mode=scenario.handler_mode,
        )
        handler = ConcreteTestHandler(config=config)
        assert handler._config_model.handler_type == scenario.handler_type
        assert handler._config_model.handler_mode == scenario.handler_mode

    def test_handlers_with_metadata(self) -> None:
        """Test handlers with metadata configuration."""
        config = FlextTestsUtilities.Tests.HandlerHelpers.create_handler_config(
            "test_handler_with_metadata",
            "Test Handler With Metadata",
            metadata=m.Metadata(
                attributes={"test_key": "test_value", "priority": 1},
            ),
        )
        handler = ConcreteTestHandler(config=config)
        assert handler._config_model.metadata is not None
        assert handler._config_model.metadata.attributes["test_key"] == "test_value"

    def test_handlers_with_timeout(self) -> None:
        """Test handlers with timeout configuration."""
        config = FlextTestsUtilities.Tests.HandlerHelpers.create_handler_config(
            "test_handler_with_timeout",
            "Test Handler With Timeout",
            command_timeout=60,
        )
        handler = ConcreteTestHandler(config=config)
        assert handler._config_model.command_timeout == 60

    def test_handlers_with_retry_config(self) -> None:
        """Test handlers with retry configuration."""
        config = FlextTestsUtilities.Tests.HandlerHelpers.create_handler_config(
            "test_handler_with_retry",
            "Test Handler With Retry",
            max_command_retries=3,
        )
        handler = ConcreteTestHandler(config=config)
        assert handler._config_model.max_command_retries == 3

    def test_handlers_inheritance_chain(self) -> None:
        """Test that handlers inherit from x."""
        config = FlextTestsUtilities.Tests.HandlerHelpers.create_handler_config(
            "test_inheritance_handler",
            "Test Inheritance Handler",
        )
        handler = ConcreteTestHandler(config=config)
        assert isinstance(handler, x)

    def test_handlers_run_pipeline_with_dict_message_command_id(self) -> None:
        """Test _run_pipeline with dict[str, t.GeneralValueType] message having command_id."""

        class DictHandler(h[dict[str, t.GeneralValueType], str]):
            def __init__(self, config: m.CqrsHandler) -> None:
                super().__init__(config=config)

            def handle(
                self, message: dict[str, t.GeneralValueType]
            ) -> FlextResult[str]:
                return FlextResult[str].ok(f"processed_{message}")

        config_raw = FlextTestsUtilities.Tests.HandlerHelpers.create_handler_config(
            "test_pipeline_dict_command_id",
            "Test Pipeline Dict Command ID",
            handler_type=c.Cqrs.HandlerType.COMMAND,
            handler_mode=c.Cqrs.HandlerType.COMMAND,
        )
        # Cast to m.CqrsHandler for type compatibility
        config = config_raw
        handler = DictHandler(config=config)
        dict_message: dict[str, t.GeneralValueType] = {
            "command_id": "cmd_123",
            "data": "test_data",
        }
        result = handler._run_pipeline(dict_message, operation="command")
        u.Tests.Result.assert_result_success(result)

    def test_handlers_run_pipeline_mode_validation_error(self) -> None:
        """Test _run_pipeline with mismatched operation and handler mode."""
        config = FlextTestsUtilities.Tests.HandlerHelpers.create_handler_config(
            "test_pipeline_mode_error",
            "Test Pipeline Mode Error",
            handler_type=c.Cqrs.HandlerType.COMMAND,
            handler_mode=c.Cqrs.HandlerType.COMMAND,
        )
        handler = ConcreteTestHandler(config=config)
        result = handler._run_pipeline("test_message", operation="query")
        u.Tests.Result.assert_result_failure_with_error(
            result,
            expected_error="Handler with mode 'command' cannot execute query pipelines",
        )

    def test_handlers_run_pipeline_cannot_handle_message_type(self) -> None:
        """Test _run_pipeline when handler cannot handle message type."""

        class RestrictiveHandler(h[str, str]):
            def __init__(self, config: m.CqrsHandler) -> None:
                super().__init__(config=config)

            def can_handle(self, message_type: object) -> bool:
                _ = message_type
                return False

            def handle(self, message: str) -> FlextResult[str]:
                return FlextResult[str].ok(f"processed_{message}")

        config_raw = FlextTestsUtilities.Tests.HandlerHelpers.create_handler_config(
            "test_pipeline_cannot_handle",
            "Test Pipeline Cannot Handle",
            handler_type=c.Cqrs.HandlerType.COMMAND,
            handler_mode=c.Cqrs.HandlerType.COMMAND,
        )
        config = cast("m.CqrsHandler", config_raw)
        handler = RestrictiveHandler(config=config)
        result = handler._run_pipeline("test_message", operation="command")
        u.Tests.Result.assert_result_failure_with_error(
            result,
            expected_error="Handler cannot handle message type str",
        )

    def test_handlers_run_pipeline_validation_failure(self) -> None:
        """Test _run_pipeline when message validation fails."""

        class ValidationFailingHandler(h[str, str]):
            def __init__(self, config: m.CqrsHandler) -> None:
                super().__init__(config=config)

            def validate_command(self, command: object) -> FlextResult[bool]:
                _ = command
                return FlextResult[bool].fail("Validation failed for test")

            def handle(self, message: str) -> FlextResult[str]:
                return FlextResult[str].ok(f"processed_{message}")

        config_raw = FlextTestsUtilities.Tests.HandlerHelpers.create_handler_config(
            "test_pipeline_validation_failure",
            "Test Pipeline Validation Failure",
            handler_type=c.Cqrs.HandlerType.COMMAND,
            handler_mode=c.Cqrs.HandlerType.COMMAND,
        )
        config = cast("m.CqrsHandler", config_raw)
        handler = ValidationFailingHandler(config=config)
        result = handler._run_pipeline("test_message", operation="command")
        u.Tests.Result.assert_result_failure_with_error(
            result,
            expected_error="Message validation failed: Validation failed for test",
        )

    def test_handlers_run_pipeline_handler_exception(self) -> None:
        """Test _run_pipeline when handler.handle() raises exception."""

        class ExceptionHandler(h[str, str]):
            def __init__(self, config: m.CqrsHandler) -> None:
                super().__init__(config=config)

            def handle(self, message: str) -> FlextResult[str]:
                _ = message
                error_message = "Test exception in handler"
                raise ValueError(error_message)

        config_raw = FlextTestsUtilities.Tests.HandlerHelpers.create_handler_config(
            "test_pipeline_exception",
            "Test Pipeline Exception",
            handler_type=c.Cqrs.HandlerType.COMMAND,
            handler_mode=c.Cqrs.HandlerType.COMMAND,
        )
        config = cast("m.CqrsHandler", config_raw)
        handler = ExceptionHandler(config=config)
        result = handler._run_pipeline("test_message", operation="command")
        u.Tests.Result.assert_result_failure_with_error(
            result,
            expected_error="Critical handler failure: Test exception in handler",
        )

    def test_handlers_create_from_callable_basic(self) -> None:
        """Test create_from_callable with basic function."""

        def simple_handler(message: str) -> str:
            return f"handled_{message}"

        # Business Rule: create_from_callable accepts HandlerCallable compatible callables
        # simple_handler is compatible with HandlerCallable at runtime
        handler = h.create_from_callable(
            cast("t.HandlerCallable", simple_handler),
            handler_name="simple_handler",
            handler_type=c.Cqrs.HandlerType.COMMAND,
        )
        assert handler is not None
        assert handler.handler_name == "simple_handler"
        result = handler.handle("test")
        u.Tests.Result.assert_success_with_value(
            result,
            "handled_test",
        )

    def test_handlers_create_from_callable_with_flext_result(self) -> None:
        """Test create_from_callable with function returning FlextResult."""

        def result_handler(message: str) -> FlextResult[str]:
            return FlextResult[str].ok(f"result_{message}")

        # Business Rule: create_from_callable accepts HandlerCallable compatible callables
        handler = h.create_from_callable(
            cast("t.HandlerCallable", result_handler),
            handler_name="result_handler",
            handler_type=c.Cqrs.HandlerType.QUERY,
        )
        assert handler.handler_name == "result_handler"
        result = handler.handle("test")
        u.Tests.Result.assert_success_with_value(
            result,
            "result_test",
        )

    def test_handlers_create_from_callable_with_exception(self) -> None:
        """Test create_from_callable with function that raises exception."""

        def failing_handler(message: str) -> str:
            _ = message
            error_message = "Handler failed"
            raise ValueError(error_message)

        # Business Rule: create_from_callable accepts HandlerCallable compatible callables
        handler = h.create_from_callable(
            cast("t.HandlerCallable", failing_handler),
            handler_name="failing_handler",
            handler_type=c.Cqrs.HandlerType.COMMAND,
        )
        result = handler.handle("test")
        u.Tests.Result.assert_result_failure_with_error(
            result,
            expected_error="Handler failed",
        )

    def test_handlers_create_from_callable_invalid_mode(self) -> None:
        """Test create_from_callable with invalid mode."""

        def invalid_handler(message: str) -> str:
            return f"invalid_{message}"

        with pytest.raises(FlextExceptions.ValidationError) as exc_info:
            # Business Rule: create_from_callable accepts HandlerCallable compatible callables
            h.create_from_callable(
                cast("t.HandlerCallable", invalid_handler),
                handler_name="invalid_handler",
                mode="invalid_mode",
            )
        assert "Invalid handler mode: invalid_mode" in str(exc_info.value)

    def test_handlers_execute_method(self) -> None:
        """Test execute method (Layer 1 interface)."""
        config = FlextTestsUtilities.Tests.HandlerHelpers.create_handler_config(
            "test_execute",
            "Test Execute",
        )
        handler = ConcreteTestHandler(config=config)
        result = handler.execute("test_message")
        u.Tests.Result.assert_success_with_value(
            result,
            "processed_test_message",
        )

    def test_handlers_callable_interface(self) -> None:
        """Test __call__ method (callable interface)."""
        config = FlextTestsUtilities.Tests.HandlerHelpers.create_handler_config(
            "test_callable",
            "Test Callable",
        )
        handler = ConcreteTestHandler(config=config)
        result = handler("test_message")
        u.Tests.Result.assert_success_with_value(
            result,
            "processed_test_message",
        )

    def test_handlers_can_handle_method(self) -> None:
        """Test can_handle method."""
        config = FlextTestsUtilities.Tests.HandlerHelpers.create_handler_config(
            "test_can_handle",
            "Test Can Handle",
        )
        handler = ConcreteTestHandler(config=config)
        assert isinstance(handler.can_handle(str), bool)

    def test_handlers_mode_property(self) -> None:
        """Test mode property."""
        config = FlextTestsUtilities.Tests.HandlerHelpers.create_handler_config(
            "test_mode_property",
            "Test Mode Property",
            handler_type=c.Cqrs.HandlerType.COMMAND,
            handler_mode=c.Cqrs.HandlerType.COMMAND,
        )
        handler = ConcreteTestHandler(config=config)
        assert handler.mode == c.Cqrs.HandlerType.COMMAND

    @pytest.mark.parametrize(
        ("handler_type", "handler_mode"),
        [
            (scenario.handler_type, scenario.handler_mode)
            for scenario in HandlerScenarios.HANDLER_TYPES
        ],
        ids=[scenario.name for scenario in HandlerScenarios.HANDLER_TYPES],
    )
    def test_handlers_validate_generic(
        self,
        handler_type: c.Cqrs.HandlerType,
        handler_mode: c.Cqrs.HandlerType,
    ) -> None:
        """Test validate method for various handler types."""
        config = FlextTestsUtilities.Tests.HandlerHelpers.create_handler_config(
            f"test_validate_generic_{handler_type.value}",
            f"Test Validate Generic {handler_type.value.title()}",
            handler_type=handler_type,
            handler_mode=handler_mode,
        )
        handler = ConcreteTestHandler(config=config)
        result = handler.validate("test_message")
        u.Tests.Result.assert_result_success(result)

    @pytest.mark.parametrize(
        ("type_name", "message"),
        HandlerScenarios.VALIDATION_TYPES,
        ids=[t[0] for t in HandlerScenarios.VALIDATION_TYPES],
    )
    def test_handlers_message_validation_types(
        self,
        type_name: str,
        message: object,
    ) -> None:
        """Test message validation with various types."""
        config = FlextTestsUtilities.Tests.HandlerHelpers.create_handler_config(
            f"test_{type_name}_message",
            f"Test {type_name.title()} Message",
        )
        handler = ConcreteTestHandler(config=config)
        # Business Rule: validate accepts AcceptableMessageType compatible objects
        # object is compatible with AcceptableMessageType at runtime
        message_typed = cast("t.AcceptableMessageType", message)
        handler_typed = cast("h[t.AcceptableMessageType, str]", handler)
        result = handler_typed.validate(message_typed)
        assertion_helpers.assert_flext_result_success(result)

    def test_handlers_validate_message_protocol(self) -> None:
        """Test validate_message protocol method."""
        config = FlextTestsUtilities.Tests.HandlerHelpers.create_handler_config(
            "test_validate_message_protocol",
            "Test Validate Message Protocol",
        )
        handler = ConcreteTestHandler(config=config)
        result = handler.validate_message("test_message")
        u.Tests.Result.assert_result_success(result)

    def test_handlers_record_metric(self) -> None:
        """Test record_metric protocol method."""
        config = FlextTestsUtilities.Tests.HandlerHelpers.create_handler_config(
            "test_record_metric",
            "Test Record Metric",
        )
        handler = ConcreteTestHandler(config=config)
        result = handler.record_metric("test_metric", 42.0)
        u.Tests.Result.assert_result_success(result)

    def test_handlers_get_metrics(self) -> None:
        """Test get_metrics protocol method."""
        config = FlextTestsUtilities.Tests.HandlerHelpers.create_handler_config(
            "test_get_metrics",
            "Test Get Metrics",
        )
        handler = ConcreteTestHandler(config=config)
        handler.record_metric("test_metric", 42.0)
        result = handler.get_metrics()
        u.Tests.Result.assert_result_success(result)
        metrics = result.value
        assert isinstance(metrics, dict)
        assert metrics.get("test_metric") == 42.0

    def test_handlers_push_context(self) -> None:
        """Test push_context protocol method."""
        config = FlextTestsUtilities.Tests.HandlerHelpers.create_handler_config(
            "test_push_context",
            "Test Push Context",
        )
        handler = ConcreteTestHandler(config=config)
        context = {"user_id": "123", "operation": "test"}
        # Convert dict[str, t.GeneralValueType] to dict[str, t.GeneralValueType]
        context_typed: dict[str, t.GeneralValueType] = {
            k: cast("t.GeneralValueType", v) for k, v in context.items()
        }
        result = handler.push_context(context_typed)
        u.Tests.Result.assert_result_success(result)

    def test_handlers_pop_context(self) -> None:
        """Test pop_context protocol method."""
        config = FlextTestsUtilities.Tests.HandlerHelpers.create_handler_config(
            "test_pop_context",
            "Test Pop Context",
        )
        handler = ConcreteTestHandler(config=config)
        # Business Rule: push_context accepts dict[str, t.GeneralValueType] compatible mappings
        # dict literal is compatible at runtime
        handler.push_context({"test": "data"})
        result = handler.pop_context()
        u.Tests.Result.assert_result_success(result)

    def test_handlers_pop_context_empty_stack(self) -> None:
        """Test pop_context when stack is empty."""
        config = FlextTestsUtilities.Tests.HandlerHelpers.create_handler_config(
            "test_pop_context_empty",
            "Test Pop Context Empty",
        )
        handler = ConcreteTestHandler(config=config)
        result = handler.pop_context()
        u.Tests.Result.assert_result_success(result)

    def test_handlers_message_with_none_raises_validation_error(self) -> None:
        """Test message validation with None value."""
        config = FlextTestsUtilities.Tests.HandlerHelpers.create_handler_config(
            "test_none_message",
            "Test None Message",
        )
        handler = ConcreteTestHandler(config=config)
        # Business Rule: validate accepts AcceptableMessageType, but None is passed to test error handling
        # None is intentionally passed to test error handling - cast to satisfy type checker
        # The cast allows None to be passed for testing error handling scenarios
        none_message = cast("t.AcceptableMessageType", None)
        handler_typed = cast("h[t.AcceptableMessageType, str]", handler)
        result = handler_typed.validate(none_message)
        u.Tests.Result.assert_result_failure(result)

    def test_handlers_pydantic_model_validation(self) -> None:
        """Test Pydantic model validation."""

        class TestMessage(BaseModel):
            value: str

        config = FlextTestsUtilities.Tests.HandlerHelpers.create_handler_config(
            "test_pydantic_validation",
            "Test Pydantic Validation",
        )
        handler = ConcreteTestHandler(config=config)
        msg = TestMessage(value="test")
        # Business Rule: validate accepts Pydantic BaseModel instances compatible with AcceptableMessageType
        # TestMessage is compatible with AcceptableMessageType at runtime
        msg_typed = cast("t.AcceptableMessageType", msg)
        handler_typed = cast("h[t.AcceptableMessageType, str]", handler)
        result = handler_typed.validate(msg_typed)
        u.Tests.Result.assert_result_success(result)

    def test_handlers_dataclass_message_validation(self) -> None:
        """Test dataclass message validation."""

        @dataclass
        class DataClassMessage:
            value: str
            number: int

        config = FlextTestsUtilities.Tests.HandlerHelpers.create_handler_config(
            "test_dataclass_message",
            "Test Dataclass Message",
        )
        handler = ConcreteTestHandler(config=config)
        msg = DataClassMessage(value="test", number=42)
        # Business Rule: validate accepts dataclass instances compatible with AcceptableMessageType
        # DataClassMessage is compatible with AcceptableMessageType at runtime
        msg_typed = cast("t.AcceptableMessageType", cast("object", msg))
        handler_typed = cast("h[t.AcceptableMessageType, str]", handler)
        result = handler_typed.validate(msg_typed)
        u.Tests.Result.assert_result_success(result)

    def test_handlers_slots_message_validation(self) -> None:
        """Test __slots__ message validation."""

        class SlotsMessage:
            __slots__ = ("number", "value")

            def __init__(self, value: str, number: int) -> None:
                self.value = value
                self.number = number

        config = FlextTestsUtilities.Tests.HandlerHelpers.create_handler_config(
            "test_slots_message",
            "Test Slots Message",
        )
        handler = ConcreteTestHandler(config=config)
        msg = SlotsMessage(value="test", number=42)
        # Business Rule: validate accepts __slots__ class instances compatible with AcceptableMessageType
        # SlotsMessage is compatible with AcceptableMessageType at runtime
        msg_typed = cast("t.AcceptableMessageType", cast("object", msg))
        handler_typed = cast("h[t.AcceptableMessageType, str]", handler)
        result = handler_typed.validate(msg_typed)
        u.Tests.Result.assert_result_success(result)


__all__ = ["TestFlextHandlers"]
