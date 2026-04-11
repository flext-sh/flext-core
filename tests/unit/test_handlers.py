from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Annotated, ClassVar, override

import pytest
from hypothesis import given, strategies as st
from pydantic import ConfigDict, Field

from flext_core import e, h, r, x
from flext_tests import tm
from tests import c, m, t, u


class TestFlextHandlers:
    class ConcreteTestHandler(h[t.ValueOrModel, t.ValueOrModel]):
        """Test handler for string messages."""

        def __init__(self, *, settings: m.Handler | None = None) -> None:
            super().__init__(settings=settings)

        @override
        def handle(self, message: t.ValueOrModel) -> r[t.ValueOrModel]:
            if not isinstance(message, str):
                return r[t.ValueOrModel].fail(
                    c.Core.Tests.TestErrors.UNEXPECTED_MESSAGE_TYPE
                )
            return r[t.ValueOrModel].ok(f"processed_{message}")

    class ValidationTestHandler(h[t.ValueOrModel, t.ValueOrModel]):
        """Test handler for validation."""

        def __init__(self, *, settings: m.Handler | None = None) -> None:
            super().__init__(settings=settings)

        @override
        def handle(self, message: t.ValueOrModel) -> r[t.ValueOrModel]:
            return r[t.ValueOrModel].ok(f"processed_{message}")

    class FailingTestHandler(h[t.ValueOrModel, t.ValueOrModel]):
        """Test handler that fails."""

        def __init__(self, *, settings: m.Handler | None = None) -> None:
            super().__init__(settings=settings)

        @override
        def handle(self, message: t.ValueOrModel) -> r[t.ValueOrModel]:
            if not isinstance(message, str):
                return r[t.ValueOrModel].fail(
                    c.Core.Tests.TestErrors.UNEXPECTED_MESSAGE_TYPE
                )
            return r[t.ValueOrModel].fail(f"Handler failed for: {message}")

    class HandlerTypeScenario(m.Value):
        """Scenario for handler types."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
        name: Annotated[str, Field(description="Handler type scenario name")]
        handler_type: Annotated[c.HandlerType, Field(description="Type")]
        handler_mode: Annotated[c.HandlerType, Field(description="Mode")]

    HANDLER_TYPES: ClassVar[Sequence[HandlerTypeScenario]] = [
        HandlerTypeScenario(
            name="command",
            handler_type=c.HandlerType.COMMAND,
            handler_mode=c.HandlerType.COMMAND,
        ),
        HandlerTypeScenario(
            name="query",
            handler_type=c.HandlerType.QUERY,
            handler_mode=c.HandlerType.QUERY,
        ),
        HandlerTypeScenario(
            name="event",
            handler_type=c.HandlerType.EVENT,
            handler_mode=c.HandlerType.EVENT,
        ),
        HandlerTypeScenario(
            name="saga",
            handler_type=c.HandlerType.SAGA,
            handler_mode=c.HandlerType.SAGA,
        ),
    ]

    VALIDATION_TYPES: ClassVar[Sequence[tuple[str, t.NormalizedValue]]] = [
        ("str", "test_message"),
        ("int", 42),
        ("float", math.pi),
        ("bool", True),
        ("dict", {"key": "value", "number": 42}),
    ]

    def test_handlers_initialization(self) -> None:
        settings = u.Core.Tests.create_handler_config(
            "test_handler_1",
            "Test Handler 1",
        )
        handlers = self.ConcreteTestHandler(settings=settings)
        assert handlers is not None
        assert isinstance(handlers, h)

    def test_handlers_with_custom_config(self) -> None:
        settings = u.Core.Tests.create_handler_config(
            "test_handler_2",
            "Test Handler 2",
            handler_type=c.HandlerType.QUERY,
            handler_mode=c.HandlerType.QUERY,
        )
        handlers = self.ConcreteTestHandler(settings=settings)
        assert handlers is not None
        assert handlers._config_model.handler_type == c.HandlerType.QUERY

    def test_handlers_handle_success(self) -> None:
        settings = u.Core.Tests.create_handler_config(
            "test_handler_3",
            "Test Handler 3",
        )
        handler = self.ConcreteTestHandler(settings=settings)
        result = handler.handle("test_message")
        u.Core.Tests.assert_success_with_value(result, "processed_test_message")

    def test_handlers_handle_failure(self) -> None:
        settings = u.Core.Tests.create_handler_config(
            "test_handler_4",
            "Test Handler 4",
        )
        handler = self.FailingTestHandler(settings=settings)
        result = handler.handle("test_message")
        u.Core.Tests.assert_result_failure_with_error(
            result,
            expected_error="Handler failed for: test_message",
        )

    def test_handlers_config_access(self) -> None:
        settings = u.Core.Tests.create_handler_config(
            "test_handler_5",
            "Test Handler 5",
            handler_type=c.HandlerType.COMMAND,
            handler_mode=c.HandlerType.COMMAND,
        )
        handler = self.ConcreteTestHandler(settings=settings)
        assert handler._config_model.handler_id == "test_handler_5"
        assert handler._config_model.handler_name == "Test Handler 5"
        assert handler._config_model.handler_type == c.HandlerType.COMMAND

    def test_handlers_execution_context(self) -> None:
        settings = u.Core.Tests.create_handler_config(
            "test_handler_6",
            "Test Handler 6",
        )
        handler = self.ConcreteTestHandler(settings=settings)
        assert handler._execution_context is not None

    def test_handlers_different_types(self) -> None:
        class IntHandler(h[t.ValueOrModel, t.ValueOrModel]):
            def __init__(self, *, settings: m.Handler | None = None) -> None:
                super().__init__(settings=settings)

            @override
            def handle(self, message: t.ValueOrModel) -> r[t.ValueOrModel]:
                if not isinstance(message, int):
                    return r[t.ValueOrModel].fail(
                        c.Core.Tests.TestErrors.UNEXPECTED_MESSAGE_TYPE
                    )
                return r[t.ValueOrModel].ok(f"processed_{message}")

        settings = u.Core.Tests.create_handler_config(
            "test_handler_10",
            "Test Handler 10",
        )
        handler = IntHandler(settings=settings)
        result = handler.handle(42)
        u.Core.Tests.assert_success_with_value(result, "processed_42")

    @pytest.mark.parametrize(
        "scenario",
        HANDLER_TYPES,
        ids=lambda s: s.name,
    )
    def test_handlers_types(self, scenario: HandlerTypeScenario) -> None:
        settings = u.Core.Tests.create_handler_config(
            f"test_{scenario.name}_handler",
            f"Test {scenario.name.title()} Handler",
            handler_type=scenario.handler_type,
            handler_mode=scenario.handler_mode,
        )
        handler = self.ConcreteTestHandler(settings=settings)
        assert handler._config_model.handler_type == scenario.handler_type
        assert handler._config_model.handler_mode == scenario.handler_mode

    def test_handlers_with_metadata(self) -> None:
        settings = u.Core.Tests.create_handler_config(
            "test_handler_with_metadata",
            "Test Handler With Metadata",
            metadata=m.Metadata(attributes={"test_key": "test_value", "priority": 1}),
        )
        handler = self.ConcreteTestHandler(settings=settings)
        assert handler._config_model.metadata is not None
        assert handler._config_model.metadata.attributes["test_key"] == "test_value"

    def test_handlers_with_timeout(self) -> None:
        settings = u.Core.Tests.create_handler_config(
            "test_handler_with_timeout",
            "Test Handler With Timeout",
            command_timeout=60,
        )
        handler = self.ConcreteTestHandler(settings=settings)
        assert handler._config_model.command_timeout == 60

    def test_handlers_with_retry_config(self) -> None:
        settings = u.Core.Tests.create_handler_config(
            "test_handler_with_retry",
            "Test Handler With Retry",
            max_command_retries=3,
        )
        handler = self.ConcreteTestHandler(settings=settings)
        assert handler._config_model.max_command_retries == 3

    def test_handlers_inheritance_chain(self) -> None:
        settings = u.Core.Tests.create_handler_config(
            "test_inheritance_handler",
            "Test Inheritance Handler",
        )
        handler = self.ConcreteTestHandler(settings=settings)
        assert isinstance(handler, x)

    def test_handlers_run_pipeline_with_dict_message_command_id(self) -> None:
        class DictHandler(h[t.ValueOrModel, t.ValueOrModel]):
            @override
            def __init__(self, settings: m.Handler) -> None:
                super().__init__(settings=settings)

            @override
            def handle(self, message: t.ValueOrModel) -> r[t.ValueOrModel]:
                if not isinstance(message, dict):
                    return r[t.ValueOrModel].fail(
                        c.Core.Tests.TestErrors.UNEXPECTED_MESSAGE_TYPE
                    )
                return r[t.ValueOrModel].ok(f"processed_{message}")

        settings = u.Core.Tests.create_handler_config(
            "test_pipeline_dict_command_id",
            "Test Pipeline Dict Command ID",
            handler_type=c.HandlerType.COMMAND,
            handler_mode=c.HandlerType.COMMAND,
        )
        handler = DictHandler(settings=settings)
        dict_message: t.MutableContainerMapping = {
            "command_id": "cmd_123",
            "data": "test_data",
        }
        result = handler._run_pipeline(dict_message, operation="command")
        _ = u.Core.Tests.assert_success(result)

    def test_handlers_run_pipeline_mode_validation_error(self) -> None:
        settings = u.Core.Tests.create_handler_config(
            "test_pipeline_mode_error",
            "Test Pipeline Mode Error",
            handler_type=c.HandlerType.COMMAND,
            handler_mode=c.HandlerType.COMMAND,
        )
        handler = self.ConcreteTestHandler(settings=settings)
        result = handler._run_pipeline("test_message", operation="query")
        u.Core.Tests.assert_result_failure_with_error(
            result,
            expected_error="Handler with mode 'command' cannot execute query pipelines",
        )

    def test_handlers_run_pipeline_cannot_handle_message_type(self) -> None:
        class RestrictiveHandler(h[t.ValueOrModel, t.ValueOrModel]):
            @override
            def __init__(self, settings: m.Handler) -> None:
                super().__init__(settings=settings)

            @override
            def can_handle(self, message_type: type) -> bool:
                _ = message_type
                return False

            @override
            def handle(self, message: t.ValueOrModel) -> r[t.ValueOrModel]:
                if not isinstance(message, str):
                    return r[t.ValueOrModel].fail(
                        c.Core.Tests.TestErrors.UNEXPECTED_MESSAGE_TYPE
                    )
                return r[t.ValueOrModel].ok(f"processed_{message}")

        settings = u.Core.Tests.create_handler_config(
            "test_pipeline_cannot_handle",
            "Test Pipeline Cannot Handle",
            handler_type=c.HandlerType.COMMAND,
            handler_mode=c.HandlerType.COMMAND,
        )
        handler = RestrictiveHandler(settings=settings)
        result = handler._run_pipeline("test_message", operation="command")
        u.Core.Tests.assert_result_failure_with_error(
            result,
            expected_error="Handler cannot handle message type str",
        )

    def test_handlers_run_pipeline_validation_failure(self) -> None:
        class ValidationFailingHandler(h[t.ValueOrModel, t.ValueOrModel]):
            @override
            def __init__(self, settings: m.Handler) -> None:
                super().__init__(settings=settings)

            @override
            def validate_message(self, data: t.ValueOrModel) -> r[bool]:
                _ = data
                return r[bool].fail(c.Core.Tests.TestErrors.VALIDATION_FAILED_FOR_TEST)

            @override
            def handle(self, message: t.ValueOrModel) -> r[t.ValueOrModel]:
                if not isinstance(message, str):
                    return r[t.ValueOrModel].fail(
                        c.Core.Tests.TestErrors.UNEXPECTED_MESSAGE_TYPE
                    )
                return r[t.ValueOrModel].ok(f"processed_{message}")

        settings = u.Core.Tests.create_handler_config(
            "test_pipeline_validation_failure",
            "Test Pipeline Validation Failure",
            handler_type=c.HandlerType.COMMAND,
            handler_mode=c.HandlerType.COMMAND,
        )
        handler = ValidationFailingHandler(settings=settings)
        result = handler._run_pipeline("test_message", operation="command")
        u.Core.Tests.assert_result_failure_with_error(
            result,
            expected_error="Message validation failed: Validation failed for test",
        )

    def test_handlers_run_pipeline_handler_exception(self) -> None:
        class ExceptionHandler(h[t.ValueOrModel, t.ValueOrModel]):
            @override
            def __init__(self, settings: m.Handler) -> None:
                super().__init__(settings=settings)

            @override
            def handle(self, message: t.ValueOrModel) -> r[t.ValueOrModel]:
                _ = message
                msg = "Test exception in handler"
                raise ValueError(msg)

        settings = u.Core.Tests.create_handler_config(
            "test_pipeline_exception",
            "Test Pipeline Exception",
            handler_type=c.HandlerType.COMMAND,
            handler_mode=c.HandlerType.COMMAND,
        )
        handler = ExceptionHandler(settings=settings)
        result = handler._run_pipeline("test_message", operation="command")
        u.Core.Tests.assert_result_failure_with_error(
            result,
            expected_error="Critical handler failure: Test exception in handler",
        )

    def test_handlers_create_from_callable_basic(self) -> None:
        def simple_handler(message: t.Scalar) -> t.Scalar:
            return f"handled_{message}"

        handler = h.create_from_callable(
            simple_handler,
            handler_name="simple_handler",
            handler_type=c.HandlerType.COMMAND,
        )
        assert handler is not None
        assert handler.handler_name == "simple_handler"
        result = handler.handle("test")
        u.Core.Tests.assert_success_with_value(result, "handled_test")

    def test_handlers_create_from_callable_with_flext_result(self) -> None:
        def result_handler(message: t.Scalar) -> t.Scalar:
            return r[t.Scalar].ok(f"result_{message}").value

        handler = h.create_from_callable(
            result_handler,
            handler_name="result_handler",
            handler_type=c.HandlerType.QUERY,
        )
        assert handler.handler_name == "result_handler"
        result = handler.handle("test")
        u.Core.Tests.assert_success_with_value(result, "result_test")

    def test_handlers_create_from_callable_with_exception(self) -> None:
        def failing_handler(message: t.Scalar) -> t.Scalar:
            _ = message
            msg = "Handler failed"
            raise ValueError(msg)

        handler = h.create_from_callable(
            failing_handler,
            handler_name="failing_handler",
            handler_type=c.HandlerType.COMMAND,
        )
        result = handler.handle("test")
        u.Core.Tests.assert_result_failure_with_error(
            result,
            expected_error="Handler failed",
        )

    def test_handlers_create_from_callable_invalid_mode(self) -> None:
        def invalid_handler(message: t.Scalar) -> t.Scalar:
            return f"invalid_{message}"

        with pytest.raises(e.ValidationError) as exc_info:
            h.create_from_callable(
                invalid_handler,
                handler_name="invalid_handler",
                mode="invalid_mode",
            )
        assert "Invalid handler mode: invalid_mode" in str(exc_info.value)

    def test_handlers_execute_method(self) -> None:
        settings = u.Core.Tests.create_handler_config(
            "test_execute",
            "Test Execute",
        )
        handler = self.ConcreteTestHandler(settings=settings)
        result = handler.execute("test_message")
        u.Core.Tests.assert_success_with_value(result, "processed_test_message")

    def test_handlers_can_handle_method(self) -> None:
        settings = u.Core.Tests.create_handler_config(
            "test_can_handle",
            "Test Can Handle",
        )
        handler = self.ConcreteTestHandler(settings=settings)
        assert isinstance(handler.can_handle(str), bool)

    def test_handlers_mode_property(self) -> None:
        settings = u.Core.Tests.create_handler_config(
            "test_mode_property",
            "Test Mode Property",
            handler_type=c.HandlerType.COMMAND,
            handler_mode=c.HandlerType.COMMAND,
        )
        handler = self.ConcreteTestHandler(settings=settings)
        assert handler.mode == c.HandlerType.COMMAND

    @pytest.mark.parametrize(
        ("handler_type", "handler_mode"),
        [(scenario.handler_type, scenario.handler_mode) for scenario in HANDLER_TYPES],
        ids=[scenario.name for scenario in HANDLER_TYPES],
    )
    def test_handlers_validate_generic(
        self,
        handler_type: c.HandlerType,
        handler_mode: c.HandlerType,
    ) -> None:
        handler_type_name = str(handler_type)
        settings = u.Core.Tests.create_handler_config(
            f"test_validate_generic_{handler_type_name}",
            f"Test Validate Generic {handler_type_name.title()}",
            handler_type=handler_type,
            handler_mode=handler_mode,
        )
        handler = self.ConcreteTestHandler(settings=settings)
        result = handler.validate_message("test_message")
        _ = u.Core.Tests.assert_success(result)

    @pytest.mark.parametrize(
        ("type_name", "message"),
        VALIDATION_TYPES,
        ids=[item[0] for item in VALIDATION_TYPES],
    )
    def test_handlers_message_validation_types(
        self,
        type_name: str,
        message: t.NormalizedValue,
    ) -> None:
        settings = u.Core.Tests.create_handler_config(
            f"test_{type_name}_message",
            f"Test {type_name.title()} Message",
        )
        handler = self.ValidationTestHandler(settings=settings)
        result = handler.validate_message(message)
        _ = u.Core.Tests.assert_success(result)

    def test_handlers_record_metric(self) -> None:
        settings = u.Core.Tests.create_handler_config(
            "test_record_metric",
            "Test Record Metric",
        )
        handler = self.ConcreteTestHandler(settings=settings)
        result = handler.record_metric("test_metric", 42.0)
        _ = u.Core.Tests.assert_success(result)

    def test_handlers_push_context(self) -> None:
        settings = u.Core.Tests.create_handler_config(
            "test_push_context",
            "Test Push Context",
        )
        handler = self.ConcreteTestHandler(settings=settings)
        context_typed: t.ContainerMapping = {
            "user_id": "123",
            "operation": "test",
        }
        result = handler.push_context(context_typed)
        _ = u.Core.Tests.assert_success(result)

    def test_handlers_pop_context(self) -> None:
        settings = u.Core.Tests.create_handler_config(
            "test_pop_context",
            "Test Pop Context",
        )
        handler = self.ConcreteTestHandler(settings=settings)
        handler.push_context({"test": "data"})
        result = handler.pop_context()
        _ = u.Core.Tests.assert_success(result)

    def test_handlers_pop_context_empty_stack(self) -> None:
        settings = u.Core.Tests.create_handler_config(
            "test_pop_context_empty",
            "Test Pop Context Empty",
        )
        handler = self.ConcreteTestHandler(settings=settings)
        result = handler.pop_context()
        _ = u.Core.Tests.assert_success(result)

    def test_handlers_message_with_none_raises_validation_error(self) -> None:
        settings = u.Core.Tests.create_handler_config(
            "test_none_message",
            "Test None Message",
        )
        handler = self.ValidationTestHandler(settings=settings)
        result = handler.validate_message(None)
        _ = u.Core.Tests.assert_failure(result)

    def test_handlers_pydantic_model_validation(self) -> None:
        class TestMessage(m.Value):
            value: str

        settings = u.Core.Tests.create_handler_config(
            "test_pydantic_validation",
            "Test Pydantic Validation",
        )
        handler = self.ValidationTestHandler(settings=settings)
        msg = TestMessage(value="test")
        result = handler.validate_message(msg)
        _ = u.Core.Tests.assert_success(result)

    def test_handlers_dataclass_message_validation(self) -> None:
        class DataClassMessage(m.Value):
            value: Annotated[str, Field(description="Message value")]
            number: Annotated[int, Field(description="Message number")]

        settings = u.Core.Tests.create_handler_config(
            "test_dataclass_message",
            "Test Dataclass Message",
        )
        handler = self.ValidationTestHandler(settings=settings)
        msg = DataClassMessage(value="test", number=42)
        result = handler.validate_message(msg)
        _ = u.Core.Tests.assert_success(result)

    def test_handlers_slots_message_validation(self) -> None:
        class SlotsMessage(m.Value):
            model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
            value: str
            number: int

        settings = u.Core.Tests.create_handler_config(
            "test_slots_message",
            "Test Slots Message",
        )
        handler = self.ValidationTestHandler(settings=settings)
        msg = SlotsMessage(value="test", number=42)
        result = handler.validate_message(msg)
        _ = u.Core.Tests.assert_success(result)

    @given(st.text(min_size=1))
    def test_create_from_callable_hypothesis(self, handler_name: str) -> None:
        """Property: create_from_callable works with any non-empty name."""
        handler = h.create_from_callable(
            handler_callable=lambda value: str(value),
            handler_name=handler_name,
        )
        tm.that(handler.handler_name, eq=handler_name)
        tm.ok(handler.execute("x"), eq="x")


__all__ = ["TestFlextHandlers"]
