"""Handler callable factory tests."""

from __future__ import annotations

import pytest

from tests import c, e, h, r, t, u
from tests.unit._handlers_support import TestsFlextFlextHandlers

HANDLER_TYPES = TestsFlextFlextHandlers.HANDLER_TYPES
HandlerTypeScenario = TestsFlextFlextHandlers.HandlerTypeScenario
VALIDATION_TYPES = TestsFlextFlextHandlers.VALIDATION_TYPES


class TestsFlextHandlersFactory(TestsFlextFlextHandlers):
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
        u.Tests.assert_success(result, expected_value="handled_test")

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
        u.Tests.assert_success(result, expected_value="result_test")

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
        u.Tests.assert_failure(result, expected_error="Handler failed")

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
        settings = u.Tests.create_handler_config(
            "test_execute",
            "Test Execute",
        )
        handler = self.ConcreteTestHandler(settings=settings)
        result = handler.execute("test_message")
        u.Tests.assert_success(result, expected_value="processed_test_message")

    def test_handlers_can_handle_method(self) -> None:
        settings = u.Tests.create_handler_config(
            "test_can_handle",
            "Test Can Handle",
        )
        handler = self.ConcreteTestHandler(settings=settings)
        assert isinstance(handler.can_handle(str), bool)

    def test_handlers_mode_property(self) -> None:
        settings = u.Tests.create_handler_config(
            "test_mode_property",
            "Test Mode Property",
            handler_type=c.HandlerType.COMMAND,
            handler_mode=c.HandlerType.COMMAND,
        )
        handler = self.ConcreteTestHandler(settings=settings)
        assert handler.mode == c.HandlerType.COMMAND
