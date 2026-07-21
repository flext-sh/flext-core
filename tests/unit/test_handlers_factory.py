"""Behavioral tests for the handler callable factory public contract."""

from __future__ import annotations

from typing import cast

import pytest

from flext_tests import h, r
from tests import c, t, u
from tests.unit._handlers_support import TestsFlextFlextHandlers


class TestsFlextCoreHandlersFactory(TestsFlextFlextHandlers):
    """Assert observable behavior of ``h.create_from_callable`` and handlers."""

    def test_callable_result_is_wrapped_in_success(self) -> None:
        # Arrange
        def simple_handler(message: t.Scalar) -> t.Scalar:
            return f"handled_{message}"

        # Act
        handler = h.create_from_callable(
            simple_handler,
            handler_name="simple_handler",
            handler_type=c.HandlerType.COMMAND,
        )

        # Assert
        assert handler.handler_name == "simple_handler"
        u.Tests.assert_success(handler.handle("test"), expected_value="handled_test")

    def test_callable_returning_result_is_passed_through(self) -> None:
        # Arrange
        def result_handler(message: t.Scalar) -> t.Scalar:
            return r[t.Scalar].ok(f"result_{message}").value

        # Act
        handler = h.create_from_callable(
            result_handler,
            handler_name="result_handler",
            handler_type=c.HandlerType.QUERY,
        )

        # Assert
        u.Tests.assert_success(handler.handle("test"), expected_value="result_test")

    def test_callable_raising_produces_failure_result(self) -> None:
        # Arrange
        def failing_handler(message: t.Scalar) -> t.Scalar:
            _ = message
            msg = "Handler failed"
            raise ValueError(msg)

        # Act
        handler = h.create_from_callable(
            failing_handler,
            handler_name="failing_handler",
            handler_type=c.HandlerType.COMMAND,
        )
        result = handler.handle("test")

        # Assert
        u.Tests.assert_failure(result, expected_error="Handler failed")

    def test_invalid_mode_raises_validation_error(self) -> None:
        # Arrange
        def invalid_handler(message: t.Scalar) -> t.Scalar:
            return f"invalid_{message}"

        # Act / Assert
        with pytest.raises(c.ValidationError):
            h.create_from_callable(
                invalid_handler,
                handler_name="invalid_handler",
                handler_type=cast("c.HandlerType", "invalid_mode"),
            )

    def test_handler_name_defaults_to_callable_name(self) -> None:
        # Arrange
        def named_callable(message: t.Scalar) -> t.Scalar:
            return f"named_{message}"

        # Act
        handler = h.create_from_callable(named_callable)

        # Assert
        assert handler.handler_name == "named_callable"

    def test_handler_config_overrides_name_and_type(self) -> None:
        # Arrange
        def any_callable(message: t.Scalar) -> t.Scalar:
            return f"any_{message}"

        config = u.Tests.create_handler_config(
            "cfg_id",
            "ConfiguredName",
            handler_type=c.HandlerType.EVENT,
            handler_mode=c.HandlerType.EVENT,
        )

        # Act
        handler = h.create_from_callable(any_callable, handler_config=config)

        # Assert
        assert handler.handler_name == "ConfiguredName"
        assert handler.mode == c.HandlerType.EVENT

    @pytest.mark.parametrize("scenario", TestsFlextFlextHandlers.HANDLER_TYPES)
    def test_mode_reflects_requested_handler_type(
        self, scenario: TestsFlextFlextHandlers.HandlerTypeScenario
    ) -> None:
        # Arrange
        def any_callable(message: t.Scalar) -> t.Scalar:
            return f"any_{message}"

        # Act
        by_type = h.create_from_callable(
            any_callable, handler_type=scenario.handler_type
        )

        # Assert
        assert by_type.mode == scenario.handler_mode

    def test_execute_returns_processed_success(self) -> None:
        # Arrange
        settings = u.Tests.create_handler_config("test_execute", "Test Execute")
        handler = self.ConcreteTestHandler(settings=settings)

        # Act
        result = handler.execute("test_message")

        # Assert
        u.Tests.assert_success(result, expected_value="processed_test_message")

    def test_execute_propagates_validation_failure(self) -> None:
        # Arrange
        settings = u.Tests.create_handler_config("test_execute", "Test Execute")
        handler = self.ConcreteTestHandler(settings=settings)

        # Act
        result = handler.execute(None)

        # Assert
        u.Tests.assert_failure(result, expected_error=c.ERR_MESSAGE_CANNOT_BE_NONE)

    def test_handle_rejects_non_string_message(self) -> None:
        # Arrange
        settings = u.Tests.create_handler_config("test_reject", "Test Reject")
        handler = self.ConcreteTestHandler(settings=settings)

        # Act
        result = handler.handle(42)

        # Assert
        u.Tests.assert_failure(result, expected_error=c.Tests.UNEXPECTED_MESSAGE_TYPE)

    def test_dispatch_message_runs_pipeline_to_success(self) -> None:
        # Arrange
        settings = u.Tests.create_handler_config(
            "test_dispatch",
            "Test Dispatch",
            handler_type=c.HandlerType.COMMAND,
            handler_mode=c.HandlerType.COMMAND,
        )
        handler = self.ConcreteTestHandler(settings=settings)

        # Act
        result = handler.dispatch_message("payload")

        # Assert
        u.Tests.assert_success(result, expected_value="processed_payload")

    def test_dispatch_message_rejects_incompatible_operation(self) -> None:
        # Arrange
        settings = u.Tests.create_handler_config(
            "test_dispatch",
            "Test Dispatch",
            handler_type=c.HandlerType.COMMAND,
            handler_mode=c.HandlerType.COMMAND,
        )
        handler = self.ConcreteTestHandler(settings=settings)

        # Act
        result = handler.dispatch_message("payload", operation=c.HandlerMode.QUERY)

        # Assert
        assert result.failure
        assert result.error is not None

    def test_can_handle_message_type(self) -> None:
        # Arrange
        settings = u.Tests.create_handler_config("test_can_handle", "Test Can Handle")
        handler = self.ConcreteTestHandler(settings=settings)

        # Act / Assert
        assert handler.can_handle(str) is True

    def test_mode_reflects_configured_handler_type(self) -> None:
        # Arrange
        settings = u.Tests.create_handler_config(
            "test_mode_property",
            "Test Mode Property",
            handler_type=c.HandlerType.COMMAND,
            handler_mode=c.HandlerType.COMMAND,
        )
        handler = self.ConcreteTestHandler(settings=settings)

        # Act / Assert
        assert handler.mode == c.HandlerType.COMMAND

    @pytest.mark.parametrize(
        ("payload", "expected_success"), [("non_empty", True), ("", False)]
    )
    def test_validate_message_reports_payload_validity(
        self, payload: str, expected_success: bool
    ) -> None:
        # Arrange
        settings = u.Tests.create_handler_config("test_validate", "Test Validate")
        handler = self.ValidationTestHandler(settings=settings)

        # Act
        result = handler.validate_message(payload)

        # Assert
        assert result.success is expected_success

    def test_failing_handler_returns_descriptive_failure(self) -> None:
        # Arrange
        settings = u.Tests.create_handler_config("test_failing", "Test Failing")
        handler = self.FailingTestHandler(settings=settings)

        # Act
        result = handler.handle("payload")

        # Assert
        u.Tests.assert_failure(result, expected_error="Handler failed for: payload")
