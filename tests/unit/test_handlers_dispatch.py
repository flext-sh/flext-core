"""Behavioral tests for the handler dispatch pipeline public contract."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import pytest

from flext_tests import h, r
from tests.constants import c
from tests.typings import t
from tests.unit._handlers_support import TestsFlextFlextHandlers
from tests.utilities import u

if TYPE_CHECKING:
    from tests.models import m
    from tests.protocols import p


class TestsFlextHandlersDispatch(TestsFlextFlextHandlers):
    """Assert observable dispatch/execute behavior, not implementation details."""

    @staticmethod
    def _command_settings(handler_id: str) -> m.Handler:
        return u.Tests.create_handler_config(
            handler_id,
            handler_id.replace("_", " ").title(),
            handler_type=c.HandlerType.COMMAND,
            handler_mode=c.HandlerType.COMMAND,
        )

    def test_execute_returns_processed_payload_for_dict_message(self) -> None:
        class DictHandler(h[t.JsonMapping, t.JsonPayload]):
            @override
            def execute(self, message: t.JsonMapping) -> p.Result[t.JsonPayload]:
                return self.handle(message)

            @override
            def handle(
                self, message: t.MappingKV[str, t.JsonValue]
            ) -> p.Result[t.JsonPayload]:
                if not isinstance(message, dict):
                    return r[t.JsonPayload].fail(c.Tests.UNEXPECTED_MESSAGE_TYPE)
                return r[t.JsonPayload].ok(f"processed_{message}")

        handler = DictHandler(settings=self._command_settings("dict_command"))
        dict_message = {"command_id": "cmd_123", "data": "test_data"}

        result = handler.execute(dict_message)

        _ = u.Tests.assert_success(result, expected_value=f"processed_{dict_message}")

    def test_execute_returns_processed_payload_for_string_message(self) -> None:
        handler = self.ConcreteTestHandler(
            settings=self._command_settings("string_command")
        )

        result = handler.execute("hello")

        _ = u.Tests.assert_success(result, expected_value="processed_hello")

    def test_execute_rejects_none_message_via_validation(self) -> None:
        handler = self.ConcreteTestHandler(
            settings=self._command_settings("none_command")
        )

        result = handler.execute(None)

        _ = u.Tests.assert_failure(result, expected_error=c.ERR_MESSAGE_CANNOT_BE_NONE)

    def test_dispatch_matching_operation_returns_processed_payload(self) -> None:
        handler = self.ConcreteTestHandler(
            settings=self._command_settings("dispatch_ok")
        )

        result = handler.dispatch_message("test_message", operation="command")

        _ = u.Tests.assert_success(result, expected_value="processed_test_message")

    @pytest.mark.parametrize(
        ("operation", "expected_error"),
        [
            ("query", "Handler with mode 'command' cannot execute query pipelines"),
            ("event", "Handler with mode 'command' cannot execute event pipelines"),
        ],
    )
    def test_dispatch_incompatible_operation_fails(
        self, operation: str, expected_error: str
    ) -> None:
        handler = self.ConcreteTestHandler(
            settings=self._command_settings("dispatch_mode_error")
        )

        result = handler.dispatch_message("test_message", operation=operation)

        _ = u.Tests.assert_failure(result, expected_error=expected_error)

    def test_dispatch_rejects_unhandleable_message_type(self) -> None:
        class RestrictiveHandler(TestsFlextFlextHandlers.ConcreteTestHandler):
            @override
            def can_handle(self, message_type: type) -> bool:
                _ = message_type
                return False

        handler = RestrictiveHandler(
            settings=self._command_settings("dispatch_cannot_handle")
        )

        result = handler.dispatch_message("test_message", operation="command")

        _ = u.Tests.assert_failure(
            result, expected_error="Handler cannot handle message type str"
        )

    def test_dispatch_propagates_validation_failure(self) -> None:
        class ValidationFailingHandler(TestsFlextFlextHandlers.ConcreteTestHandler):
            @override
            def validate_message(self, data: t.JsonPayload) -> p.Result[bool]:
                _ = data
                return r[bool].fail(c.Tests.VALIDATION_FAILED_FOR_TEST)

        handler = ValidationFailingHandler(
            settings=self._command_settings("dispatch_validation_failure")
        )

        result = handler.dispatch_message("test_message", operation="command")

        _ = u.Tests.assert_failure(
            result,
            expected_error="Message validation failed: Validation failed for test",
        )

    def test_dispatch_converts_handler_exception_to_critical_failure(self) -> None:
        class ExceptionHandler(TestsFlextFlextHandlers.ConcreteTestHandler):
            @override
            def handle(self, message: t.JsonPayload) -> p.Result[t.JsonPayload]:
                _ = message
                msg = "Test exception in handler"
                raise ValueError(msg)

        handler = ExceptionHandler(
            settings=self._command_settings("dispatch_exception")
        )

        result = handler.dispatch_message("test_message", operation="command")

        _ = u.Tests.assert_failure(
            result, expected_error="Critical handler failure: Test exception in handler"
        )

    @pytest.mark.parametrize("message_type", [str, int, dict])
    def test_flexible_handler_accepts_any_message_type(
        self, message_type: type
    ) -> None:
        handler = self.ConcreteTestHandler(
            settings=self._command_settings("can_handle")
        )

        assert handler.can_handle(message_type) is True

    def test_validate_message_reports_success_and_failure(self) -> None:
        handler = self.ConcreteTestHandler(
            settings=self._command_settings("validate_message")
        )

        _ = u.Tests.assert_success(
            handler.validate_message("test_message"), expected_value=True
        )
        _ = u.Tests.assert_failure(
            handler.validate_message(None), expected_error=c.ERR_MESSAGE_CANNOT_BE_NONE
        )
