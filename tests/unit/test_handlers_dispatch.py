"""Handler dispatch pipeline tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from flext_tests import h, r

from tests.constants import c
from tests.typings import t
from tests.unit._handlers_support import TestsFlextFlextHandlers
from tests.utilities import u

if TYPE_CHECKING:
    from tests.models import m
    from tests.protocols import p

HANDLER_TYPES = TestsFlextFlextHandlers.HANDLER_TYPES
HandlerTypeScenario = TestsFlextFlextHandlers.HandlerTypeScenario
VALIDATION_TYPES = TestsFlextFlextHandlers.VALIDATION_TYPES


class TestsFlextHandlersDispatch(TestsFlextFlextHandlers):
    def test_handlers_run_pipeline_with_dict_message_command_id(self) -> None:
        class DictHandler(h[t.JsonMapping, t.JsonPayload]):
            @override
            def __init__(self, settings: m.Handler) -> None:
                super().__init__(settings=settings)

            @override
            def execute(
                self,
                message: t.JsonMapping,
            ) -> p.Result[t.JsonPayload]:
                return self.handle(message)

            @override
            def handle(
                self,
                message: t.MappingKV[str, t.JsonValue],
            ) -> p.Result[t.JsonPayload]:
                if not isinstance(message, dict):
                    return r[t.JsonPayload].fail(c.Tests.UNEXPECTED_MESSAGE_TYPE)
                return r[t.JsonPayload].ok(f"processed_{message}")

        settings = u.Tests.create_handler_config(
            "test_pipeline_dict_command_id",
            "Test Pipeline Dict Command ID",
            handler_type=c.HandlerType.COMMAND,
            handler_mode=c.HandlerType.COMMAND,
        )
        handler = DictHandler(settings=settings)
        dict_message = {"command_id": "cmd_123", "data": "test_data"}
        result = handler.execute(dict_message)
        _ = u.Tests.assert_success(result)

    def test_handlers_dispatch_message_mode_validation_error(self) -> None:
        settings = u.Tests.create_handler_config(
            "test_pipeline_mode_error",
            "Test Pipeline Mode Error",
            handler_type=c.HandlerType.COMMAND,
            handler_mode=c.HandlerType.COMMAND,
        )
        handler = self.ConcreteTestHandler(settings=settings)
        result = handler.dispatch_message("test_message", operation="query")
        u.Tests.assert_failure(
            result,
            expected_error="Handler with mode 'command' cannot execute query pipelines",
        )

    def test_handlers_dispatch_message_cannot_handle_message_type(self) -> None:
        class RestrictiveHandler(TestsFlextFlextHandlers.ConcreteTestHandler):
            @override
            def __init__(self, settings: m.Handler) -> None:
                super().__init__(settings=settings)

            @override
            def can_handle(self, message_type: type) -> bool:
                _ = message_type
                return False

        settings = u.Tests.create_handler_config(
            "test_pipeline_cannot_handle",
            "Test Pipeline Cannot Handle",
            handler_type=c.HandlerType.COMMAND,
            handler_mode=c.HandlerType.COMMAND,
        )
        handler = RestrictiveHandler(settings=settings)
        result = handler.dispatch_message("test_message", operation="command")
        u.Tests.assert_failure(
            result,
            expected_error="Handler cannot handle message type str",
        )

    def test_handlers_dispatch_message_validation_failure(self) -> None:
        class ValidationFailingHandler(TestsFlextFlextHandlers.ConcreteTestHandler):
            @override
            def __init__(self, settings: m.Handler) -> None:
                super().__init__(settings=settings)

            @override
            def validate_message(self, data: t.JsonPayload) -> p.Result[bool]:
                _ = data
                return r[bool].fail(c.Tests.VALIDATION_FAILED_FOR_TEST)

        settings = u.Tests.create_handler_config(
            "test_pipeline_validation_failure",
            "Test Pipeline Validation Failure",
            handler_type=c.HandlerType.COMMAND,
            handler_mode=c.HandlerType.COMMAND,
        )
        handler = ValidationFailingHandler(settings=settings)
        result = handler.dispatch_message("test_message", operation="command")
        u.Tests.assert_failure(
            result,
            expected_error="Message validation failed: Validation failed for test",
        )

    def test_handlers_dispatch_message_handler_exception(self) -> None:
        class ExceptionHandler(TestsFlextFlextHandlers.ConcreteTestHandler):
            @override
            def __init__(self, settings: m.Handler) -> None:
                super().__init__(settings=settings)

            @override
            def handle(self, message: t.JsonPayload) -> p.Result[t.JsonPayload]:
                _ = message
                msg = "Test exception in handler"
                raise ValueError(msg)

        settings = u.Tests.create_handler_config(
            "test_pipeline_exception",
            "Test Pipeline Exception",
            handler_type=c.HandlerType.COMMAND,
            handler_mode=c.HandlerType.COMMAND,
        )
        handler = ExceptionHandler(settings=settings)
        result = handler.dispatch_message("test_message", operation="command")
        u.Tests.assert_failure(
            result,
            expected_error="Critical handler failure: Test exception in handler",
        )
