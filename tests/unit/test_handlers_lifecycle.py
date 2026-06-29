"""Handler lifecycle and mode tests."""

from __future__ import annotations

from typing import override

import pytest
from flext_tests import h, r, x

from tests.constants import c
from tests.models import m
from tests.protocols import p
from tests.typings import t
from tests.unit._handlers_support import TestsFlextFlextHandlers
from tests.utilities import u

HANDLER_TYPES = TestsFlextFlextHandlers.HANDLER_TYPES
HandlerTypeScenario = TestsFlextFlextHandlers.HandlerTypeScenario
VALIDATION_TYPES = TestsFlextFlextHandlers.VALIDATION_TYPES


class TestsFlextHandlersLifecycle(TestsFlextFlextHandlers):
    def test_handlers_initialization(self) -> None:
        settings = u.Tests.create_handler_config(
            "test_handler_1",
            "Test Handler 1",
        )
        handlers = self.ConcreteTestHandler(settings=settings)
        assert handlers is not None
        assert isinstance(handlers, h)

    def test_handlers_with_custom_config(self) -> None:
        settings = u.Tests.create_handler_config(
            "test_handler_2",
            "Test Handler 2",
            handler_type=c.HandlerType.QUERY,
            handler_mode=c.HandlerType.QUERY,
        )
        handlers = self.ConcreteTestHandler(settings=settings)
        assert handlers is not None
        assert handlers.mode == c.HandlerType.QUERY
        assert handlers.handler_name == "Test Handler 2"

    def test_handlers_handle_success(self) -> None:
        settings = u.Tests.create_handler_config(
            "test_handler_3",
            "Test Handler 3",
        )
        handler = self.ConcreteTestHandler(settings=settings)
        result = handler.handle("test_message")
        u.Tests.assert_success(result, expected_value="processed_test_message")

    def test_handlers_handle_failure(self) -> None:
        settings = u.Tests.create_handler_config(
            "test_handler_4",
            "Test Handler 4",
        )
        handler = self.FailingTestHandler(settings=settings)
        result = handler.handle("test_message")
        u.Tests.assert_failure(
            result, expected_error="Handler failed for: test_message"
        )

    def test_handlers_config_access(self) -> None:
        settings = u.Tests.create_handler_config(
            "test_handler_5",
            "Test Handler 5",
            handler_type=c.HandlerType.COMMAND,
            handler_mode=c.HandlerType.COMMAND,
        )
        handler = self.ConcreteTestHandler(settings=settings)
        assert handler.handler_name == "Test Handler 5"
        assert handler.mode == c.HandlerType.COMMAND

    def test_handlers_execution_context(self) -> None:
        settings = u.Tests.create_handler_config(
            "test_handler_6",
            "Test Handler 6",
        )
        handler: TestsFlextFlextHandlers.ConcreteTestHandler = self.ConcreteTestHandler(
            settings=settings
        )
        result = handler.execute("test_message")
        u.Tests.assert_success(result, expected_value="processed_test_message")

    def test_handlers_different_types(self) -> None:
        class IntHandler(h[t.JsonPayload, t.JsonPayload]):
            def __init__(self, *, settings: m.Handler | None = None) -> None:
                super().__init__(settings=settings)

            @override
            def handle(self, message: t.JsonPayload) -> p.Result[t.JsonPayload]:
                if not isinstance(message, int):
                    return r[t.JsonPayload].fail(c.Tests.UNEXPECTED_MESSAGE_TYPE)
                return r[t.JsonPayload].ok(f"processed_{message}")

        settings = u.Tests.create_handler_config(
            "test_handler_10",
            "Test Handler 10",
        )
        handler = IntHandler(settings=settings)
        result = handler.handle(42)
        u.Tests.assert_success(result, expected_value="processed_42")

    @pytest.mark.parametrize(
        "scenario",
        HANDLER_TYPES,
        ids=lambda s: s.name,
    )
    def test_handlers_types(self, scenario: HandlerTypeScenario) -> None:
        settings = u.Tests.create_handler_config(
            f"test_{scenario.name}_handler",
            f"Test {scenario.name.title()} Handler",
            handler_type=scenario.handler_type,
            handler_mode=scenario.handler_mode,
        )
        handler = self.ConcreteTestHandler(settings=settings)
        assert handler.mode == scenario.handler_mode

    def test_handlers_with_metadata(self) -> None:
        settings = u.Tests.create_handler_config(
            "test_handler_with_metadata",
            "Test Handler With Metadata",
            metadata=m.Metadata(attributes={"test_key": "test_value", "priority": 1}),
        )
        handler = self.ConcreteTestHandler(settings=settings)
        result = handler.execute("test_message")
        u.Tests.assert_success(result, expected_value="processed_test_message")

    def test_handlers_with_timeout(self) -> None:
        settings = u.Tests.create_handler_config(
            "test_handler_with_timeout",
            "Test Handler With Timeout",
            command_timeout=60,
        )
        handler = self.ConcreteTestHandler(settings=settings)
        result = handler.execute("test_message")
        u.Tests.assert_success(result, expected_value="processed_test_message")

    def test_handlers_with_retry_config(self) -> None:
        settings = u.Tests.create_handler_config(
            "test_handler_with_retry",
            "Test Handler With Retry",
            max_command_retries=3,
        )
        handler = self.ConcreteTestHandler(settings=settings)
        result = handler.execute("test_message")
        u.Tests.assert_success(result, expected_value="processed_test_message")

    def test_handlers_inheritance_chain(self) -> None:
        settings = u.Tests.create_handler_config(
            "test_inheritance_handler",
            "Test Inheritance Handler",
        )
        handler = self.ConcreteTestHandler(settings=settings)
        assert isinstance(handler, x)
