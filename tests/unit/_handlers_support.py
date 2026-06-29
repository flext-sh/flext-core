"""Shared handler classes and scenarios for split unit tests."""

from __future__ import annotations

import math
from collections.abc import (
    Sequence,
)
from typing import Annotated, ClassVar, override

from tests import c, h, m, p, r, t


class TestsFlextFlextHandlers:
    class ConcreteTestHandler(h[t.JsonPayload, t.JsonPayload]):
        """Test handler for string messages."""

        def __init__(self, *, settings: m.Handler | None = None) -> None:
            super().__init__(settings=settings)

        @override
        def dispatch_message(
            self,
            message: t.JsonPayload,
            operation: str = c.DEFAULT_HANDLER_MODE,
        ) -> p.Result[t.JsonPayload]:
            handler_mode = getattr(
                self._config_model.handler_mode,
                "value",
                self._config_model.handler_mode,
            )
            valid_operations = {
                c.DEFAULT_HANDLER_MODE,
                c.HandlerMode.QUERY,
                c.HandlerType.EVENT.value,
            }
            if operation != handler_mode and operation in valid_operations:
                error_msg = c.ERR_HANDLER_INCOMPATIBLE_PIPELINE_MODE.format(
                    handler_mode=handler_mode,
                    operation=operation,
                )
                return r[t.JsonPayload].fail_op(
                    "validate handler pipeline mode",
                    error_msg,
                )
            message_type = message.__class__
            if not self.can_handle(message_type):
                error_msg = c.ERR_HANDLER_CANNOT_HANDLE_MESSAGE_TYPE.format(
                    type_name=message_type.__name__,
                )
                return r[t.JsonPayload].fail_op(
                    "validate handler message type",
                    error_msg,
                )
            validation = self.validate_message(message)
            if validation.failure:
                error_detail = validation.error or c.ERR_VALIDATION_FAILED
                error_msg = c.ERR_HANDLER_MESSAGE_VALIDATION_FAILED.format(
                    error=error_detail,
                )
                return r[t.JsonPayload].fail_op(
                    "validate handler message",
                    error_msg,
                )
            try:
                return self.handle(message)
            except c.EXC_BROAD_RUNTIME as exc:
                return r[t.JsonPayload].fail_op(
                    "run handler pipeline",
                    c.ERR_HANDLER_CRITICAL_FAILURE.format(error=str(exc)),
                )

        @override
        def execute(self, message: t.JsonPayload) -> p.Result[t.JsonPayload]:
            validation = self.validate_message(message)
            if validation.failure:
                return r[t.JsonPayload].fail_op(
                    "execute handler validation",
                    validation.error or c.ERR_VALIDATION_FAILED,
                )
            return self.handle(message)

        @override
        def handle(self, message: t.JsonPayload) -> p.Result[t.JsonPayload]:
            if not isinstance(message, str):
                return r[t.JsonPayload].fail(c.Tests.UNEXPECTED_MESSAGE_TYPE)
            return r[t.JsonPayload].ok(f"processed_{message}")

        @override
        def validate_message(self, data: t.JsonPayload) -> p.Result[bool]:
            if data is None:
                return r[bool].fail_op(
                    "validate handler message",
                    c.ERR_MESSAGE_CANNOT_BE_NONE,
                )
            return r[bool].ok(True)

    class ValidationTestHandler(h[t.JsonPayload, t.JsonPayload]):
        """Test handler for validation."""

        def __init__(self, *, settings: m.Handler | None = None) -> None:
            super().__init__(settings=settings)

        @override
        def validate_message(self, data: t.JsonPayload) -> p.Result[bool]:
            return (
                r[bool].ok(True)
                if data
                else r[bool].fail(c.Tests.VALIDATION_FAILED_FOR_TEST)
            )

        @override
        def handle(self, message: t.JsonPayload) -> p.Result[t.JsonPayload]:
            return r[t.JsonPayload].ok(f"processed_{message}")

    class FailingTestHandler(h[t.JsonPayload, t.JsonPayload]):
        """Test handler that fails."""

        def __init__(self, *, settings: m.Handler | None = None) -> None:
            super().__init__(settings=settings)

        @override
        def handle(self, message: t.JsonPayload) -> p.Result[t.JsonPayload]:
            if not isinstance(message, str):
                return r[t.JsonPayload].fail(c.Tests.UNEXPECTED_MESSAGE_TYPE)
            return r[t.JsonPayload].fail(f"Handler failed for: {message}")

    class HandlerTypeScenario(m.Value):
        """Scenario for handler types."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)
        name: Annotated[str, m.Field(description="Handler type scenario name")]
        handler_type: Annotated[c.HandlerType, m.Field(description="Type")]
        handler_mode: Annotated[c.HandlerType, m.Field(description="Mode")]

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

    VALIDATION_TYPES: ClassVar[Sequence[t.Pair[str, t.JsonPayload]]] = [
        ("str", "test_message"),
        ("int", 42),
        ("float", math.pi),
        ("bool", True),
        ("dict", {"key": "value", "number": 42}),
    ]
