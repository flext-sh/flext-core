"""Coverage tests for currently supported handlers APIs."""

from __future__ import annotations

from types import ModuleType
from typing import override

import pytest

from tests import c, e, h, m, p, r, t


class TestHandlersFullCoverage:
    class _Handler(h[t.JsonPayload, t.JsonPayload]):
        def __init__(self, *, settings: m.Handler | None = None) -> None:
            super().__init__(settings=settings)

        @override
        def handle(self, message: t.JsonPayload) -> p.Result[t.JsonPayload]:
            if isinstance(message, (str, int, float, bool)):
                return r[t.JsonPayload].ok(message)
            return r[t.JsonPayload].fail(c.Core.Tests.TestErrors.UNSUPPORTED_MESSAGE)

    class _QueryHandler(_Handler):
        def __init__(self, *, settings: m.Handler | None = None) -> None:
            super().__init__(settings=settings)

        @override
        def validate_message(self, data: t.JsonPayload) -> p.Result[bool]:
            _ = data
            return r[bool].ok(True)

    class _EventHandler(_Handler):
        def __init__(self, *, settings: m.Handler | None = None) -> None:
            super().__init__(settings=settings)

        @override
        def validate_message(self, data: t.JsonPayload) -> p.Result[bool]:
            _ = data
            return r[bool].ok(True)

    def test_handler_type_literal_and_invalid(self) -> None:
        assert h._handler_type_to_literal(c.HandlerType.OPERATION) == "operation"
        assert h._handler_type_to_literal(c.HandlerType.SAGA) == "saga"
        original_literals = h._HANDLER_TYPE_LITERALS
        h._HANDLER_TYPE_LITERALS = {}
        with pytest.raises(ValueError, match="Unsupported handler type"):
            h._handler_type_to_literal(c.HandlerType.OPERATION)
        h._HANDLER_TYPE_LITERALS = original_literals

    def test_invalid_handler_mode_init_raises(self) -> None:
        base_config = m.Handler(
            handler_id="h1",
            handler_name="bad",
            handler_type=c.HandlerType.COMMAND,
            handler_mode=c.HandlerType.COMMAND,
            command_timeout=10,
            max_command_retries=1,
            metadata=None,
        )
        invalid_config = base_config.model_copy(update={"handler_mode": "invalid"})
        with pytest.raises(
            e.ValidationError,
            match="Invalid handler mode",
        ):
            self._Handler(settings=invalid_config)

    def test_create_from_callable_branches(self) -> None:
        handler_from_config = h.create_from_callable(
            lambda msg: msg,
            handler_config=m.Handler(
                handler_id="cfg",
                handler_name="cfg",
                handler_type=c.HandlerType.COMMAND,
                handler_mode=c.HandlerType.COMMAND,
            ),
        )
        assert handler_from_config.handler_name == "cfg"
        enum_mode_handler = h.create_from_callable(
            lambda msg: msg,
            mode=c.HandlerType.QUERY,
        )
        assert enum_mode_handler.mode == c.HandlerType.QUERY
        str_mode_handler = h.create_from_callable(lambda msg: msg, mode="event")
        assert str_mode_handler.mode == c.HandlerType.EVENT
        invalid_general = h.create_from_callable(lambda msg: msg)
        invalid_general_result = invalid_general.handle("{1, 2, 3}")
        assert invalid_general_result.success
        assert invalid_general_result.value == "{1, 2, 3}"
        tuple_result = invalid_general.handle("('x', 'y')")
        assert tuple_result.success

    def test_dispatch_message_query_and_event_paths(self) -> None:
        qh: TestHandlersFullCoverage._Handler = self._QueryHandler(
            settings=m.Handler(
                handler_id="q2",
                handler_name="q2",
                handler_type=c.HandlerType.QUERY,
                handler_mode=c.HandlerType.QUERY,
            ),
        )
        query_message: t.JsonPayload = "query"
        qr = qh.dispatch_message(
            query_message,
            c.HandlerType.QUERY,
        )
        assert qr.success
        eh: TestHandlersFullCoverage._Handler = self._EventHandler(
            settings=m.Handler(
                handler_id="e2",
                handler_name="e2",
                handler_type=c.HandlerType.EVENT,
                handler_mode=c.HandlerType.EVENT,
            ),
        )
        event_message: t.JsonPayload = "event"
        er = eh.dispatch_message(
            event_message,
            c.HandlerType.EVENT.value,
        )
        assert er.success

    def test_discovery_narrowed_function_paths(self) -> None:
        decorator = h.handler(str)

        @decorator
        def exposed(value: m.BaseModel) -> m.ConfigMap:
            _ = value
            return m.ConfigMap(root={"value": 123})

        module = ModuleType("handlers_discovery")
        setattr(module, "exposed", exposed)
        discovered = h.Discovery.scan_module(module)
        assert len(discovered) == 1
        wrapped = discovered[0][1]
        wrapped_result = wrapped(m.ConfigMap(root={"value": "x"}))
        assert isinstance(wrapped_result, str)
        assert wrapped_result == "root={'value': 123}"
