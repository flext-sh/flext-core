"""Coverage tests for currently supported handlers APIs."""

from __future__ import annotations

from types import ModuleType
from typing import override

import pytest
from pydantic import BaseModel

from flext_core import e, h, r
from tests import c, m, t


class TestHandlersFullCoverage:
    class _Handler(h[t.ValueOrModel, t.ValueOrModel]):
        def __init__(self, *, config: m.Handler | None = None) -> None:
            super().__init__(config=config)

        @override
        def handle(self, message: t.ValueOrModel) -> r[t.ValueOrModel]:
            if isinstance(message, (str, int, float, bool)):
                return r[t.ValueOrModel].ok(message)
            return r[t.ValueOrModel].fail(c.Core.Tests.TestErrors.UNSUPPORTED_MESSAGE)

    class _QueryHandler(_Handler):
        def __init__(self, *, config: m.Handler | None = None) -> None:
            super().__init__(config=config)

        @override
        def validate_message(self, data: t.ValueOrModel) -> r[bool]:
            _ = data
            return r[bool].ok(True)

    class _EventHandler(_Handler):
        def __init__(self, *, config: m.Handler | None = None) -> None:
            super().__init__(config=config)

        @override
        def validate_message(self, data: t.ValueOrModel) -> r[bool]:
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
        invalid_config = m.Handler.model_construct(
            handler_id="h1",
            handler_name="bad",
            handler_type=c.HandlerType.COMMAND,
            # type: ignore[arg-type]  # Intentionally invalid to cover FlextHandlers.__init__ guard.
            handler_mode="invalid",
            command_timeout=10,
            max_command_retries=1,
            metadata=None,
        )
        with pytest.raises(
            e.ValidationError,
            match="Invalid handler mode",
        ):
            self._Handler(config=invalid_config)

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

    def test_run_pipeline_query_and_event_paths(self) -> None:
        qh: TestHandlersFullCoverage._Handler = self._QueryHandler(
            config=m.Handler(
                handler_id="q2",
                handler_name="q2",
                handler_type=c.HandlerType.QUERY,
                handler_mode=c.HandlerType.QUERY,
            ),
        )
        query_message: t.ValueOrModel = "query"
        qr = qh._run_pipeline(
            query_message,
            c.HandlerType.QUERY,
        )
        assert qr.success
        eh: TestHandlersFullCoverage._Handler = self._EventHandler(
            config=m.Handler(
                handler_id="e2",
                handler_name="e2",
                handler_type=c.HandlerType.EVENT,
                handler_mode=c.HandlerType.EVENT,
            ),
        )
        event_message: t.ValueOrModel = "event"
        er = eh._run_pipeline(
            event_message,
            c.HandlerType.EVENT.value,
        )
        assert er.success

    def test_discovery_narrowed_function_paths(self) -> None:
        decorator = h.handler(str)

        @decorator
        def exposed(value: BaseModel) -> BaseModel:
            _ = value
            return t.ConfigMap(root={"value": 123})

        module = ModuleType("handlers_discovery")
        setattr(module, "exposed", exposed)
        discovered = h.Discovery.scan_module(module)
        assert len(discovered) == 1
        wrapped = discovered[0][1]
        wrapped_result = wrapped(t.ConfigMap(root={"value": "x"}))
        assert isinstance(wrapped_result, str)
        assert wrapped_result == "root={'value': 123}"
