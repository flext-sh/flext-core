"""Coverage tests for currently supported handlers APIs."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import cast, override

import pytest
from pydantic import BaseModel

from flext_core import FlextExceptions, FlextHandlers, h, r
from tests import c, m, t

handlers_module = importlib.import_module("flext_core.handlers")


class _Handler(FlextHandlers[t.Scalar, t.Container]):
    def __init__(self, *, config: m.Handler | None = None) -> None:
        super().__init__(config=config)

    @override
    def handle(self, message: t.Scalar) -> r[t.Container]:
        if isinstance(message, (str, int, float, bool)):
            return r[t.Container].ok(message)
        return r[t.Container].fail("unsupported message")


class _QueryHandler(_Handler):
    def __init__(self, *, config: m.Handler | None = None) -> None:
        super().__init__(config=config)

    @override
    def validate_input(self, value: t.Scalar) -> r[bool]:
        _ = value
        return r[bool].ok(True)


class _EventHandler(_Handler):
    def __init__(self, *, config: m.Handler | None = None) -> None:
        super().__init__(config=config)

    @override
    def validate_input(self, value: t.Scalar) -> r[bool]:
        _ = value
        return r[bool].ok(True)


def test_handler_type_literal_and_invalid() -> None:
    assert (
        FlextHandlers._handler_type_to_literal(c.Cqrs.HandlerType.OPERATION)
        == "operation"
    )
    assert FlextHandlers._handler_type_to_literal(c.Cqrs.HandlerType.SAGA) == "saga"
    with pytest.raises(ValueError, match="Unsupported handler type"):
        FlextHandlers._handler_type_to_literal(cast("c.Cqrs.HandlerType", "bad"))


def test_invalid_handler_mode_init_raises() -> None:
    invalid_config = m.Handler.model_construct(
        handler_id="h1",
        handler_name="bad",
        handler_type=c.Cqrs.HandlerType.COMMAND,
        handler_mode=cast("c.Cqrs.HandlerType", "invalid"),
        command_timeout=10,
        max_command_retries=1,
        metadata=None,
    )
    with pytest.raises(FlextExceptions.ValidationError, match="Invalid handler mode"):
        _Handler(config=invalid_config)


def test_create_from_callable_branches() -> None:
    handler_from_config = h.create_from_callable(
        lambda msg: msg,
        handler_config=m.Handler(
            handler_id="cfg",
            handler_name="cfg",
            handler_type=c.Cqrs.HandlerType.COMMAND,
            handler_mode=c.Cqrs.HandlerType.COMMAND,
        ),
    )
    assert handler_from_config.handler_name == "cfg"
    enum_mode_handler = h.create_from_callable(
        lambda msg: msg,
        mode=c.Cqrs.HandlerType.QUERY,
    )
    assert enum_mode_handler.mode == c.Cqrs.HandlerType.QUERY
    str_mode_handler = h.create_from_callable(lambda msg: msg, mode="event")
    assert str_mode_handler.mode == c.Cqrs.HandlerType.EVENT
    invalid_general = h.create_from_callable(lambda msg: msg)
    invalid_general_result = invalid_general.handle(cast("t.Scalar", "{1, 2, 3}"))
    assert invalid_general_result.is_success
    assert invalid_general_result.value == "{1, 2, 3}"
    tuple_result = invalid_general.handle(cast("t.Scalar", "('x', 'y')"))
    assert tuple_result.is_success


def test_run_pipeline_query_and_event_paths() -> None:
    qh = _QueryHandler(
        config=m.Handler(
            handler_id="q2",
            handler_name="q2",
            handler_type=c.Cqrs.HandlerType.QUERY,
            handler_mode=c.Cqrs.HandlerType.QUERY,
        ),
    )
    qr = qh._run_pipeline("query", operation=c.Dispatcher.HANDLER_MODE_QUERY)
    assert qr.is_success
    eh = _EventHandler(
        config=m.Handler(
            handler_id="e2",
            handler_name="e2",
            handler_type=c.Cqrs.HandlerType.EVENT,
            handler_mode=c.Cqrs.HandlerType.EVENT,
        ),
    )
    er = eh._run_pipeline("event", operation=c.Cqrs.HandlerType.EVENT.value)
    assert er.is_success


def test_discovery_narrowed_function_paths() -> None:
    decorator = h.handler(str)

    @decorator
    def exposed(value: BaseModel) -> BaseModel:
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
