"""Tests for Handler models full coverage."""

from __future__ import annotations

import pytest

from flext_core import r
from tests import c, m, t, u


def test_models_handler_branches() -> None:
    assert c.Errors.UNKNOWN_ERROR
    assert isinstance(m.Categories(), m.Categories)
    assert r[int].ok(1).is_success
    assert isinstance(t.ConfigMap({"k": 1}), t.ConfigMap)
    assert u.to_str(1) == "1"
    req = m.RegistrationRequest(
        handler=lambda value: value, handler_mode=c.Cqrs.HandlerType.COMMAND
    )
    assert req.handler_mode == c.Cqrs.HandlerType.COMMAND
    with pytest.raises(Exception, match="Handler must be callable"):
        m.Registration.model_validate({"name": "bad", "handler": 1})
    ctx = m.ExecutionContext.create_for_handler("h1", c.Cqrs.HandlerType.COMMAND)
    raw_execution_time = ctx.execution_time_ms
    execution_time_ms = (
        raw_execution_time() if callable(raw_execution_time) else raw_execution_time
    )
    assert abs(execution_time_ms - 0.0) < 1e-9
    state = ctx.metrics_state
    assert isinstance(state, t.Dict)
    ctx.set_metrics_state(t.Dict(root={"x": 1}))
    assert ctx.has_metrics is True


def test_models_handler_uncovered_mode_and_reset_paths() -> None:
    ctx = m.ExecutionContext.create_for_handler("h2", c.Cqrs.HandlerType.QUERY)
    assert ctx.is_running is False
    ctx.start_execution()
    assert ctx.is_running
    ctx.set_metrics_state(t.Dict(root={"count": 1}))
    ctx.reset()
    assert ctx.is_running is False
    assert ctx.has_metrics is False
