"""Tests for Handler models full coverage."""

from __future__ import annotations

import pytest
from flext_core import c, m, r, t, u

handler_models = __import__(
    "flext_core._models.handler", fromlist=["FlextModelsHandler"]
)
FlextModelsHandler = handler_models.FlextModelsHandler


def test_models_handler_branches() -> None:
    assert c.Errors.UNKNOWN_ERROR
    assert isinstance(m.Categories(), m.Categories)
    assert r[int].ok(1).is_success
    assert isinstance(t.ConfigMap.model_validate({"k": 1}), t.ConfigMap)
    assert u.Conversion.to_str(1) == "1"

    req = m.Handler.RegistrationRequest(
        handler=lambda value: value,
        handler_mode="command",
    )
    assert req.handler_mode == "command"

    with pytest.raises(TypeError, match="Handler must be callable"):
        FlextModelsHandler.Registration(name="bad", handler=1)

    ctx = m.Handler.ExecutionContext.create_for_handler("h1", "command")
    assert ctx.execution_time_ms == pytest.approx(0.0)
    state = ctx.metrics_state
    assert isinstance(state, t.Dict)
    ctx.set_metrics_state(t.Dict(root={"x": 1}))
    assert ctx.has_metrics is True


def test_models_handler_uncovered_mode_and_reset_paths() -> None:
    # validate_mode was a no-op and was removed; Literal type handles validation

    ctx = m.Handler.ExecutionContext.create_for_handler("h2", "query")
    assert ctx.is_running is False
    ctx.start_execution()
    assert ctx.is_running
    ctx.set_metrics_state(t.Dict(root={"count": 1}))
    ctx.reset()
    assert ctx.is_running is False
    assert ctx.has_metrics is False
