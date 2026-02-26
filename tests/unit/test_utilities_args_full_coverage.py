from __future__ import annotations

from typing import Annotated, cast

import pytest
from flext_core import c, m, p, r, t, u


def _annotated_func(mode: Annotated[c.Cqrs.HandlerType, "meta"]) -> None:
    _ = mode


class UnknownHint:
    pass


def _bad_hints_func(mode: UnknownHint) -> None:
    _ = mode


def _no_op_func(mode: c.Cqrs.HandlerType) -> None:
    _ = mode


def test_args_get_enum_params_branches() -> None:
    assert c.Errors.UNKNOWN_ERROR
    assert isinstance(m.Categories(), m.Categories)
    assert r[int].ok(1).is_success
    assert isinstance(t.ConfigMap.model_validate({"k": 1}), t.ConfigMap)
    assert u.Conversion.to_str(1) == "1"

    annotated = u.Args.get_enum_params(cast("p.CallableWithHints", _annotated_func))
    assert "mode" in annotated

    failed = u.Args.get_enum_params(cast("p.CallableWithHints", _bad_hints_func))
    assert failed == {}


def test_args_get_enum_params_annotated_unwrap_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import flext_core._utilities.args as args_module

    monkeypatch.setattr(
        args_module,
        "get_type_hints",
        lambda _func: {"mode": Annotated[c.Cqrs.HandlerType, "meta"]},
    )
    params = u.Args.get_enum_params(
        cast("p.CallableWithHints", _no_op_func),
    )
    assert params["mode"] is c.Cqrs.HandlerType
