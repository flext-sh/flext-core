from __future__ import annotations

from typing import Annotated

from flext_core import c, m, r, t, u


def _annotated_func(mode: Annotated[c.Cqrs.HandlerType, "meta"]) -> None:
    _ = mode


def _bad_hints_func(mode: "UnknownHint") -> None:
    _ = mode


def test_args_get_enum_params_branches() -> None:
    assert c.Errors.UNKNOWN_ERROR
    assert isinstance(m.Categories(), m.Categories)
    assert r[int].ok(1).is_success
    assert isinstance(t.ConfigMap.model_validate({"k": 1}), t.ConfigMap)
    assert u.Conversion.to_str(1) == "1"

    annotated = u.Args.get_enum_params(_annotated_func)
    assert "mode" in annotated

    failed = u.Args.get_enum_params(_bad_hints_func)
    assert failed == {}


def test_args_get_enum_params_annotated_unwrap_branch(monkeypatch) -> None:
    import flext_core._utilities.args as args_module

    monkeypatch.setattr(
        args_module,
        "get_type_hints",
        lambda _func: {"mode": Annotated[c.Cqrs.HandlerType, "meta"]},
    )
    params = u.Args.get_enum_params(lambda mode: None)
    assert params["mode"] is c.Cqrs.HandlerType
