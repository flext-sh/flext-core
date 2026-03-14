"""Tests for Args utilities full coverage."""

from __future__ import annotations

from typing import Annotated

import pytest

import flext_core._utilities.args as args_module
from flext_core import c, m, r, u
from flext_tests import t


def _annotated_func(mode: Annotated[c.Cqrs.HandlerType, "meta"]) -> None:
    _ = mode


class UnknownHint:
    """Test class for unknown hints."""


def _bad_hints_func(mode: UnknownHint) -> None:
    _ = mode


def _no_op_func(mode: c.Cqrs.HandlerType) -> None:
    _ = mode


def test_args_get_enum_params_branches() -> None:
    assert c.Errors.UNKNOWN_ERROR
    assert isinstance(m.Categories(), m.Categories)
    assert r[int].ok(1).is_success
    assert isinstance(m.ConfigMap({"k": 1}), m.ConfigMap)
    assert u.to_str(1) == "1"
    annotated = u.get_enum_params(_annotated_func)
    assert "mode" in annotated
    failed = u.get_enum_params(_bad_hints_func)
    assert failed == {}


def test_args_get_enum_params_annotated_unwrap_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _mock_get_type_hints(
        _func: t.TypeHintSpecifier,
    ) -> dict[str, t.TypeHintSpecifier]:
        return {"mode": Annotated[c.Cqrs.HandlerType, "meta"]}

    monkeypatch.setattr(
        args_module,
        "get_type_hints",
        _mock_get_type_hints,
    )
    params = u.get_enum_params(_no_op_func)
    assert params["mode"] is c.Cqrs.HandlerType
