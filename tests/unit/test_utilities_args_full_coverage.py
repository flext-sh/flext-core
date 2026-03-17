"""Tests for Args utilities full coverage."""

from __future__ import annotations

from typing import Annotated, cast

import pytest

import flext_core._utilities.args as args_module
from flext_core import r
from tests import c, m, t, u


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
    assert isinstance(t.ConfigMap({"k": 1}), t.ConfigMap)
    assert u.to_str(1) == "1"
    annotated = u.get_enum_params(_annotated_func)
    assert "mode" in annotated
    failed = u.get_enum_params(_bad_hints_func)
    assert failed == {}


def test_args_get_enum_params_annotated_unwrap_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_candidates: list[type[object] | str] = []

    def _mock_get_type_hints(
        _func: t.TypeHintSpecifier,
        include_extras: bool = False,
    ) -> dict[str, t.TypeHintSpecifier]:
        _ = include_extras
        return {
            "mode": cast("t.TypeHintSpecifier", Annotated[c.Cqrs.HandlerType, "meta"])
        }

    def _mock_validate_enum_type(
        candidate: type[object] | str,
    ) -> r[type[c.Cqrs.HandlerType]]:
        seen_candidates.append(candidate)
        if candidate is c.Cqrs.HandlerType:
            return r[type[c.Cqrs.HandlerType]].ok(c.Cqrs.HandlerType)
        return r[type[c.Cqrs.HandlerType]].fail("invalid")

    monkeypatch.setattr(
        args_module,
        "get_type_hints",
        _mock_get_type_hints,
    )
    monkeypatch.setattr(
        args_module.FlextUtilitiesArgs,
        "_validate_enum_type",
        _mock_validate_enum_type,
    )
    params = u.get_enum_params(_no_op_func)
    assert seen_candidates == [c.Cqrs.HandlerType]
    assert params.get("mode") is c.Cqrs.HandlerType
