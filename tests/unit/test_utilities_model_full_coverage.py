"""Tests for FlextUtilitiesModel to achieve full coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import cast

from flext_core import c, m, r, t, u
from pydantic import BaseModel


class _Cfg(BaseModel):
    x: int = 0
    y: str = "a"


class _BadCopyModel(BaseModel):
    x: int = 1


def test_merge_defaults_and_dump_paths() -> None:
    assert c.Errors.UNKNOWN_ERROR
    assert isinstance(m.Categories(), m.Categories)
    assert r[int].ok(1).is_success

    merged = u.Model.merge_defaults(_Cfg, {"x": 1, "y": "a"}, {"x": 2})
    assert merged.is_success
    assert merged.value.x == 2

    dumped = u.Model.dump(_Cfg(x=3, y="z"), exclude_none=True)
    assert dumped["x"] == 3


def test_update_exception_path() -> None:
    result = u.Model.update(cast("_BadCopyModel", cast("BaseModel", object())), x=5)
    assert result.is_failure


def test_update_success_path_returns_ok_result() -> None:
    result = u.Model.update(_Cfg(x=1, y="a"), x=9)
    assert result.is_success
    assert result.value.x == 9


def test_normalize_to_pydantic_dict_and_value_branches() -> None:
    assert u.Model.normalize_to_pydantic_dict(None) == {}

    data = t.ConfigMap(root={"a": 1, "b": _Cfg(x=1), "c": [1, _Cfg(x=2)]})
    normalized = u.Model.normalize_to_pydantic_dict(data)
    assert normalized["a"] == 1
    assert isinstance(normalized["b"], str)

    assert u.Model._normalize_to_pydantic_value(None) is None
    assert u.Model._normalize_to_pydantic_value(True) is True
    assert u.Model._normalize_to_pydantic_value(1) == 1
    assert u.Model._normalize_to_pydantic_value("x") == "x"
    list_value = u.Model._normalize_to_pydantic_value([1, _Cfg(x=3), None])
    assert isinstance(list_value, list)
    assert list_value[0] == 1
    assert isinstance(list_value[1], str)
    assert list_value[2] is None
    assert isinstance(u.Model._normalize_to_pydantic_value(_Cfg(x=1)), str)
