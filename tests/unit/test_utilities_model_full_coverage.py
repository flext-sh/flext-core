"""Tests for FlextUtilitiesModel to achieve full coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import cast

from pydantic import BaseModel

from flext_core import r
from tests import TestUnitModels, c, m, t, u


def test_merge_defaults_and_dump_paths() -> None:
    assert c.UNKNOWN_ERROR
    assert isinstance(m.Categories(categories={}), m.Categories)
    assert r[int].ok(1).is_success
    merged = u.merge_defaults(TestUnitModels._Cfg, {"x": 1, "y": "a"}, {"x": 2})
    assert merged.is_success
    assert merged.value.x == 2
    dumped = u.dump(TestUnitModels._Cfg(x=3, y="z"), exclude_none=True)
    assert dumped["x"] == 3


def test_update_exception_path() -> None:
    result = u.update(
        cast("TestUnitModels._BadCopyModel", cast("BaseModel", "normalized")),
        x=5,
    )
    assert result.is_failure


def test_update_success_path_returns_ok_result() -> None:
    result = u.update(TestUnitModels._Cfg(x=1, y="a"), x=9)
    assert result.is_success
    assert result.value.x == 9


def test_normalize_to_pydantic_dict_and_value_branches() -> None:
    assert u.normalize_to_pydantic_dict(None) == {}
    data = t.ConfigMap(
        root=cast(
            "MutableMapping[str, t.NormalizedValue | BaseModel]",
            {"a": 1, "b": TestUnitModels._Cfg(x=1), "c": [1, TestUnitModels._Cfg(x=2)]},
        )
    )
    normalized = u.normalize_to_pydantic_dict(data)
    assert normalized["a"] == 1
    assert isinstance(normalized["b"], str)
    assert u._normalize_to_pydantic_value(None) == ""
    assert u._normalize_to_pydantic_value(True) is True
    assert u._normalize_to_pydantic_value(1) == 1
    assert u._normalize_to_pydantic_value("x") == "x"
    list_value = u._normalize_to_pydantic_value(
        cast(
            "t.NormalizedValue",
            [1, TestUnitModels._Cfg(x=3), None],
        )
    )
    assert isinstance(list_value, list)
    assert list_value[0] == 1
    assert isinstance(list_value[1], str)
    assert list_value[2] == ""
    assert isinstance(u._normalize_to_pydantic_value(TestUnitModels._Cfg(x=1)), str)
