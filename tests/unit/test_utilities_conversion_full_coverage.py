# mypy: ignore-errors
"""Tests for FlextUtilitiesConversion to achieve full coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections import UserList
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import cast

from pydantic import BaseModel

from flext_core import c, m, r, t, u
from flext_core._utilities.conversion import StrictJsonValue


class _SeqLike(UserList[int]):
    pass


class _Model(BaseModel):
    value: int


def test_conversion_string_and_join_paths() -> None:
    assert c.Errors.UNKNOWN_ERROR
    assert isinstance(m.Categories(), m.Categories)
    assert r[int].ok(1).is_success
    assert isinstance(m.ConfigMap.model_validate({"a": 1}), m.ConfigMap)

    assert u.Conversion.to_str(cast("StrictJsonValue", object())).startswith("<")
    assert u.Conversion.to_str_list(None) == []
    assert u.Conversion.to_str_list(cast("StrictJsonValue", _SeqLike([1, 2]))) == [
        "1",
        "2",
    ]
    assert u.Conversion.normalize("Ab") == "Ab"
    assert u.Conversion.join([]) == ""
    assert u.Conversion.join(["A", "B"], case="lower") == "a b"


def test_to_flexible_value_and_safe_list_branches() -> None:
    none_result = u.Conversion.to_flexible_value(None)
    assert none_result.is_failure

    model_result = u.Conversion.to_flexible_value(
        cast("StrictJsonValue", cast("object", _Model(value=1))),
    )
    assert model_result.is_failure

    mapping_value: Mapping[str, t.ContainerValue] = {"x": 1}
    mapping_result = u.Conversion.to_flexible_value(
        cast("StrictJsonValue", mapping_value),
    )
    assert mapping_result.is_failure

    datetime_result = u.Conversion.to_flexible_value(
        cast("StrictJsonValue", cast("object", datetime.now(UTC))),
    )
    assert datetime_result.is_success

    assert u.Conversion.to_str_list_safe(None) == []
    assert u.Conversion.to_str_list_safe("abc") == ["abc"]
    assert u.Conversion.to_str_list_safe(7) == ["7"]


def test_to_flexible_value_fallback_none_branch_for_unsupported_type() -> None:
    result = u.Conversion.to_flexible_value(cast("StrictJsonValue", (lambda: None)))
    assert result.is_success
    assert isinstance(result.value, str)
