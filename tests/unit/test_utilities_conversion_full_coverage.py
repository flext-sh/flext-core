"""Tests for FlextUtilitiesConversion to achieve full coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections import UserList

from flext_core import c, m, r, u


class _SeqLike(UserList[int]):
    pass


def test_conversion_string_and_join_paths() -> None:
    assert c.Errors.UNKNOWN_ERROR
    assert isinstance(m.Categories(), m.Categories)
    assert r[int].ok(1).is_success
    assert isinstance(m.ConfigMap.model_validate({"a": 1}), m.ConfigMap)
    assert u.Conversion.to_str_list(None) == []
    assert u.Conversion.normalize("Ab") == "Ab"
    assert u.Conversion.join([]) == ""
    assert u.Conversion.join(["A", "B"], case="lower") == "a b"
