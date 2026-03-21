"""Tests for FlextUtilitiesConversion to achieve full coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import r
from tests import c, m, t, u


def test_conversion_string_and_join_paths() -> None:
    assert c.UNKNOWN_ERROR
    assert isinstance(m.Categories(), m.Categories)
    assert r[int].ok(1).is_success
    assert isinstance(t.ConfigMap({"a": 1}), t.ConfigMap)
    assert u.to_str_list(None) == []
    assert u.normalize("Ab") == "Ab"
    assert u.join([]) == ""
    assert u.join(["A", "B"], case="lower") == "a b"
