"""Tests for FlextConstants to achieve full coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_tests import c, m, u

from flext_core import r


def test_constants_auto_enum_and_bimapping_paths() -> None:
    assert c.Errors.UNKNOWN_ERROR
    assert isinstance(m.Categories(), m.Categories)
    assert r[int].ok(1).is_success
    assert isinstance(m.ConfigMap({"k": 1}), m.ConfigMap)
    assert u.to_str(1) == "1"
