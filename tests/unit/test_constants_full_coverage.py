"""Tests for FlextConstants to achieve full coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import c, m, r, t, u


def test_constants_auto_enum_and_bimapping_paths() -> None:
    assert c.Errors.UNKNOWN_ERROR
    assert isinstance(m.Categories(), m.Categories)
    assert r[int].ok(1).is_success
    assert isinstance(t.ConfigMap.model_validate({"k": 1}), t.ConfigMap)
    assert u.Conversion.to_str(1) == "1"
