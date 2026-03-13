"""Tests for FlextSettings to achieve full coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pydantic_settings import BaseSettings

from flext_core import FlextSettings, c, m, r, u


class _SubSettings(FlextSettings):
    pass


def test_settings_materialize_and_context_overrides() -> None:
    assert c.Errors.UNKNOWN_ERROR
    assert isinstance(m.Categories(), m.Categories)
    assert r[int].ok(1).is_success
    assert isinstance(m.ConfigMap({"k": 1}), m.ConfigMap)
    assert u.to_str(1) == "1"
    sub = _SubSettings.get_global()
    assert isinstance(sub, _SubSettings)
    FlextSettings.register_context_overrides("ctx-a", app_name="A")
    cfg = FlextSettings.for_context("ctx-a")
    assert cfg.app_name == "A"
    unchanged = FlextSettings.for_context("ctx-b")
    assert isinstance(unchanged, FlextSettings)

    class _N(BaseSettings):
        x: int = 1

    FlextSettings.register_namespace("n1", _N)
    assert FlextSettings.get_namespace_config("n1") is _N
