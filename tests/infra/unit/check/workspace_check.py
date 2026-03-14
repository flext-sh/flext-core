"""Tests for flext_infra.check.workspace_check module.

Tests the real entry-point behavior.
"""

from __future__ import annotations

from flext_infra.check.workspace_check import main as main_func


def test_workspace_check_main_returns_error_without_projects() -> None:
    exit_code = main_func([])
    assert exit_code == 1
