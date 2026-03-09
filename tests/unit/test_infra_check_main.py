"""Tests for flext_infra.check.__main__ CLI entry point."""

from __future__ import annotations

from flext_infra.check.__main__ import main as main_func


def test_check_main_executes_real_cli() -> None:
    exit_code = main_func()
    assert isinstance(exit_code, int)
    assert exit_code in {0, 1, 2, 3}
