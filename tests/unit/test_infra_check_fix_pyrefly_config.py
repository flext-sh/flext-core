"""Tests for flext_infra.check.fix_pyrefly_config module.

Tests the real CLI entry point.
"""

from __future__ import annotations

import subprocess
import sys


def test_fix_pyrefly_config_main_executes_real_cli_help() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "flext_infra.check", "fix-pyrefly-config", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "usage:" in completed.stdout
