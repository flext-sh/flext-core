"""Tests for flext_infra.check.__main__ CLI entry point."""

from __future__ import annotations

import subprocess
import sys


def test_check_main_executes_real_cli() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "flext_infra.check", "run", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "usage:" in completed.stdout
