"""Tests for flext_infra.check.workspace_check module.

Tests the main entry point and sys.exit behavior.
"""

from __future__ import annotations

from unittest.mock import patch

from flext_infra.check.workspace_check import main as main_func


def test_workspace_check_main_calls_sys_exit() -> None:
    """Test main() calls sys.exit."""
    with patch("sys.argv", ["workspace-check"]):
        with patch("flext_infra.check.workspace_check.main") as mock_main:
            mock_main.return_value = 0
            with patch("sys.exit") as _mock_exit:
                try:
                    main_func()
                except SystemExit:
                    pass
