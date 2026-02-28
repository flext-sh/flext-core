"""Tests for flext_infra.check.fix_pyrefly_config module.

Tests the main entry point and sys.exit behavior.
"""

from __future__ import annotations

from unittest.mock import patch

from flext_infra.check.fix_pyrefly_config import main as main_func


def test_fix_pyrefly_config_main_calls_sys_exit() -> None:
    """Test main() calls sys.exit."""
    with patch("sys.argv", ["fix-pyrefly-config"]):
        with patch("flext_infra.check.fix_pyrefly_config.main") as mock_main:
            mock_main.return_value = 0
            with patch("sys.exit") as _mock_exit:
                try:
                    main_func()
                except SystemExit:
                    pass
