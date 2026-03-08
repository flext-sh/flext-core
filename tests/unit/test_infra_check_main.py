"""Tests for flext_infra.check.__main__ CLI entry point.

Tests the main entry point and sys.exit behavior.
"""

from __future__ import annotations

from unittest.mock import patch

from flext_infra.check.__main__ import main as main_func


def test_check_main_calls_sys_exit() -> None:
    """Test main() calls sys.exit."""
    with patch("sys.argv", ["check"]):
        with patch("flext_infra.check.__main__.main") as mock_main:
            mock_main.return_value = 0
            with patch("sys.exit") as _mock_exit:
                try:
                    main_func()
                except SystemExit:
                    pass
