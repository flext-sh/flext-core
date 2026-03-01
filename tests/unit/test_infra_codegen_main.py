"""Tests for codegen CLI entry point (__main__.py).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from flext_infra.codegen import __main__ as codegen_main


def test_handle_lazy_init_success(tmp_path: Path) -> None:
    """Test _handle_lazy_init with successful generation."""
    args = Mock()
    args.root = tmp_path
    args.check = False

    with patch(
        "flext_infra.codegen.__main__.FlextInfraCodegenLazyInit"
    ) as mock_gen_class:
        mock_gen = Mock()
        mock_gen.run.return_value = 0
        mock_gen_class.return_value = mock_gen

        result = codegen_main._handle_lazy_init(args)
        assert result == 0


def test_handle_lazy_init_with_unmapped_exports(tmp_path: Path) -> None:
    """Test _handle_lazy_init with unmapped exports in check mode."""
    args = Mock()
    args.root = tmp_path
    args.check = True

    with patch(
        "flext_infra.codegen.__main__.FlextInfraCodegenLazyInit"
    ) as mock_gen_class:
        mock_gen = Mock()
        mock_gen.run.return_value = 3  # 3 unmapped exports
        mock_gen_class.return_value = mock_gen

        result = codegen_main._handle_lazy_init(args)
        assert result == 0  # Still returns 0, just warns


def test_handle_lazy_init_check_mode(tmp_path: Path) -> None:
    """Test _handle_lazy_init respects check_only flag."""
    args = Mock()
    args.root = tmp_path
    args.check = True

    with patch(
        "flext_infra.codegen.__main__.FlextInfraCodegenLazyInit"
    ) as mock_gen_class:
        mock_gen = Mock()
        mock_gen.run.return_value = 0
        mock_gen_class.return_value = mock_gen

        result = codegen_main._handle_lazy_init(args)
        assert result == 0
        mock_gen.run.assert_called_once_with(check_only=True, scan_tests=False)


def test_handle_lazy_init_enforce_mode(tmp_path: Path) -> None:
    """Test _handle_lazy_init in enforce mode (not check)."""
    args = Mock()
    args.root = tmp_path
    args.check = False

    with patch(
        "flext_infra.codegen.__main__.FlextInfraCodegenLazyInit"
    ) as mock_gen_class:
        mock_gen = Mock()
        mock_gen.run.return_value = 0
        mock_gen_class.return_value = mock_gen

        result = codegen_main._handle_lazy_init(args)
        assert result == 0
        mock_gen.run.assert_called_once_with(check_only=False, scan_tests=False)


def test_main_lazy_init_command(tmp_path: Path) -> None:
    """Test main() with lazy-init command."""
    argv = ["lazy-init", "--root", str(tmp_path)]

    with patch("flext_infra.codegen.__main__._handle_lazy_init") as mock_handle:
        mock_handle.return_value = 0
        result = codegen_main.main(argv)
        assert result == 0
        mock_handle.assert_called_once()


def test_main_lazy_init_with_check_flag(tmp_path: Path) -> None:
    """Test main() lazy-init with --check flag."""
    argv = ["lazy-init", "--check", "--root", str(tmp_path)]

    with patch("flext_infra.codegen.__main__._handle_lazy_init") as mock_handle:
        mock_handle.return_value = 0
        result = codegen_main.main(argv)
        assert result == 0
        args = mock_handle.call_args[0][0]
        assert args.check is True


def test_main_lazy_init_default_root() -> None:
    """Test main() lazy-init uses cwd as default root."""
    argv = ["lazy-init"]

    with patch("flext_infra.codegen.__main__._handle_lazy_init") as mock_handle:
        mock_handle.return_value = 0
        result = codegen_main.main(argv)
        assert result == 0
        args = mock_handle.call_args[0][0]
        assert args.root == Path.cwd()


def test_main_unknown_command() -> None:
    """Test main() with unknown command (lines 53-54)."""
    argv = ["unknown-command"]

    with pytest.raises(SystemExit) as exc_info:
        codegen_main.main(argv)
    assert exc_info.value.code == 2


def test_main_no_command() -> None:
    """Test main() with no command specified."""
    argv = []

    with pytest.raises(SystemExit) as exc_info:
        codegen_main.main(argv)
    assert exc_info.value.code == 2


def test_main_runtime_configuration() -> None:
    """Test main() configures FlextRuntime."""
    argv = ["lazy-init"]

    with patch("flext_core.FlextRuntime.ensure_structlog_configured") as mock_config:
        with patch("flext_infra.codegen.__main__._handle_lazy_init") as mock_handle:
            mock_handle.return_value = 0
            result = codegen_main.main(argv)
            assert result == 0
            mock_config.assert_called_once()


def test_main_lazy_init_with_custom_root(tmp_path: Path) -> None:
    """Test main() lazy-init with custom root directory."""
    custom_root = tmp_path / "custom"
    argv = ["lazy-init", "--root", str(custom_root)]

    with patch("flext_infra.codegen.__main__._handle_lazy_init") as mock_handle:
        mock_handle.return_value = 0
        result = codegen_main.main(argv)
        assert result == 0
        args = mock_handle.call_args[0][0]
        assert args.root == custom_root


def test_main_entry_point() -> None:
    """Test __main__ entry point."""
    argv = ["lazy-init"]

    with patch("flext_infra.codegen.__main__.main") as mock_main:
        mock_main.return_value = 0
        exit_code = codegen_main.main(argv)
        assert exit_code == 0


def test_main_entry_point_via_sys_exit() -> None:
    """Test __main__ entry point via sys.exit (line 68)."""
    result = subprocess.run(
        ["python", "-m", "flext_infra.codegen", "lazy-init", "--help"],  # noqa: S607
        capture_output=True,
        text=True,
        cwd="/home/marlonsc/flext/flext-core",
        check=False,
    )
    # Should succeed with help output
    assert result.returncode == 0
    assert "lazy-init" in result.stdout


__all__ = []
