"""Tests for flext_infra.__main__ CLI entry point.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from unittest.mock import Mock, patch

from flext_infra.__main__ import FlextInfraMainCLI


class TestFlextInfraMainCLI:
    """Test suite for FlextInfraMainCLI."""

    def test_main_prints_help_when_no_args(self) -> None:
        """Test main() prints help when no arguments provided."""
        with patch("sys.argv", ["flext-infra"]):
            with patch("flext_infra.__main__.output") as mock_output:
                result = FlextInfraMainCLI.main()
                assert result == 1
                mock_output.info.assert_called()

    def test_main_prints_help_with_help_flag(self) -> None:
        """Test main() prints help with -h flag."""
        with patch("sys.argv", ["flext-infra", "-h"]):
            with patch("flext_infra.__main__.output") as mock_output:
                result = FlextInfraMainCLI.main()
                assert result == 0
                mock_output.info.assert_called()

    def test_main_prints_help_with_long_help_flag(self) -> None:
        """Test main() prints help with --help flag."""
        with patch("sys.argv", ["flext-infra", "--help"]):
            with patch("flext_infra.__main__.output") as mock_output:
                result = FlextInfraMainCLI.main()
                assert result == 0
                mock_output.info.assert_called()

    def test_main_returns_error_for_unknown_group(self) -> None:
        """Test main() returns error for unknown group."""
        with patch("sys.argv", ["flext-infra", "unknown"]):
            with patch("flext_infra.__main__.output") as mock_output:
                result = FlextInfraMainCLI.main()
                assert result == 1
                mock_output.error.assert_called()

    def test_main_dispatches_to_valid_group(self) -> None:
        """Test main() dispatches to valid group module."""
        with patch("sys.argv", ["flext-infra", "basemk", "generate"]):
            with patch("importlib.import_module") as mock_import:
                mock_module = Mock()
                mock_module.main.return_value = 0
                mock_import.return_value = mock_module
                result = FlextInfraMainCLI.main()
                assert result == 0
                mock_import.assert_called_once_with("flext_infra.basemk.__main__")
                mock_module.main.assert_called_once()

    def test_main_rewrites_argv_for_group(self) -> None:
        """Test main() rewrites sys.argv for group module."""
        original_argv = sys.argv.copy()
        try:
            with patch("sys.argv", ["flext-infra", "check", "lint"]):
                with patch("importlib.import_module") as mock_import:
                    mock_module = Mock()
                    mock_module.main.return_value = 0
                    mock_import.return_value = mock_module
                    FlextInfraMainCLI.main()
                    assert sys.argv[0] == "flext-infra check"
                    assert sys.argv[1] == "lint"
        finally:
            sys.argv = original_argv

    def test_main_handles_group_module_returning_none(self) -> None:
        """Test main() handles group module returning None."""
        with patch("sys.argv", ["flext-infra", "docs", "build"]):
            with patch("importlib.import_module") as mock_import:
                mock_module = Mock()
                mock_module.main.return_value = None
                mock_import.return_value = mock_module
                result = FlextInfraMainCLI.main()
                assert result == 0

    def test_main_handles_group_module_returning_string(self) -> None:
        """Test main() converts string exit code to int."""
        with patch("sys.argv", ["flext-infra", "release", "tag"]):
            with patch("importlib.import_module") as mock_import:
                mock_module = Mock()
                mock_module.main.return_value = "0"
                mock_import.return_value = mock_module
                result = FlextInfraMainCLI.main()
                assert result == 0
                assert isinstance(result, int)

    def test_main_all_groups_defined(self) -> None:
        """Test that all documented groups are defined in GROUPS."""
        expected_groups = {
            "basemk",
            "check",
            "codegen",
            "core",
            "deps",
            "docs",
            "github",
            "maintenance",
            "refactor",
            "release",
            "workspace",
        }
        assert set(FlextInfraMainCLI.GROUPS.keys()) == expected_groups

    def test_main_group_modules_are_valid(self) -> None:
        """Test that all group modules are valid module paths."""
        for group, module_path in FlextInfraMainCLI.GROUPS.items():
            assert isinstance(module_path, str)
            assert module_path.startswith("flext_infra.")
            assert (
                module_path.endswith(".__main__")
                or module_path == "flext_infra.refactor"
            )
            assert group in module_path

    def test_print_help_outputs_all_groups(self) -> None:
        """Test _print_help() outputs all groups."""
        with patch("flext_infra.__main__.output") as mock_output:
            FlextInfraMainCLI._print_help()
            assert mock_output.info.call_count >= 10

    def test_main_ensures_structlog_configured(self) -> None:
        """Test main() ensures structlog is configured."""
        with patch("sys.argv", ["flext-infra", "-h"]):
            with patch(
                "flext_core.FlextRuntime.ensure_structlog_configured"
            ) as mock_ensure:
                with patch("flext_infra.__main__.output"):
                    FlextInfraMainCLI.main()
                    mock_ensure.assert_called_once()

    def test_main_calls_sys_exit(self) -> None:
        """Test FlextInfraMainCLI.main() can be used with sys.exit."""
        with patch("sys.argv", ["flext-infra", "docs", "build"]):
            with patch("importlib.import_module") as mock_import:
                mock_module = Mock()
                mock_module.main.return_value = 0
                mock_import.return_value = mock_module
                result = FlextInfraMainCLI.main()
                assert result == 0
