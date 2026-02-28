"""Tests for flext_infra.deps.__main__ with 100% coverage."""

from __future__ import annotations

import sys
from unittest.mock import Mock, patch

# Ensure modules are imported for coverage
import flext_infra.deps.__main__  # noqa: F401
import pytest
from flext_infra.deps.__main__ import _SUBCOMMANDS, main, main as main_func


class TestSubcommandMapping:
    """Test subcommand mapping."""

    def test_subcommands_mapping_exists(self) -> None:
        """Test that subcommands mapping exists."""
        assert _SUBCOMMANDS is not None
        assert len(_SUBCOMMANDS) == 5

    def test_subcommands_mapping_has_detect(self) -> None:
        """Test detect subcommand in mapping."""
        assert "detect" in _SUBCOMMANDS
        assert _SUBCOMMANDS["detect"] == "flext_infra.deps.detector"

    def test_subcommands_mapping_has_extra_paths(self) -> None:
        """Test extra-paths subcommand in mapping."""
        assert "extra-paths" in _SUBCOMMANDS
        assert _SUBCOMMANDS["extra-paths"] == "flext_infra.deps.extra_paths"

    def test_subcommands_mapping_has_internal_sync(self) -> None:
        """Test internal-sync subcommand in mapping."""
        assert "internal-sync" in _SUBCOMMANDS
        assert _SUBCOMMANDS["internal-sync"] == "flext_infra.deps.internal_sync"

    def test_subcommands_mapping_has_modernize(self) -> None:
        """Test modernize subcommand in mapping."""
        assert "modernize" in _SUBCOMMANDS
        assert _SUBCOMMANDS["modernize"] == "flext_infra.deps.modernizer"

    def test_subcommands_mapping_has_path_sync(self) -> None:
        """Test path-sync subcommand in mapping."""
        assert "path-sync" in _SUBCOMMANDS
        assert _SUBCOMMANDS["path-sync"] == "flext_infra.deps.path_sync"


class TestMainHelpAndErrors:
    """Test main function help and error handling."""

    def test_main_with_help_flag(self) -> None:
        """Test main with -h flag."""
        with patch("sys.argv", ["prog", "-h"]):
            with patch("flext_infra.deps.__main__.output.info") as mock_info:
                with patch(
                    "flext_infra.deps.__main__.FlextRuntime.ensure_structlog_configured"
                ):
                    result = main()
                    assert result == 0
                    mock_info.assert_called()

    def test_main_with_help_long_flag(self) -> None:
        """Test main with --help flag."""
        with patch("sys.argv", ["prog", "--help"]):
            with patch("flext_infra.deps.__main__.output.info") as mock_info:
                with patch(
                    "flext_infra.deps.__main__.FlextRuntime.ensure_structlog_configured"
                ):
                    result = main()
                    assert result == 0
                    mock_info.assert_called()

    def test_main_with_no_arguments(self) -> None:
        """Test main with no arguments."""
        with patch("sys.argv", ["prog"]):
            with patch("flext_infra.deps.__main__.output.info") as mock_info:
                with patch(
                    "flext_infra.deps.__main__.FlextRuntime.ensure_structlog_configured"
                ):
                    result = main()
                    assert result == 1
                    mock_info.assert_called()

    def test_main_with_unknown_subcommand(self) -> None:
        """Test main with unknown subcommand."""
        with patch("sys.argv", ["prog", "unknown"]):
            with patch("flext_infra.deps.__main__.output.error") as mock_error:
                with patch(
                    "flext_infra.deps.__main__.FlextRuntime.ensure_structlog_configured"
                ):
                    result = main()
                    assert result == 1
                    mock_error.assert_called()

    def test_main_help_lists_subcommands(self) -> None:
        """Test main help output lists all subcommands."""
        with patch("sys.argv", ["prog", "-h"]):
            with patch("flext_infra.deps.__main__.output.info") as mock_info:
                with patch(
                    "flext_infra.deps.__main__.FlextRuntime.ensure_structlog_configured"
                ):
                    main()
                    # Check that info was called with subcommand names
                    calls = [str(call) for call in mock_info.call_args_list]
                    assert any("detect" in str(call) for call in calls)


class TestMainSubcommandDispatch:
    """Test main function subcommand dispatching."""

    def test_main_with_detect_subcommand(self) -> None:
        """Test main with detect subcommand."""
        mock_module = Mock()
        mock_module.main = Mock(return_value=0)

        with patch("sys.argv", ["prog", "detect"]):
            with patch(
                "flext_infra.deps.__main__.importlib.import_module",
                return_value=mock_module,
            ):
                with patch(
                    "flext_infra.deps.__main__.FlextRuntime.ensure_structlog_configured"
                ):
                    result = main()
                    assert result == 0
                    mock_module.main.assert_called_once()

    def test_main_with_extra_paths_subcommand(self) -> None:
        """Test main with extra-paths subcommand."""
        mock_module = Mock()
        mock_module.main = Mock(return_value=0)

        with patch("sys.argv", ["prog", "extra-paths"]):
            with patch(
                "flext_infra.deps.__main__.importlib.import_module",
                return_value=mock_module,
            ):
                with patch(
                    "flext_infra.deps.__main__.FlextRuntime.ensure_structlog_configured"
                ):
                    result = main()
                    assert result == 0
                    mock_module.main.assert_called_once()

    def test_main_with_internal_sync_subcommand(self) -> None:
        """Test main with internal-sync subcommand."""
        mock_module = Mock()
        mock_module.main = Mock(return_value=0)

        with patch("sys.argv", ["prog", "internal-sync"]):
            with patch(
                "flext_infra.deps.__main__.importlib.import_module",
                return_value=mock_module,
            ):
                with patch(
                    "flext_infra.deps.__main__.FlextRuntime.ensure_structlog_configured"
                ):
                    result = main()
                    assert result == 0
                    mock_module.main.assert_called_once()

    def test_main_with_modernize_subcommand(self) -> None:
        """Test main with modernize subcommand."""
        mock_module = Mock()
        mock_module.main = Mock(return_value=0)

        with patch("sys.argv", ["prog", "modernize"]):
            with patch(
                "flext_infra.deps.__main__.importlib.import_module",
                return_value=mock_module,
            ):
                with patch(
                    "flext_infra.deps.__main__.FlextRuntime.ensure_structlog_configured"
                ):
                    result = main()
                    assert result == 0
                    mock_module.main.assert_called_once()

    def test_main_with_path_sync_subcommand(self) -> None:
        """Test main with path-sync subcommand."""
        mock_module = Mock()
        mock_module.main = Mock(return_value=0)

        with patch("sys.argv", ["prog", "path-sync"]):
            with patch(
                "flext_infra.deps.__main__.importlib.import_module",
                return_value=mock_module,
            ):
                with patch(
                    "flext_infra.deps.__main__.FlextRuntime.ensure_structlog_configured"
                ):
                    result = main()
                    assert result == 0
                    mock_module.main.assert_called_once()


class TestMainReturnValues:
    """Test main function return value handling."""

    def test_main_subcommand_returns_none(self) -> None:
        """Test main when subcommand returns None."""
        mock_module = Mock()
        mock_module.main = Mock(return_value=None)

        with patch("sys.argv", ["prog", "detect"]):
            with patch(
                "flext_infra.deps.__main__.importlib.import_module",
                return_value=mock_module,
            ):
                with patch(
                    "flext_infra.deps.__main__.FlextRuntime.ensure_structlog_configured"
                ):
                    result = main()
                    assert result == 0

    def test_main_subcommand_returns_zero(self) -> None:
        """Test main when subcommand returns 0."""
        mock_module = Mock()
        mock_module.main = Mock(return_value=0)

        with patch("sys.argv", ["prog", "detect"]):
            with patch(
                "flext_infra.deps.__main__.importlib.import_module",
                return_value=mock_module,
            ):
                with patch(
                    "flext_infra.deps.__main__.FlextRuntime.ensure_structlog_configured"
                ):
                    result = main()
                    assert result == 0

    def test_main_subcommand_returns_non_zero(self) -> None:
        """Test main when subcommand returns non-zero."""
        mock_module = Mock()
        mock_module.main = Mock(return_value=42)

        with patch("sys.argv", ["prog", "detect"]):
            with patch(
                "flext_infra.deps.__main__.importlib.import_module",
                return_value=mock_module,
            ):
                with patch(
                    "flext_infra.deps.__main__.FlextRuntime.ensure_structlog_configured"
                ):
                    result = main()
                    assert result == 42

    def test_main_subcommand_returns_string_zero(self) -> None:
        """Test main when subcommand returns string '0'."""
        mock_module = Mock()
        mock_module.main = Mock(return_value="0")

        with patch("sys.argv", ["prog", "detect"]):
            with patch(
                "flext_infra.deps.__main__.importlib.import_module",
                return_value=mock_module,
            ):
                with patch(
                    "flext_infra.deps.__main__.FlextRuntime.ensure_structlog_configured"
                ):
                    result = main()
                    assert result == 0

    def test_main_subcommand_returns_false(self) -> None:
        """Test main when subcommand returns False."""
        mock_module = Mock()
        mock_module.main = Mock(return_value=False)

        with patch("sys.argv", ["prog", "detect"]):
            with patch(
                "flext_infra.deps.__main__.importlib.import_module",
                return_value=mock_module,
            ):
                with patch(
                    "flext_infra.deps.__main__.FlextRuntime.ensure_structlog_configured"
                ):
                    result = main()
                    assert result == 0

    def test_main_subcommand_returns_true(self) -> None:
        """Test main when subcommand returns True."""
        mock_module = Mock()
        mock_module.main = Mock(return_value=True)

        with patch("sys.argv", ["prog", "detect"]):
            with patch(
                "flext_infra.deps.__main__.importlib.import_module",
                return_value=mock_module,
            ):
                with patch(
                    "flext_infra.deps.__main__.FlextRuntime.ensure_structlog_configured"
                ):
                    result = main()
                    assert result == 1


class TestMainModuleImport:
    """Test main function module importing."""

    def test_main_imports_correct_module_for_detect(self) -> None:
        """Test main imports correct module for detect."""
        mock_module = Mock()
        mock_module.main = Mock(return_value=0)

        with patch("sys.argv", ["prog", "detect"]):
            with patch(
                "flext_infra.deps.__main__.importlib.import_module",
                return_value=mock_module,
            ) as mock_import:
                with patch(
                    "flext_infra.deps.__main__.FlextRuntime.ensure_structlog_configured"
                ):
                    main()
                    mock_import.assert_called_with("flext_infra.deps.detector")

    def test_main_imports_correct_module_for_modernize(self) -> None:
        """Test main imports correct module for modernize."""
        mock_module = Mock()
        mock_module.main = Mock(return_value=0)

        with patch("sys.argv", ["prog", "modernize"]):
            with patch(
                "flext_infra.deps.__main__.importlib.import_module",
                return_value=mock_module,
            ) as mock_import:
                with patch(
                    "flext_infra.deps.__main__.FlextRuntime.ensure_structlog_configured"
                ):
                    main()
                    mock_import.assert_called_with("flext_infra.deps.modernizer")

    def test_main_imports_correct_module_for_path_sync(self) -> None:
        """Test main imports correct module for path-sync."""
        mock_module = Mock()
        mock_module.main = Mock(return_value=0)

        with patch("sys.argv", ["prog", "path-sync"]):
            with patch(
                "flext_infra.deps.__main__.importlib.import_module",
                return_value=mock_module,
            ) as mock_import:
                with patch(
                    "flext_infra.deps.__main__.FlextRuntime.ensure_structlog_configured"
                ):
                    main()
                    mock_import.assert_called_with("flext_infra.deps.path_sync")


class TestMainSysArgvModification:
    """Test main function sys.argv modification."""

    def test_main_modifies_sys_argv_for_subcommand(self) -> None:
        """Test main modifies sys.argv for subcommand."""
        mock_module = Mock()
        mock_module.main = Mock(return_value=0)

        original_argv = ["prog", "detect", "--arg1", "value1"]
        with patch("sys.argv", original_argv.copy()):
            with patch(
                "flext_infra.deps.__main__.importlib.import_module",
                return_value=mock_module,
            ):
                with patch(
                    "flext_infra.deps.__main__.FlextRuntime.ensure_structlog_configured"
                ):
                    main()
                    # sys.argv should be modified to include subcommand name
                    assert "detect" in sys.argv[0]

    def test_main_passes_remaining_args_to_subcommand(self) -> None:
        """Test main passes remaining arguments to subcommand."""
        mock_module = Mock()
        mock_module.main = Mock(return_value=0)

        original_argv = ["prog", "detect", "-q", "--no-fail"]
        with patch("sys.argv", original_argv.copy()):
            with patch(
                "flext_infra.deps.__main__.importlib.import_module",
                return_value=mock_module,
            ):
                with patch(
                    "flext_infra.deps.__main__.FlextRuntime.ensure_structlog_configured"
                ):
                    main()
                    # Check that remaining args are in sys.argv
                    assert "-q" in sys.argv
                    assert "--no-fail" in sys.argv


class TestMainStructlogConfiguration:
    """Test main function structlog configuration."""

    def test_main_ensures_structlog_configured(self) -> None:
        """Test main ensures structlog is configured."""
        mock_module = Mock()
        mock_module.main = Mock(return_value=0)

        with patch("sys.argv", ["prog", "detect"]):
            with patch(
                "flext_infra.deps.__main__.FlextRuntime.ensure_structlog_configured"
            ) as mock_ensure:
                with patch(
                    "flext_infra.deps.__main__.importlib.import_module",
                    return_value=mock_module,
                ):
                    main()
                    mock_ensure.assert_called_once()

    def test_main_ensures_structlog_before_dispatch(self) -> None:
        """Test main ensures structlog is configured before dispatch."""
        mock_module = Mock()
        mock_module.main = Mock(return_value=0)
        call_order = []

        def track_ensure():
            call_order.append("ensure")

        def track_import(*args: object, **kwargs: object) -> object:
            call_order.append("import")
            return mock_module

        with patch("sys.argv", ["prog", "detect"]):
            with patch(
                "flext_infra.deps.__main__.FlextRuntime.ensure_structlog_configured",
                side_effect=track_ensure,
            ):
                with patch(
                    "flext_infra.deps.__main__.importlib.import_module",
                    side_effect=track_import,
                ):
                    main()
                    assert call_order[0] == "ensure"
                    assert call_order[1] == "import"


class TestMainExceptionHandling:
    """Test main function exception handling."""

    def test_main_subcommand_exception_propagates(self) -> None:
        """Test main propagates subcommand exceptions."""
        mock_module = Mock()
        mock_module.main = Mock(side_effect=Exception("Test error"))

        with patch("sys.argv", ["prog", "detect"]):
            with patch(
                "flext_infra.deps.__main__.importlib.import_module",
                return_value=mock_module,
            ):
                with patch(
                    "flext_infra.deps.__main__.FlextRuntime.ensure_structlog_configured"
                ):
                    with pytest.raises(Exception, match="Test error"):
                        main()

    def test_main_calls_sys_exit(self) -> None:
        """Test main() calls sys.exit."""
        with patch("sys.argv", ["deps", "detect"]):
            with patch("flext_infra.deps.__main__.main") as mock_main:
                mock_main.return_value = 0
                with patch("sys.exit") as _mock_exit:
                    try:
                        main_func()
                    except SystemExit:
                        pass
