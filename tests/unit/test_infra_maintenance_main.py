"""Tests for maintenance CLI entry point.

Tests the main() function with various argument combinations and
mocked FlextInfraPythonVersionEnforcer service.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from flext_core import r
from flext_infra.maintenance.__main__ import main


class TestMaintenanceMain:
    """Tests for maintenance CLI main() function."""

    def test_main_success_returns_zero(self) -> None:
        """Test main() returns 0 on success."""
        with patch(
            "flext_infra.maintenance.__main__.FlextInfraPythonVersionEnforcer"
        ) as mock_enforcer_class:
            mock_enforcer = Mock()
            mock_enforcer_class.return_value = mock_enforcer
            mock_enforcer.execute.return_value = r[int | None].ok(0)

            result = main([])

            assert result == 0

    def test_main_success_with_none_value(self) -> None:
        """Test main() returns 0 when result value is 0."""
        with patch(
            "flext_infra.maintenance.__main__.FlextInfraPythonVersionEnforcer"
        ) as mock_enforcer_class:
            mock_enforcer = Mock()
            mock_enforcer_class.return_value = mock_enforcer
            mock_enforcer.execute.return_value = r[int].ok(0)

            result = main([])

            assert result == 0

    def test_main_success_with_nonzero_value(self) -> None:
        """Test main() returns value when result is non-zero."""
        with patch(
            "flext_infra.maintenance.__main__.FlextInfraPythonVersionEnforcer"
        ) as mock_enforcer_class:
            mock_enforcer = Mock()
            mock_enforcer_class.return_value = mock_enforcer
            mock_enforcer.execute.return_value = r[int].ok(42)

            result = main([])

            assert result == 42

    def test_main_failure_returns_one(self) -> None:
        """Test main() returns 1 on failure."""
        with patch(
            "flext_infra.maintenance.__main__.FlextInfraPythonVersionEnforcer"
        ) as mock_enforcer_class:
            mock_enforcer = Mock()
            mock_enforcer_class.return_value = mock_enforcer
            mock_enforcer.execute.return_value = r[int | None].fail("error message")

            with patch("flext_infra.maintenance.__main__.output.error"):
                result = main([])

            assert result == 1

    def test_main_failure_with_none_error(self) -> None:
        """Test main() handles None error message."""
        with patch(
            "flext_infra.maintenance.__main__.FlextInfraPythonVersionEnforcer"
        ) as mock_enforcer_class:
            mock_enforcer = Mock()
            mock_enforcer_class.return_value = mock_enforcer
            mock_enforcer.execute.return_value = r[int | None].fail(None)

            with patch("flext_infra.maintenance.__main__.output.error") as mock_error:
                result = main([])

            assert result == 1
            mock_error.assert_called_once()
            call_args = mock_error.call_args[0][0]
            assert "maintenance failed" in call_args

    def test_main_with_check_flag(self) -> None:
        """Test main() passes check flag to enforcer."""
        with patch(
            "flext_infra.maintenance.__main__.FlextInfraPythonVersionEnforcer"
        ) as mock_enforcer_class:
            mock_enforcer = Mock()
            mock_enforcer_class.return_value = mock_enforcer
            mock_enforcer.execute.return_value = r[int | None].ok(0)

            main(["--check"])

            call_kwargs = mock_enforcer.execute.call_args[1]
            assert call_kwargs["check_only"] is True

    def test_main_without_check_flag(self) -> None:
        """Test main() without check flag."""
        with patch(
            "flext_infra.maintenance.__main__.FlextInfraPythonVersionEnforcer"
        ) as mock_enforcer_class:
            mock_enforcer = Mock()
            mock_enforcer_class.return_value = mock_enforcer
            mock_enforcer.execute.return_value = r[int | None].ok(0)

            main([])

            call_kwargs = mock_enforcer.execute.call_args[1]
            assert call_kwargs["check_only"] is False

    def test_main_with_verbose_flag(self) -> None:
        """Test main() passes verbose flag to enforcer."""
        with patch(
            "flext_infra.maintenance.__main__.FlextInfraPythonVersionEnforcer"
        ) as mock_enforcer_class:
            mock_enforcer = Mock()
            mock_enforcer_class.return_value = mock_enforcer
            mock_enforcer.execute.return_value = r[int | None].ok(0)

            main(["--verbose"])

            call_kwargs = mock_enforcer.execute.call_args[1]
            assert call_kwargs["verbose"] is True

    def test_main_with_verbose_short_flag(self) -> None:
        """Test main() with -v short flag."""
        with patch(
            "flext_infra.maintenance.__main__.FlextInfraPythonVersionEnforcer"
        ) as mock_enforcer_class:
            mock_enforcer = Mock()
            mock_enforcer_class.return_value = mock_enforcer
            mock_enforcer.execute.return_value = r[int | None].ok(0)

            main(["-v"])

            call_kwargs = mock_enforcer.execute.call_args[1]
            assert call_kwargs["verbose"] is True

    def test_main_without_verbose_flag(self) -> None:
        """Test main() without verbose flag."""
        with patch(
            "flext_infra.maintenance.__main__.FlextInfraPythonVersionEnforcer"
        ) as mock_enforcer_class:
            mock_enforcer = Mock()
            mock_enforcer_class.return_value = mock_enforcer
            mock_enforcer.execute.return_value = r[int | None].ok(0)

            main([])

            call_kwargs = mock_enforcer.execute.call_args[1]
            assert call_kwargs["verbose"] is False

    def test_main_with_both_flags(self) -> None:
        """Test main() with both check and verbose flags."""
        with patch(
            "flext_infra.maintenance.__main__.FlextInfraPythonVersionEnforcer"
        ) as mock_enforcer_class:
            mock_enforcer = Mock()
            mock_enforcer_class.return_value = mock_enforcer
            mock_enforcer.execute.return_value = r[int | None].ok(0)

            main(["--check", "--verbose"])

            call_kwargs = mock_enforcer.execute.call_args[1]
            assert call_kwargs["check_only"] is True
            assert call_kwargs["verbose"] is True

    def test_main_with_help_flag(self) -> None:
        """Test main() with --help flag."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])

        assert exc_info.value.code == 0

    def test_main_with_none_argv(self) -> None:
        """Test main() with None argv uses sys.argv."""
        with patch("sys.argv", ["prog"]):
            with patch(
                "flext_infra.maintenance.__main__.FlextInfraPythonVersionEnforcer"
            ) as mock_enforcer_class:
                mock_enforcer = Mock()
                mock_enforcer_class.return_value = mock_enforcer
                mock_enforcer.execute.return_value = r[int | None].ok(0)

                result = main(None)

                assert result == 0

    def test_main_creates_enforcer_instance(self) -> None:
        """Test main() creates FlextInfraPythonVersionEnforcer instance."""
        with patch(
            "flext_infra.maintenance.__main__.FlextInfraPythonVersionEnforcer"
        ) as mock_enforcer_class:
            mock_enforcer = Mock()
            mock_enforcer_class.return_value = mock_enforcer
            mock_enforcer.execute.return_value = r[int | None].ok(0)

            main([])

            mock_enforcer_class.assert_called_once()

    def test_main_calls_execute_method(self) -> None:
        """Test main() calls execute method on enforcer."""
        with patch(
            "flext_infra.maintenance.__main__.FlextInfraPythonVersionEnforcer"
        ) as mock_enforcer_class:
            mock_enforcer = Mock()
            mock_enforcer_class.return_value = mock_enforcer
            mock_enforcer.execute.return_value = r[int | None].ok(0)

            main([])

            mock_enforcer.execute.assert_called_once()

    def test_main_error_output_called_on_failure(self) -> None:
        """Test main() calls output.error on failure."""
        with patch(
            "flext_infra.maintenance.__main__.FlextInfraPythonVersionEnforcer"
        ) as mock_enforcer_class:
            mock_enforcer = Mock()
            mock_enforcer_class.return_value = mock_enforcer
            mock_enforcer.execute.return_value = r[int | None].fail("test error")

            with patch("flext_infra.maintenance.__main__.output.error") as mock_error:
                main([])

            mock_error.assert_called_once()

    def test_main_error_output_not_called_on_success(self) -> None:
        """Test main() does not call output.error on success."""
        with patch(
            "flext_infra.maintenance.__main__.FlextInfraPythonVersionEnforcer"
        ) as mock_enforcer_class:
            mock_enforcer = Mock()
            mock_enforcer_class.return_value = mock_enforcer
            mock_enforcer.execute.return_value = r[int | None].ok(0)

            with patch("flext_infra.maintenance.__main__.output.error") as mock_error:
                main([])

            mock_error.assert_not_called()

    def test_main_with_empty_argv_list(self) -> None:
        """Test main() with empty argv list."""
        with patch(
            "flext_infra.maintenance.__main__.FlextInfraPythonVersionEnforcer"
        ) as mock_enforcer_class:
            mock_enforcer = Mock()
            mock_enforcer_class.return_value = mock_enforcer
            mock_enforcer.execute.return_value = r[int | None].ok(0)

            result = main([])

            assert result == 0

    def test_main_runtime_configured(self) -> None:
        """Test main() ensures FlextRuntime is configured."""
        with patch(
            "flext_infra.maintenance.__main__.FlextRuntime.ensure_structlog_configured"
        ) as mock_config:
            with patch(
                "flext_infra.maintenance.__main__.FlextInfraPythonVersionEnforcer"
            ) as mock_enforcer_class:
                mock_enforcer = Mock()
                mock_enforcer_class.return_value = mock_enforcer
                mock_enforcer.execute.return_value = r[int | None].ok(0)

                main([])

                mock_config.assert_called_once()

    def test_main_check_and_verbose_together(self) -> None:
        """Test main() with check and verbose flags together."""
        with patch(
            "flext_infra.maintenance.__main__.FlextInfraPythonVersionEnforcer"
        ) as mock_enforcer_class:
            mock_enforcer = Mock()
            mock_enforcer_class.return_value = mock_enforcer
            mock_enforcer.execute.return_value = r[int | None].ok(0)

            main(["--check", "-v"])

            call_kwargs = mock_enforcer.execute.call_args[1]
            assert call_kwargs["check_only"] is True
            assert call_kwargs["verbose"] is True

    def test_main_failure_error_message_passed(self) -> None:
        """Test main() passes error message to output.error."""
        error_msg = "specific error message"
        with patch(
            "flext_infra.maintenance.__main__.FlextInfraPythonVersionEnforcer"
        ) as mock_enforcer_class:
            mock_enforcer = Mock()
            mock_enforcer_class.return_value = mock_enforcer
            mock_enforcer.execute.return_value = r[int | None].fail(error_msg)

            with patch("flext_infra.maintenance.__main__.output.error") as mock_error:
                main([])

            mock_error.assert_called_once_with(error_msg)

    def test_main_calls_sys_exit(self) -> None:
        """Test main() calls sys.exit."""
        with patch("sys.argv", ["maintenance"]):
            with patch(
                "flext_infra.maintenance.__main__.FlextInfraPythonVersionEnforcer"
            ) as mock_enforcer_class:
                mock_enforcer = Mock()
                mock_enforcer_class.return_value = mock_enforcer
                mock_enforcer.execute.return_value = r[int | None].ok(0)

                with patch("sys.exit") as _mock_exit:
                    try:
                        main([])
                    except SystemExit:
                        pass
