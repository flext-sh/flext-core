"""Tests for FlextInfraUtilitiesSubprocess.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_infra import FlextInfraUtilitiesSubprocess, m


class TestFlextInfraCommandRunner:
    """Test suite for FlextInfraUtilitiesSubprocess."""

    def test_run_raw_success(self) -> None:
        """Test successful raw command execution."""
        runner = FlextInfraUtilitiesSubprocess()
        result = runner.run_raw(["echo", "hello"])
        assert result.is_success
        output = result.value
        assert isinstance(output, m.Infra.Core.CommandOutput)
        assert "hello" in output.stdout
        assert output.exit_code == 0

    def test_run_raw_with_stderr(self) -> None:
        """Test raw command execution with stderr output."""
        runner = FlextInfraUtilitiesSubprocess()
        result = runner.run_raw(["sh", "-c", "echo error >&2"])
        assert result.is_success
        output = result.value
        assert "error" in output.stderr

    def test_run_raw_nonzero_exit(self) -> None:
        """Test raw command execution with nonzero exit code."""
        runner = FlextInfraUtilitiesSubprocess()
        result = runner.run_raw(["sh", "-c", "exit 42"])
        assert result.is_success
        output = result.value
        assert output.exit_code == 42

    def test_run_raw_with_cwd(self, tmp_path: Path) -> None:
        """Test raw command execution with working directory."""
        runner = FlextInfraUtilitiesSubprocess()
        result = runner.run_raw(["pwd"], cwd=tmp_path)
        assert result.is_success
        output = result.value
        assert str(tmp_path) in output.stdout or output.stdout.strip() == str(tmp_path)

    def test_run_raw_timeout(self) -> None:
        """Test raw command execution timeout."""
        runner = FlextInfraUtilitiesSubprocess()
        result = runner.run_raw(["sleep", "10"], timeout=1)
        assert result.is_failure
        assert isinstance(result.error, str)
        assert "timeout" in result.error.lower()

    def test_run_raw_invalid_command(self) -> None:
        """Test raw command execution with invalid command."""
        runner = FlextInfraUtilitiesSubprocess()
        result = runner.run_raw(["nonexistent_command_xyz"])
        assert result.is_failure

    def test_run_success(self) -> None:
        """Test successful command execution with zero exit check."""
        runner = FlextInfraUtilitiesSubprocess()
        result = runner.run(["echo", "hello"])
        assert result.is_success
        assert "hello" in result.value.stdout

    def test_run_nonzero_exit_failure(self) -> None:
        """Test command execution fails on nonzero exit."""
        runner = FlextInfraUtilitiesSubprocess()
        result = runner.run(["sh", "-c", "exit 1"])
        assert result.is_failure
        assert isinstance(result.error, str)
        assert "command failed" in result.error.lower()

    def test_run_with_cwd(self, tmp_path: Path) -> None:
        """Test command execution with working directory."""
        runner = FlextInfraUtilitiesSubprocess()
        result = runner.run(["pwd"], cwd=tmp_path)
        assert result.is_success

    def test_run_timeout(self) -> None:
        """Test command execution timeout."""
        runner = FlextInfraUtilitiesSubprocess()
        result = runner.run(["sleep", "10"], timeout=1)
        assert result.is_failure
        assert isinstance(result.error, str)
        assert "timeout" in result.error.lower()

    def test_capture_success(self) -> None:
        """Test successful output capture."""
        runner = FlextInfraUtilitiesSubprocess()
        result = runner.capture(["echo", "captured"])
        assert result.is_success
        assert "captured" in result.value

    def test_capture_strips_whitespace(self) -> None:
        """Test capture strips trailing whitespace."""
        runner = FlextInfraUtilitiesSubprocess()
        result = runner.capture(["echo", "text"])
        assert result.is_success
        assert result.value == "text"

    def test_capture_nonzero_exit_failure(self) -> None:
        """Test capture fails on nonzero exit."""
        runner = FlextInfraUtilitiesSubprocess()
        result = runner.capture(["sh", "-c", "exit 1"])
        assert result.is_failure

    def test_capture_with_cwd(self, tmp_path: Path) -> None:
        """Test capture with working directory."""
        runner = FlextInfraUtilitiesSubprocess()
        result = runner.capture(["pwd"], cwd=tmp_path)
        assert result.is_success

    def test_capture_timeout(self) -> None:
        """Test capture timeout."""
        runner = FlextInfraUtilitiesSubprocess()
        result = runner.capture(["sleep", "10"], timeout=1)
        assert result.is_failure

    def test_run_with_env(self, tmp_path: Path) -> None:
        """Test command execution with environment override."""
        runner = FlextInfraUtilitiesSubprocess()
        env = {"TEST_VAR": "test_value"}
        result = runner.run(["sh", "-c", "echo $TEST_VAR"], env=env)
        assert result.is_success
        assert "test_value" in result.value.stdout

    def test_capture_with_env(self) -> None:
        """Test capture with environment override."""
        runner = FlextInfraUtilitiesSubprocess()
        env = {"TEST_VAR": "captured_value"}
        result = runner.capture(["sh", "-c", "echo $TEST_VAR"], env=env)
        assert result.is_success
        assert "captured_value" in result.value

    def test_run_raw_with_env(self) -> None:
        """Test raw command with environment override."""
        runner = FlextInfraUtilitiesSubprocess()
        env = {"TEST_VAR": "raw_value"}
        result = runner.run_raw(["sh", "-c", "echo $TEST_VAR"], env=env)
        assert result.is_success
        assert "raw_value" in result.value.stdout

    def test_command_output_model(self) -> None:
        """Test CommandOutput model creation."""
        output = m.Infra.Core.CommandOutput(stdout="out", stderr="err", exit_code=0)
        assert output.stdout == "out"
        assert output.stderr == "err"
        assert output.exit_code == 0

    def test_run_with_sequence_input(self) -> None:
        """Test run accepts sequence of strings."""
        runner = FlextInfraUtilitiesSubprocess()
        cmd_list = ["echo", "sequence"]
        result = runner.run(cmd_list)
        assert result.is_success
        assert "sequence" in result.value.stdout

    def test_run_checked_success(self) -> None:
        """Test run_checked returns True on success."""
        runner = FlextInfraUtilitiesSubprocess()
        result = runner.run_checked(["echo", "test"])
        assert result.is_success
        assert result.value is True

    def test_run_checked_failure(self) -> None:
        """Test run_checked returns failure on nonzero exit."""
        runner = FlextInfraUtilitiesSubprocess()
        result = runner.run_checked(["sh", "-c", "exit 1"])
        assert result.is_failure
        assert isinstance(result.error, str)
        assert "command failed" in result.error.lower()

    def test_run_to_file_success(self, tmp_path: Path) -> None:
        """Test run_to_file writes output to file."""
        runner = FlextInfraUtilitiesSubprocess()
        output_file = tmp_path / "output.txt"
        result = runner.run_to_file(["echo", "hello"], output_file)
        assert result.is_success
        assert result.value == 0
        assert output_file.exists()
        assert "hello" in output_file.read_text()

    def test_run_to_file_timeout(self, tmp_path: Path) -> None:
        """Test run_to_file timeout error."""
        runner = FlextInfraUtilitiesSubprocess()
        output_file = tmp_path / "output.txt"
        result = runner.run_to_file(["sleep", "10"], output_file, timeout=1)
        assert result.is_failure
        assert isinstance(result.error, str)
        assert "timeout" in result.error.lower()

    def test_run_to_file_oserror(self, tmp_path: Path) -> None:
        """Test run_to_file OSError handling."""
        runner = FlextInfraUtilitiesSubprocess()
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(292)
        output_file = readonly_dir / "output.txt"
        try:
            result = runner.run_to_file(["echo", "test"], output_file)
            assert result.is_failure
            assert isinstance(result.error, str)
            assert "file output error" in result.error.lower()
        finally:
            readonly_dir.chmod(493)

    def test_run_to_file_valueerror(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test run_to_file ValueError handling."""
        runner = FlextInfraUtilitiesSubprocess()
        output_file = tmp_path / "output.txt"

        def mock_run(*args: object, **kwargs: object) -> object:
            msg = "Invalid argument"
            raise ValueError(msg)

        monkeypatch.setattr("subprocess.run", mock_run)
        result = runner.run_to_file(["echo", "test"], output_file)
        assert result.is_failure
        assert isinstance(result.error, str)
        assert "execution error" in result.error.lower()
