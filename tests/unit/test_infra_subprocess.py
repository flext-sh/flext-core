"""Tests for FlextInfraCommandRunner.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_infra import FlextInfraCommandRunner, m


class TestFlextInfraCommandRunner:
    """Test suite for FlextInfraCommandRunner."""

    def test_run_raw_success(self) -> None:
        """Test successful raw command execution."""
        runner = FlextInfraCommandRunner()
        result = runner.run_raw(["echo", "hello"])

        assert result.is_success
        output = result.value
        assert isinstance(output, m.CommandOutput)
        assert "hello" in output.stdout
        assert output.exit_code == 0

    def test_run_raw_with_stderr(self) -> None:
        """Test raw command execution with stderr output."""
        runner = FlextInfraCommandRunner()
        # Use a command that writes to stderr
        result = runner.run_raw(["sh", "-c", "echo error >&2"])

        assert result.is_success
        output = result.value
        assert "error" in output.stderr

    def test_run_raw_nonzero_exit(self) -> None:
        """Test raw command execution with nonzero exit code."""
        runner = FlextInfraCommandRunner()
        result = runner.run_raw(["sh", "-c", "exit 42"])

        assert result.is_success
        output = result.value
        assert output.exit_code == 42

    def test_run_raw_with_cwd(self, tmp_path: Path) -> None:
        """Test raw command execution with working directory."""
        runner = FlextInfraCommandRunner()
        result = runner.run_raw(["pwd"], cwd=tmp_path)

        assert result.is_success
        output = result.value
        assert str(tmp_path) in output.stdout or output.stdout.strip() == str(tmp_path)

    def test_run_raw_timeout(self) -> None:
        """Test raw command execution timeout."""
        runner = FlextInfraCommandRunner()
        result = runner.run_raw(["sleep", "10"], timeout=1)

        assert result.is_failure
        assert "timeout" in result.error.lower()

    def test_run_raw_invalid_command(self) -> None:
        """Test raw command execution with invalid command."""
        runner = FlextInfraCommandRunner()
        result = runner.run_raw(["nonexistent_command_xyz"])

        assert result.is_failure

    def test_run_success(self) -> None:
        """Test successful command execution with zero exit check."""
        runner = FlextInfraCommandRunner()
        result = runner.run(["echo", "hello"])

        assert result.is_success
        assert "hello" in result.value.stdout

    def test_run_nonzero_exit_failure(self) -> None:
        """Test command execution fails on nonzero exit."""
        runner = FlextInfraCommandRunner()
        result = runner.run(["sh", "-c", "exit 1"])

        assert result.is_failure
        assert "command failed" in result.error.lower()

    def test_run_with_cwd(self, tmp_path: Path) -> None:
        """Test command execution with working directory."""
        runner = FlextInfraCommandRunner()
        result = runner.run(["pwd"], cwd=tmp_path)

        assert result.is_success

    def test_run_timeout(self) -> None:
        """Test command execution timeout."""
        runner = FlextInfraCommandRunner()
        result = runner.run(["sleep", "10"], timeout=1)

        assert result.is_failure
        assert "timeout" in result.error.lower()

    def test_capture_success(self) -> None:
        """Test successful output capture."""
        runner = FlextInfraCommandRunner()
        result = runner.capture(["echo", "captured"])

        assert result.is_success
        assert "captured" in result.value

    def test_capture_strips_whitespace(self) -> None:
        """Test capture strips trailing whitespace."""
        runner = FlextInfraCommandRunner()
        result = runner.capture(["echo", "text"])

        assert result.is_success
        assert result.value == "text"

    def test_capture_nonzero_exit_failure(self) -> None:
        """Test capture fails on nonzero exit."""
        runner = FlextInfraCommandRunner()
        result = runner.capture(["sh", "-c", "exit 1"])

        assert result.is_failure

    def test_capture_with_cwd(self, tmp_path: Path) -> None:
        """Test capture with working directory."""
        runner = FlextInfraCommandRunner()
        result = runner.capture(["pwd"], cwd=tmp_path)

        assert result.is_success

    def test_capture_timeout(self) -> None:
        """Test capture timeout."""
        runner = FlextInfraCommandRunner()
        result = runner.capture(["sleep", "10"], timeout=1)

        assert result.is_failure

    def test_run_with_env(self, tmp_path: Path) -> None:
        """Test command execution with environment override."""
        runner = FlextInfraCommandRunner()
        env = {"TEST_VAR": "test_value"}
        result = runner.run(["sh", "-c", "echo $TEST_VAR"], env=env)

        assert result.is_success
        assert "test_value" in result.value.stdout

    def test_capture_with_env(self) -> None:
        """Test capture with environment override."""
        runner = FlextInfraCommandRunner()
        env = {"TEST_VAR": "captured_value"}
        result = runner.capture(["sh", "-c", "echo $TEST_VAR"], env=env)

        assert result.is_success
        assert "captured_value" in result.value

    def test_run_raw_with_env(self) -> None:
        """Test raw command with environment override."""
        runner = FlextInfraCommandRunner()
        env = {"TEST_VAR": "raw_value"}
        result = runner.run_raw(["sh", "-c", "echo $TEST_VAR"], env=env)

        assert result.is_success
        assert "raw_value" in result.value.stdout

    def test_command_output_model(self) -> None:
        """Test CommandOutput model creation."""
        output = m.CommandOutput(stdout="out", stderr="err", exit_code=0)

        assert output.stdout == "out"
        assert output.stderr == "err"
        assert output.exit_code == 0

    def test_run_with_sequence_input(self) -> None:
        """Test run accepts sequence of strings."""
        runner = FlextInfraCommandRunner()
        cmd_list = ["echo", "sequence"]
        result = runner.run(cmd_list)

        assert result.is_success
        assert "sequence" in result.value.stdout
