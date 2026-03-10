"""Tests for FlextInfraUtilitiesSubprocess — core run/capture operations.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_infra import FlextInfraUtilitiesSubprocess, m
from flext_tests import tm


class TestFlextInfraCommandRunnerCore:
    """Test suite for FlextInfraUtilitiesSubprocess core operations."""

    def test_run_raw_success(self) -> None:
        """Test successful raw command execution."""
        runner = FlextInfraUtilitiesSubprocess()
        result = runner.run_raw(["echo", "hello"])
        tm.ok(result)
        output = result.value
        assert isinstance(output, m.Infra.Core.CommandOutput)
        assert "hello" in output.stdout
        assert output.exit_code == 0

    def test_run_raw_with_stderr(self) -> None:
        """Test raw command execution with stderr output."""
        runner = FlextInfraUtilitiesSubprocess()
        result = runner.run_raw(["sh", "-c", "echo error >&2"])
        tm.ok(result)
        output = result.value
        assert "error" in output.stderr

    def test_run_raw_nonzero_exit(self) -> None:
        """Test raw command execution with nonzero exit code."""
        runner = FlextInfraUtilitiesSubprocess()
        result = runner.run_raw(["sh", "-c", "exit 42"])
        tm.ok(result)
        output = result.value
        assert output.exit_code == 42

    def test_run_raw_with_cwd(self, tmp_path: Path) -> None:
        """Test raw command execution with working directory."""
        runner = FlextInfraUtilitiesSubprocess()
        result = runner.run_raw(["pwd"], cwd=tmp_path)
        tm.ok(result)
        output = result.value
        assert str(tmp_path) in output.stdout or output.stdout.strip() == str(tmp_path)

    def test_run_raw_timeout(self) -> None:
        """Test raw command execution timeout."""
        runner = FlextInfraUtilitiesSubprocess()
        result = runner.run_raw(["sleep", "10"], timeout=1)
        tm.fail(result)
        assert isinstance(result.error, str)
        assert "timeout" in result.error.lower()

    def test_run_raw_invalid_command(self) -> None:
        """Test raw command execution with invalid command."""
        runner = FlextInfraUtilitiesSubprocess()
        result = runner.run_raw(["nonexistent_command_xyz"])
        tm.fail(result)

    def test_run_success(self) -> None:
        """Test successful command execution with zero exit check."""
        runner = FlextInfraUtilitiesSubprocess()
        result = runner.run(["echo", "hello"])
        tm.ok(result)
        assert "hello" in result.value.stdout

    def test_run_nonzero_exit_failure(self) -> None:
        """Test command execution fails on nonzero exit."""
        runner = FlextInfraUtilitiesSubprocess()
        result = runner.run(["sh", "-c", "exit 1"])
        tm.fail(result)
        assert isinstance(result.error, str)
        assert "command failed" in result.error.lower()

    def test_run_with_cwd(self, tmp_path: Path) -> None:
        """Test command execution with working directory."""
        runner = FlextInfraUtilitiesSubprocess()
        result = runner.run(["pwd"], cwd=tmp_path)
        tm.ok(result)

    def test_run_timeout(self) -> None:
        """Test command execution timeout."""
        runner = FlextInfraUtilitiesSubprocess()
        result = runner.run(["sleep", "10"], timeout=1)
        tm.fail(result)
        assert isinstance(result.error, str)
        assert "timeout" in result.error.lower()

    def test_capture_success(self) -> None:
        """Test successful output capture."""
        runner = FlextInfraUtilitiesSubprocess()
        result = runner.capture(["echo", "captured"])
        tm.ok(result)
        assert "captured" in result.value

    def test_capture_strips_whitespace(self) -> None:
        """Test capture strips trailing whitespace."""
        runner = FlextInfraUtilitiesSubprocess()
        result = runner.capture(["echo", "text"])
        tm.ok(result)
        assert result.value == "text"

    def test_capture_nonzero_exit_failure(self) -> None:
        """Test capture fails on nonzero exit."""
        runner = FlextInfraUtilitiesSubprocess()
        result = runner.capture(["sh", "-c", "exit 1"])
        tm.fail(result)

    def test_capture_with_cwd(self, tmp_path: Path) -> None:
        """Test capture with working directory."""
        runner = FlextInfraUtilitiesSubprocess()
        result = runner.capture(["pwd"], cwd=tmp_path)
        tm.ok(result)

    def test_capture_timeout(self) -> None:
        """Test capture timeout."""
        runner = FlextInfraUtilitiesSubprocess()
        result = runner.capture(["sleep", "10"], timeout=1)
        tm.fail(result)

    def test_run_with_env(self, tmp_path: Path) -> None:
        """Test command execution with environment override."""
        runner = FlextInfraUtilitiesSubprocess()
        env = {"TEST_VAR": "test_value"}
        result = runner.run(["sh", "-c", "echo $TEST_VAR"], env=env)
        tm.ok(result)
        assert "test_value" in result.value.stdout

    def test_capture_with_env(self) -> None:
        """Test capture with environment override."""
        runner = FlextInfraUtilitiesSubprocess()
        env = {"TEST_VAR": "captured_value"}
        result = runner.capture(["sh", "-c", "echo $TEST_VAR"], env=env)
        tm.ok(result)
        assert "captured_value" in result.value

    def test_run_raw_with_env(self) -> None:
        """Test raw command with environment override."""
        runner = FlextInfraUtilitiesSubprocess()
        env = {"TEST_VAR": "raw_value"}
        result = runner.run_raw(["sh", "-c", "echo $TEST_VAR"], env=env)
        tm.ok(result)
        assert "raw_value" in result.value.stdout
