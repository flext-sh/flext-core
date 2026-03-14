"""Tests for FlextInfraUtilitiesSubprocess — model, checked, and file operations.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_core import t
from flext_infra import FlextInfraUtilitiesSubprocess, m
from flext_tests import tm


class TestFlextInfraCommandRunnerExtra:
    """Test suite for FlextInfraUtilitiesSubprocess extra operations."""

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
        tm.ok(result)
        assert "sequence" in result.value.stdout

    def test_run_checked_success(self) -> None:
        """Test run_checked returns True on success."""
        runner = FlextInfraUtilitiesSubprocess()
        result = runner.run_checked(["echo", "test"])
        tm.ok(result)
        assert result.value is True

    def test_run_checked_failure(self) -> None:
        """Test run_checked returns failure on nonzero exit."""
        runner = FlextInfraUtilitiesSubprocess()
        result = runner.run_checked(["sh", "-c", "exit 1"])
        tm.fail(result)
        assert isinstance(result.error, str)
        assert "command failed" in result.error.lower()

    def test_run_to_file_success(self, tmp_path: Path) -> None:
        """Test run_to_file writes output to file."""
        runner = FlextInfraUtilitiesSubprocess()
        output_file = tmp_path / "output.txt"
        result = runner.run_to_file(["echo", "hello"], output_file)
        tm.ok(result)
        assert result.value == 0
        assert output_file.exists()
        assert "hello" in output_file.read_text()

    def test_run_to_file_timeout(self, tmp_path: Path) -> None:
        """Test run_to_file timeout error."""
        runner = FlextInfraUtilitiesSubprocess()
        output_file = tmp_path / "output.txt"
        result = runner.run_to_file(["sleep", "10"], output_file, timeout=1)
        tm.fail(result)
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
            tm.fail(result)
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

        def mock_run(*args: t.Scalar, **kwargs: t.Scalar) -> None:
            msg = "Invalid argument"
            raise ValueError(msg)

        monkeypatch.setattr("subprocess.run", mock_run)
        result = runner.run_to_file(["echo", "test"], output_file)
        tm.fail(result)
        assert isinstance(result.error, str)
        assert "execution error" in result.error.lower()
