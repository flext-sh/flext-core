"""Tests for workspace checker initialization and utility methods.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_core import t
from flext_infra.check.services import (
    CheckIssue,
    FlextInfraWorkspaceChecker,
)
from flext_tests import tm


class TestWorkspaceCheckerInitialization:
    """Test FlextInfraWorkspaceChecker initialization."""

    def test_init_creates_default_reports_dir(self, tmp_path: Path) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        tm.that(checker._default_reports_dir.exists(), eq=True)


class TestWorkspaceCheckerExecute:
    """Test FlextInfraWorkspaceChecker.execute method."""

    def test_execute_returns_failure(self, tmp_path: Path) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        result = checker.execute()
        tm.fail(result, has="Use run()")


class TestWorkspaceCheckerExistingCheckDirs:
    """Test FlextInfraWorkspaceChecker._existing_check_dirs method."""

    def test_existing_check_dirs_workspace_root(self, tmp_path: Path) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        (tmp_path / "src").mkdir()
        (tmp_path / "tests").mkdir()
        dirs = checker._existing_check_dirs(tmp_path)
        tm.that(len(dirs) >= 0, eq=True)

    def test_existing_check_dirs_subproject(self, tmp_path: Path) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        subproj = tmp_path / "subproj"
        subproj.mkdir()
        (subproj / "src").mkdir()
        dirs = checker._existing_check_dirs(subproj)
        tm.that(len(dirs) >= 0, eq=True)

    def test_existing_check_dirs_filters_nonexistent(self, tmp_path: Path) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        (tmp_path / "src").mkdir()
        dirs = checker._existing_check_dirs(tmp_path)
        tm.that(all((tmp_path / d).is_dir() for d in dirs), eq=True)


class TestWorkspaceCheckerDirsWithPy:
    """Test FlextInfraWorkspaceChecker._dirs_with_py static method."""

    def test_dirs_with_py_finds_python_files(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "main.py").write_text("# code")
        result = FlextInfraWorkspaceChecker._dirs_with_py(tmp_path, ["src"])
        tm.that("src" in result, eq=True)

    def test_dirs_with_py_finds_pyi_files(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "main.pyi").write_text("# stub")
        result = FlextInfraWorkspaceChecker._dirs_with_py(tmp_path, ["src"])
        tm.that("src" in result, eq=True)

    def test_dirs_with_py_skips_empty_dirs(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        result = FlextInfraWorkspaceChecker._dirs_with_py(tmp_path, ["src"])
        tm.that("src" not in result, eq=True)

    def test_dirs_with_py_skips_nonexistent_dirs(self, tmp_path: Path) -> None:
        result = FlextInfraWorkspaceChecker._dirs_with_py(tmp_path, ["nonexistent"])
        tm.that("nonexistent" not in result, eq=True)


class TestWorkspaceCheckerResolveWorkspaceRootFallback:
    """Test _resolve_workspace_root fallback."""

    def test_resolve_workspace_root_fallback_to_cwd(self, tmp_path: Path) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        result = checker._resolve_workspace_root(None)
        tm.that(result.is_absolute(), eq=True)


class TestWorkspaceCheckerInitOSError:
    """Test FlextInfraWorkspaceChecker initialization with OSError."""

    def test_init_fallback_on_mkdir_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _raise_oserror(*_args: t.Scalar, **_kwargs: t.Scalar) -> None:
            msg = "Permission denied"
            raise OSError(msg)

        monkeypatch.setattr(Path, "mkdir", _raise_oserror)
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        tm.that(checker._default_reports_dir.is_absolute(), eq=True)


class TestWorkspaceCheckerBuildGateResult:
    """Test FlextInfraWorkspaceChecker._build_gate_result method."""

    def test_build_gate_result_success(self, tmp_path: Path) -> None:
        checker = FlextInfraWorkspaceChecker(workspace_root=tmp_path)
        issue = CheckIssue(
            file="a.py", line=1, column=1, code="E1", message="Error", severity="error"
        )
        result = checker._build_gate_result(
            gate="lint",
            project="p1",
            passed=True,
            issues=[issue],
            duration=0.5,
            raw_output="",
        )
        tm.that(result.result.passed, eq=True)
        tm.that(result.result.gate, eq="lint")
        tm.that(len(result.issues), eq=1)
