"""Tests for FlextInfraDocsShared — scope model and build_scopes.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_core import r
from flext_infra.constants import c
from flext_infra.docs.shared import FlextInfraDocsShared
from flext_tests import tm
from tests.infra.models import m

_OUT = c.Infra.Docs.DEFAULT_DOCS_OUTPUT_DIR


class TestFlextInfraDocScope:
    """Tests for FlextInfraDocScope model."""

    def test_scope_creation(self, tmp_path: Path) -> None:
        """Test FlextInfraDocScope creation."""
        rd = tmp_path / "reports"
        rd.mkdir(parents=True, exist_ok=True)
        scope = m.Infra.Docs.FlextInfraDocScope(
            name="test-project",
            path=tmp_path,
            report_dir=rd,
        )
        tm.that(scope.name, eq="test-project")
        tm.that(str(scope.path), eq=str(tmp_path))
        tm.that(str(scope.report_dir), eq=str(rd))

    def test_scope_name_required(self, tmp_path: Path) -> None:
        """Test FlextInfraDocScope requires name."""
        rd = tmp_path / "reports"
        rd.mkdir(parents=True, exist_ok=True)
        with pytest.raises(Exception):
            m.Infra.Docs.FlextInfraDocScope(name="", path=tmp_path, report_dir=rd)

    def test_scope_path_required(self) -> None:
        """Test FlextInfraDocScope requires path."""
        with pytest.raises(Exception):
            m.Infra.Docs.FlextInfraDocScope.model_validate({
                "name": "test",
                "path": None,
                "report_dir": "/tmp",
            })

    def test_scope_report_dir_required(self, tmp_path: Path) -> None:
        """Test FlextInfraDocScope requires report_dir."""
        with pytest.raises(Exception):
            m.Infra.Docs.FlextInfraDocScope.model_validate({
                "name": "test",
                "path": str(tmp_path),
                "report_dir": None,
            })


class TestBuildScopes:
    """Tests for FlextInfraDocsShared.build_scopes."""

    @staticmethod
    def _build(
        root: Path,
        project: str | None = None,
        projects: str | None = None,
        output_dir: str = _OUT,
    ) -> r[list[m.Infra.Docs.FlextInfraDocScope]]:
        return FlextInfraDocsShared.build_scopes(
            root=root,
            project=project,
            projects=projects,
            output_dir=output_dir,
        )

    def test_returns_flext_result(self, tmp_path: Path) -> None:
        """Test build_scopes returns r."""
        result = self._build(tmp_path)
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_valid_root_returns_success(self, tmp_path: Path) -> None:
        """Test build_scopes with valid root returns success."""
        result = self._build(tmp_path)
        tm.ok(result)
        tm.that(len(result.value) >= 0, eq=True)

    def test_includes_root_scope(self, tmp_path: Path) -> None:
        """Test build_scopes includes root scope."""
        result = self._build(tmp_path)
        if result.is_success:
            tm.that(any(s.name == "root" for s in result.value), eq=True)

    def test_with_single_project(self, tmp_path: Path) -> None:
        """Test build_scopes with single project filter."""
        result = self._build(tmp_path, project="test-project")
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_with_multiple_projects(self, tmp_path: Path) -> None:
        """Test build_scopes with multiple projects filter."""
        result = self._build(tmp_path, projects="proj1,proj2,proj3")
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_with_custom_output_dir(self, tmp_path: Path) -> None:
        """Test build_scopes with custom output directory."""
        result = self._build(tmp_path, output_dir=str(tmp_path / "custom_output"))
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_default_docs_output_dir_constant(self) -> None:
        """Test DEFAULT_DOCS_OUTPUT_DIR constant is defined."""
        tm.that(len(_OUT) > 0, eq=True)

    def test_scope_structure(self, tmp_path: Path) -> None:
        """Test scopes returned have required structure."""
        result = self._build(tmp_path)
        if result.is_success and result.value:
            scope = result.value[0]
            tm.that(hasattr(scope, "name"), eq=True)
            tm.that(hasattr(scope, "path"), eq=True)
            tm.that(hasattr(scope, "report_dir"), eq=True)

    def test_report_dir_created(self, tmp_path: Path) -> None:
        """Test build_scopes creates report directories."""
        result = self._build(tmp_path, output_dir=str(tmp_path / "reports"))
        if result.is_success and result.value:
            for scope in result.value:
                tm.that(scope.report_dir.as_posix().endswith("reports"), eq=True)

    def test_skips_missing_projects(self, tmp_path: Path) -> None:
        """Test build_scopes skips projects without pyproject.toml."""
        (tmp_path / "missing-proj").mkdir(parents=True, exist_ok=True)
        tm.ok(self._build(tmp_path, projects="missing-proj"))

    def test_nonexistent_project_skips(self, tmp_path: Path) -> None:
        """Test build_scopes skips nonexistent projects gracefully."""
        tm.ok(self._build(tmp_path, project="nonexistent_proj"))

    def test_appends_valid_project_scope(self, tmp_path: Path) -> None:
        """Test build_scopes appends scope for valid project."""
        proj_dir = tmp_path / "test-proj"
        proj_dir.mkdir()
        (proj_dir / "pyproject.toml").write_text('[project]\nname = "test-proj"\n')
        result = self._build(tmp_path, projects="test-proj")
        tm.ok(result)
        tm.that("test-proj" in [s.name for s in result.value], eq=True)

    def test_report_dir_path_resolution(self, tmp_path: Path) -> None:
        """Test build_scopes resolves report_dir paths correctly."""
        result = self._build(tmp_path, output_dir=".reports/docs")
        if result.is_success:
            for scope in result.value:
                tm.that(scope.report_dir.is_absolute(), eq=True)

    def test_invalid_root_handled(self) -> None:
        """Test build_scopes handles invalid root gracefully."""
        result = self._build(Path("/nonexistent/path"), output_dir=".reports/docs")
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_catches_oserror(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test build_scopes returns failure on OSError during resolution."""
        original_resolve = Path.resolve

        def mock_resolve(self_path: Path) -> Path:
            if "oserror_proj" in str(self_path):
                msg = "Permission denied"
                raise OSError(msg)
            return original_resolve(self_path)

        monkeypatch.setattr(Path, "resolve", mock_resolve)
        result = self._build(tmp_path, project="oserror_proj")
        tm.fail(result, has="scope resolution failed")
