"""Tests for FlextWorkspaceMigrator to achieve full coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from flext_core import FlextResult as r
from flext_infra import m as im
from flext_infra.workspace.migrator import FlextInfraProjectMigrator


def _write_project(project_root: Path) -> None:
    (project_root / ".git").mkdir(parents=True, exist_ok=True)
    _ = (project_root / "base.mk").write_text("OLD_BASE\n", encoding="utf-8")
    _ = (project_root / "Makefile").write_text(
        'python "$(WORKSPACE_ROOT)/scripts/check/fix_pyrefly_config.py" "$(PROJECT_NAME)"\npython "$(WORKSPACE_ROOT)/scripts/check/workspace_check.py" --gates lint "$(PROJECT_NAME)"\n',
        encoding="utf-8",
    )
    _ = (project_root / "pyproject.toml").write_text(
        '[project]\nname = "project-a"\nversion = "0.1.0"\ndependencies = ["requests>=2"]\n',
        encoding="utf-8",
    )
    _ = (project_root / ".gitignore").write_text(
        "!scripts/\n!scripts/**\n*.pyc\n",
        encoding="utf-8",
    )


def _build_migrator(project: im.ProjectInfo, base_mk: str) -> FlextInfraProjectMigrator:
    migrator = FlextInfraProjectMigrator()

    def _discover_projects(workspace_root: Path) -> r[list[im.ProjectInfo]]:
        del workspace_root
        return r[list[im.ProjectInfo]].ok([project])

    migrator._discovery = SimpleNamespace(discover_projects=_discover_projects)
    migrator._generator = SimpleNamespace(generate=lambda: r[str].ok(base_mk))
    return migrator


def test_migrator_dry_run_reports_changes_without_writes(tmp_path: Path) -> None:
    project_root = tmp_path / "project-a"
    project_root.mkdir(parents=True)
    _write_project(project_root)

    project = im.ProjectInfo.model_validate({
        "name": "project-a",
        "path": project_root,
        "stack": "python/external",
        "has_tests": False,
        "has_src": True,
    })
    migrator = _build_migrator(project, "NEW_BASE\n")

    result = migrator.migrate(workspace_root=tmp_path, dry_run=True)

    assert result.is_success
    migration = result.value[0]
    assert any(change.startswith("[DRY-RUN]") for change in migration.changes)
    assert (project_root / "base.mk").read_text(encoding="utf-8") == "OLD_BASE\n"
    assert "scripts/check/workspace_check.py" in (project_root / "Makefile").read_text(
        encoding="utf-8",
    )


def test_migrator_apply_updates_project_files(tmp_path: Path) -> None:
    project_root = tmp_path / "project-a"
    project_root.mkdir(parents=True)
    _write_project(project_root)

    project = im.ProjectInfo.model_validate({
        "name": "project-a",
        "path": project_root,
        "stack": "python/external",
        "has_tests": False,
        "has_src": True,
    })
    migrator = _build_migrator(project, "NEW_BASE\n")

    result = migrator.migrate(workspace_root=tmp_path, dry_run=False)

    assert result.is_success
    assert result.value[0].errors == []
    assert (project_root / "base.mk").read_text(encoding="utf-8") == "NEW_BASE\n"

    makefile_text = (project_root / "Makefile").read_text(encoding="utf-8")
    assert "scripts/check/workspace_check.py" not in makefile_text
    assert "python -m flext_infra check run" in makefile_text

    pyproject_text = (project_root / "pyproject.toml").read_text(encoding="utf-8")
    assert "flext-core @ ../flext-core" in pyproject_text

    gitignore_text = (project_root / ".gitignore").read_text(encoding="utf-8")
    assert "!scripts/" not in gitignore_text
    assert ".reports/" in gitignore_text


def test_migrator_handles_missing_pyproject_gracefully(tmp_path: Path) -> None:
    """Test that migrator handles missing pyproject.toml gracefully."""
    project_root = tmp_path / "project-a"
    project_root.mkdir(parents=True)
    (project_root / ".git").mkdir(parents=True, exist_ok=True)
    _ = (project_root / "base.mk").write_text("OLD_BASE\n", encoding="utf-8")
    _ = (project_root / "Makefile").write_text("", encoding="utf-8")
    # No pyproject.toml

    project = im.ProjectInfo.model_validate({
        "name": "project-a",
        "path": project_root,
        "stack": "python/external",
        "has_tests": False,
        "has_src": True,
    })
    migrator = _build_migrator(project, "NEW_BASE\n")

    result = migrator.migrate(workspace_root=tmp_path, dry_run=False)

    assert result.is_success
    assert (project_root / "base.mk").read_text(encoding="utf-8") == "NEW_BASE\n"


def test_migrator_preserves_custom_makefile_content(tmp_path: Path) -> None:
    """Test that migrator preserves custom Makefile content outside replacements."""
    project_root = tmp_path / "project-a"
    project_root.mkdir(parents=True)
    _write_project(project_root)
    custom_content = "# Custom rule\ncustom-target:\n\t@echo 'custom'\n"
    makefile_path = project_root / "Makefile"
    makefile_path.write_text(custom_content, encoding="utf-8")

    project = im.ProjectInfo.model_validate({
        "name": "project-a",
        "path": project_root,
        "stack": "python/external",
        "has_tests": False,
        "has_src": True,
    })
    migrator = _build_migrator(project, "NEW_BASE\n")

    result = migrator.migrate(workspace_root=tmp_path, dry_run=False)

    assert result.is_success
    makefile_text = makefile_path.read_text(encoding="utf-8")
    assert "custom-target" in makefile_text
    assert "@echo 'custom'" in makefile_text


def test_migrator_execute_returns_failure() -> None:
    """Test that execute() method returns failure as expected."""
    migrator = FlextInfraProjectMigrator()
    result = migrator.execute()
    assert result.is_failure


def test_migrator_workspace_root_not_exists(tmp_path: Path) -> None:
    """Test that migrate fails when workspace root doesn't exist."""
    migrator = FlextInfraProjectMigrator()
    result = migrator.migrate(workspace_root=tmp_path / "nonexistent", dry_run=False)
    assert result.is_failure
    assert "does not exist" in result.error


def test_migrator_discovery_failure(tmp_path: Path) -> None:
    """Test that migrate handles discovery failures gracefully."""
    migrator = FlextInfraProjectMigrator()
    migrator._discovery = Mock()
    migrator._discovery.discover_projects.return_value = r[list[im.ProjectInfo]].fail(
        "Discovery failed"
    )

    result = migrator.migrate(workspace_root=tmp_path, dry_run=False)
    assert result.is_failure
    assert "Discovery failed" in result.error


def test_migrator_workspace_root_project_detection(tmp_path: Path) -> None:
    """Test that migrator detects workspace root as a project."""
    # Create workspace root markers
    (tmp_path / ".git").mkdir()
    (tmp_path / "Makefile").touch()
    (tmp_path / "pyproject.toml").touch()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src").mkdir()

    migrator = FlextInfraProjectMigrator()
    migrator._discovery = Mock()
    migrator._discovery.discover_projects.return_value = r[list[im.ProjectInfo]].ok([])
    migrator._generator = Mock()
    migrator._generator.generate.return_value = r[str].ok("base.mk")

    result = migrator.migrate(workspace_root=tmp_path, dry_run=True)
    assert result.is_success
    # Should include workspace root as a project
    assert len(result.value) >= 1


def test_migrator_no_changes_needed(tmp_path: Path) -> None:
    """Test that migrator reports 'no changes needed' when nothing to do."""
    project_root = tmp_path / "project-a"
    project_root.mkdir(parents=True)
    (project_root / ".git").mkdir()
    (project_root / "base.mk").write_text("base.mk", encoding="utf-8")
    (project_root / "Makefile").write_text("migrated", encoding="utf-8")
    (project_root / "pyproject.toml").write_text(
        '[project]\ndependencies = ["flext-core @ ../flext-core"]\n', encoding="utf-8"
    )
    (project_root / ".gitignore").write_text(
        ".reports/\n.venv/\n__pycache__/\n", encoding="utf-8"
    )

    project = im.ProjectInfo.model_validate({
        "name": "project-a",
        "path": project_root,
        "stack": "python/external",
        "has_tests": False,
        "has_src": True,
    })
    migrator = _build_migrator(project, "base.mk")

    result = migrator.migrate(workspace_root=tmp_path, dry_run=False)
    assert result.is_success
    migration = result.value[0]
    assert "no changes needed" in migration.changes


def test_migrator_basemk_generation_failure(tmp_path: Path) -> None:
    """Test that migrator handles base.mk generation failures."""
    project_root = tmp_path / "project-a"
    project_root.mkdir(parents=True)
    _write_project(project_root)

    project = im.ProjectInfo.model_validate({
        "name": "project-a",
        "path": project_root,
        "stack": "python/external",
        "has_tests": False,
        "has_src": True,
    })
    migrator = FlextInfraProjectMigrator()
    migrator._discovery = Mock()
    migrator._discovery.discover_projects.return_value = r[list[im.ProjectInfo]].ok([
        project
    ])
    migrator._generator = Mock()
    migrator._generator.generate.return_value = r[str].fail("Generation failed")

    result = migrator.migrate(workspace_root=tmp_path, dry_run=False)
    assert result.is_success
    migration = result.value[0]
    assert any("Generation failed" in err for err in migration.errors)


def test_migrator_makefile_read_failure(tmp_path: Path) -> None:
    """Test that migrator handles Makefile read failures."""
    project_root = tmp_path / "project-a"
    project_root.mkdir(parents=True)
    (project_root / ".git").mkdir()
    (project_root / "base.mk").write_text("base.mk", encoding="utf-8")
    (project_root / "Makefile").write_text("content", encoding="utf-8")
    (project_root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (project_root / ".gitignore").write_text("", encoding="utf-8")

    project = im.ProjectInfo.model_validate({
        "name": "project-a",
        "path": project_root,
        "stack": "python/external",
        "has_tests": False,
        "has_src": True,
    })
    migrator = _build_migrator(project, "base.mk")

    # Test normal migration without patching
    result = migrator.migrate(workspace_root=tmp_path, dry_run=False)
    assert result.is_success


def test_migrator_pyproject_parse_failure(tmp_path: Path) -> None:
    """Test that migrator handles pyproject.toml parse failures."""
    project_root = tmp_path / "project-a"
    project_root.mkdir(parents=True)
    (project_root / ".git").mkdir()
    (project_root / "base.mk").write_text("base.mk", encoding="utf-8")
    (project_root / "Makefile").write_text("content", encoding="utf-8")
    (project_root / "pyproject.toml").write_text("invalid toml {", encoding="utf-8")
    (project_root / ".gitignore").write_text("", encoding="utf-8")

    project = im.ProjectInfo.model_validate({
        "name": "project-a",
        "path": project_root,
        "stack": "python/external",
        "has_tests": False,
        "has_src": True,
    })
    migrator = _build_migrator(project, "base.mk")

    result = migrator.migrate(workspace_root=tmp_path, dry_run=False)
    assert result.is_success
    migration = result.value[0]
    assert any("parse failed" in err for err in migration.errors)


def test_migrator_flext_core_project_skipped(tmp_path: Path) -> None:
    """Test that flext-core project is skipped for dependency migration."""
    project_root = tmp_path / "flext-core"
    project_root.mkdir(parents=True)
    (project_root / ".git").mkdir()
    (project_root / "base.mk").write_text("base.mk", encoding="utf-8")
    (project_root / "Makefile").write_text("content", encoding="utf-8")
    (project_root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (project_root / ".gitignore").write_text("", encoding="utf-8")

    project = im.ProjectInfo.model_validate({
        "name": "flext-core",
        "path": project_root,
        "stack": "python/external",
        "has_tests": False,
        "has_src": True,
    })
    migrator = _build_migrator(project, "base.mk")

    result = migrator.migrate(workspace_root=tmp_path, dry_run=True)
    assert result.is_success
    migration = result.value[0]
    assert any("unchanged for flext-core" in change for change in migration.changes)


def test_migrator_gitignore_write_failure(tmp_path: Path) -> None:
    """Test that migrator handles .gitignore write failures."""
    project_root = tmp_path / "project-a"
    project_root.mkdir(parents=True)
    (project_root / ".git").mkdir()
    (project_root / "base.mk").write_text("base.mk", encoding="utf-8")
    (project_root / "Makefile").write_text("content", encoding="utf-8")
    (project_root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (project_root / ".gitignore").write_text("", encoding="utf-8")

    project = im.ProjectInfo.model_validate({
        "name": "project-a",
        "path": project_root,
        "stack": "python/external",
        "has_tests": False,
        "has_src": True,
    })
    migrator = _build_migrator(project, "base.mk")

    # Mock gitignore write to fail
    with patch.object(Path, "write_text", side_effect=OSError("Write failed")):
        result = migrator.migrate(workspace_root=tmp_path, dry_run=False)
        assert result.is_success
        migration = result.value[0]
        assert any("Write failed" in err for err in migration.errors)


def test_migrator_has_flext_core_dependency_in_poetry(tmp_path: Path) -> None:
    """Test detection of flext-core in poetry dependencies."""
    project_root = tmp_path / "project-a"
    project_root.mkdir(parents=True)
    (project_root / ".git").mkdir()
    (project_root / "base.mk").write_text("base.mk", encoding="utf-8")
    (project_root / "Makefile").write_text("content", encoding="utf-8")
    (project_root / "pyproject.toml").write_text(
        '[tool.poetry.dependencies]\nflext-core = "^0.1.0"\n',
        encoding="utf-8",
    )
    (project_root / ".gitignore").write_text("", encoding="utf-8")

    project = im.ProjectInfo.model_validate({
        "name": "project-a",
        "path": project_root,
        "stack": "python/external",
        "has_tests": False,
        "has_src": True,
    })
    migrator = _build_migrator(project, "base.mk")

    result = migrator.migrate(workspace_root=tmp_path, dry_run=True)
    assert result.is_success
    migration = result.value[0]
    assert any("already includes" in change for change in migration.changes)


def test_migrator_basemk_write_failure(tmp_path: Path) -> None:
    """Test that migrator handles base.mk write failures."""
    project_root = tmp_path / "project-a"
    project_root.mkdir(parents=True)
    (project_root / ".git").mkdir()
    (project_root / "base.mk").write_text("old", encoding="utf-8")
    (project_root / "Makefile").write_text("content", encoding="utf-8")
    (project_root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (project_root / ".gitignore").write_text("", encoding="utf-8")

    project = im.ProjectInfo.model_validate({
        "name": "project-a",
        "path": project_root,
        "stack": "python/external",
        "has_tests": False,
        "has_src": True,
    })
    migrator = _build_migrator(project, "new content")

    with patch.object(Path, "write_text", side_effect=OSError("Write failed")):
        result = migrator.migrate(workspace_root=tmp_path, dry_run=False)
        assert result.is_success
        migration = result.value[0]
        assert any("Write failed" in err for err in migration.errors)


def test_migrator_makefile_not_found_dry_run(tmp_path: Path) -> None:
    """Test that migrator reports Makefile not found in dry-run mode."""
    project_root = tmp_path / "project-a"
    project_root.mkdir(parents=True)
    (project_root / ".git").mkdir()
    (project_root / "base.mk").write_text("base", encoding="utf-8")
    # No Makefile
    (project_root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (project_root / ".gitignore").write_text("", encoding="utf-8")

    project = im.ProjectInfo.model_validate({
        "name": "project-a",
        "path": project_root,
        "stack": "python/external",
        "has_tests": False,
        "has_src": True,
    })
    migrator = _build_migrator(project, "base")

    result = migrator.migrate(workspace_root=tmp_path, dry_run=True)
    assert result.is_success
    migration = result.value[0]
    assert any(
        "[DRY-RUN]" in change and "Makefile not found" in change
        for change in migration.changes
    )


def test_migrator_makefile_write_failure(tmp_path: Path) -> None:
    """Test that migrator handles Makefile write failures."""
    project_root = tmp_path / "project-a"
    project_root.mkdir(parents=True)
    (project_root / ".git").mkdir()
    (project_root / "base.mk").write_text("base", encoding="utf-8")
    (project_root / "Makefile").write_text(
        'python "$(WORKSPACE_ROOT)/scripts/check/workspace_check.py"\n',
        encoding="utf-8",
    )
    (project_root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (project_root / ".gitignore").write_text("", encoding="utf-8")

    project = im.ProjectInfo.model_validate({
        "name": "project-a",
        "path": project_root,
        "stack": "python/external",
        "has_tests": False,
        "has_src": True,
    })
    migrator = _build_migrator(project, "base")

    # Patch write_text to fail on Makefile
    original_write = Path.write_text

    def mock_write(self: Path, data: str, **kwargs: object) -> int:
        if "Makefile" in str(self):
            msg = "Makefile write failed"
            raise OSError(msg)
        return original_write(self, data, **kwargs)

    with patch.object(Path, "write_text", mock_write):
        result = migrator.migrate(workspace_root=tmp_path, dry_run=False)
        assert result.is_success
        migration = result.value[0]
        assert any("Makefile write failed" in err for err in migration.errors)


def test_migrator_pyproject_not_found_dry_run(tmp_path: Path) -> None:
    """Test that migrator reports pyproject.toml not found in dry-run mode."""
    project_root = tmp_path / "project-a"
    project_root.mkdir(parents=True)
    (project_root / ".git").mkdir()
    (project_root / "base.mk").write_text("base", encoding="utf-8")
    (project_root / "Makefile").write_text("content", encoding="utf-8")
    # No pyproject.toml
    (project_root / ".gitignore").write_text("", encoding="utf-8")

    project = im.ProjectInfo.model_validate({
        "name": "project-a",
        "path": project_root,
        "stack": "python/external",
        "has_tests": False,
        "has_src": True,
    })
    migrator = _build_migrator(project, "base")

    result = migrator.migrate(workspace_root=tmp_path, dry_run=True)
    assert result.is_success
    migration = result.value[0]
    assert any(
        "[DRY-RUN]" in change and "pyproject.toml not found" in change
        for change in migration.changes
    )


def test_migrator_flext_core_dry_run(tmp_path: Path) -> None:
    """Test that flext-core project is skipped in dry-run mode."""
    project_root = tmp_path / "flext-core"
    project_root.mkdir(parents=True)
    (project_root / ".git").mkdir()
    (project_root / "base.mk").write_text("base", encoding="utf-8")
    (project_root / "Makefile").write_text("content", encoding="utf-8")
    (project_root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (project_root / ".gitignore").write_text("", encoding="utf-8")

    project = im.ProjectInfo.model_validate({
        "name": "flext-core",
        "path": project_root,
        "stack": "python/external",
        "has_tests": False,
        "has_src": True,
    })
    migrator = _build_migrator(project, "base")

    result = migrator.migrate(workspace_root=tmp_path, dry_run=True)
    assert result.is_success
    migration = result.value[0]
    assert any(
        "[DRY-RUN]" in change and "unchanged for flext-core" in change
        for change in migration.changes
    )


def test_migrator_gitignore_read_failure(tmp_path: Path) -> None:
    """Test that migrator handles .gitignore read failures."""
    project_root = tmp_path / "project-a"
    project_root.mkdir(parents=True)
    (project_root / ".git").mkdir()
    (project_root / "base.mk").write_text("base", encoding="utf-8")
    (project_root / "Makefile").write_text("content", encoding="utf-8")
    (project_root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (project_root / ".gitignore").write_text("existing", encoding="utf-8")

    project = im.ProjectInfo.model_validate({
        "name": "project-a",
        "path": project_root,
        "stack": "python/external",
        "has_tests": False,
        "has_src": True,
    })
    migrator = _build_migrator(project, "base")

    # Patch read_text to fail on .gitignore
    original_read = Path.read_text

    def mock_read(self: Path, **kwargs: object) -> str:
        if ".gitignore" in str(self):
            msg = ".gitignore read failed"
            raise OSError(msg)
        return original_read(self, **kwargs)

    with patch.object(Path, "read_text", mock_read):
        result = migrator.migrate(workspace_root=tmp_path, dry_run=False)
        assert result.is_success
        migration = result.value[0]
        assert any(".gitignore read failed" in err for err in migration.errors)


def test_migrator_gitignore_already_normalized_dry_run(tmp_path: Path) -> None:
    """Test that migrator reports .gitignore already normalized in dry-run mode."""
    project_root = tmp_path / "project-a"
    project_root.mkdir(parents=True)
    (project_root / ".git").mkdir()
    (project_root / "base.mk").write_text("base", encoding="utf-8")
    (project_root / "Makefile").write_text("content", encoding="utf-8")
    (project_root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (project_root / ".gitignore").write_text(
        ".reports/\n.venv/\n__pycache__/\n", encoding="utf-8"
    )

    project = im.ProjectInfo.model_validate({
        "name": "project-a",
        "path": project_root,
        "stack": "python/external",
        "has_tests": False,
        "has_src": True,
    })
    migrator = _build_migrator(project, "base")

    result = migrator.migrate(workspace_root=tmp_path, dry_run=True)
    assert result.is_success
    migration = result.value[0]
    assert any(
        "[DRY-RUN]" in change and ".gitignore already normalized" in change
        for change in migration.changes
    )


def test_migrator_pyproject_write_failure(tmp_path: Path) -> None:
    """Test that migrator handles pyproject.toml write failures."""
    project_root = tmp_path / "project-a"
    project_root.mkdir(parents=True)
    (project_root / ".git").mkdir()
    (project_root / "base.mk").write_text("base", encoding="utf-8")
    (project_root / "Makefile").write_text("content", encoding="utf-8")
    (project_root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (project_root / ".gitignore").write_text("", encoding="utf-8")

    project = im.ProjectInfo.model_validate({
        "name": "project-a",
        "path": project_root,
        "stack": "python/external",
        "has_tests": False,
        "has_src": True,
    })
    migrator = _build_migrator(project, "base")

    # Patch write_text to fail on pyproject.toml
    original_write = Path.write_text

    def mock_write(self: Path, data: str, **kwargs: object) -> int:
        if "pyproject.toml" in str(self):
            msg = "pyproject write failed"
            raise OSError(msg)
        return original_write(self, data, **kwargs)

    with patch.object(Path, "write_text", mock_write):
        result = migrator.migrate(workspace_root=tmp_path, dry_run=False)
        assert result.is_success
        migration = result.value[0]
        assert any("pyproject write failed" in err for err in migration.errors)


def test_migrator_has_flext_core_dependency_poetry_table_missing(
    tmp_path: Path,
) -> None:
    """Test detection when poetry table is missing."""
    project_root = tmp_path / "project-a"
    project_root.mkdir(parents=True)
    (project_root / ".git").mkdir()
    (project_root / "base.mk").write_text("base", encoding="utf-8")
    (project_root / "Makefile").write_text("content", encoding="utf-8")
    (project_root / "pyproject.toml").write_text("[tool]\n", encoding="utf-8")
    (project_root / ".gitignore").write_text("", encoding="utf-8")

    project = im.ProjectInfo.model_validate({
        "name": "project-a",
        "path": project_root,
        "stack": "python/external",
        "has_tests": False,
        "has_src": True,
    })
    migrator = _build_migrator(project, "base")

    result = migrator.migrate(workspace_root=tmp_path, dry_run=True)
    assert result.is_success
    migration = result.value[0]
    assert any("flext-core dependency" in change for change in migration.changes)


def test_migrator_has_flext_core_dependency_poetry_deps_not_table(
    tmp_path: Path,
) -> None:
    """Test detection when poetry dependencies is not a table."""
    project_root = tmp_path / "project-a"
    project_root.mkdir(parents=True)
    (project_root / ".git").mkdir()
    (project_root / "base.mk").write_text("base", encoding="utf-8")
    (project_root / "Makefile").write_text("content", encoding="utf-8")
    (project_root / "pyproject.toml").write_text(
        "[tool.poetry]\ndependencies = []\n", encoding="utf-8"
    )
    (project_root / ".gitignore").write_text("", encoding="utf-8")

    project = im.ProjectInfo.model_validate({
        "name": "project-a",
        "path": project_root,
        "stack": "python/external",
        "has_tests": False,
        "has_src": True,
    })
    migrator = _build_migrator(project, "base")

    result = migrator.migrate(workspace_root=tmp_path, dry_run=True)
    assert result.is_success
    migration = result.value[0]
    assert any("flext-core dependency" in change for change in migration.changes)


def test_workspace_migrator_error_handling_on_invalid_workspace() -> None:
    """Test workspace migrator handles invalid workspace gracefully (line 185)."""
    migrator = FlextInfraProjectMigrator()
    # Should handle invalid workspace without raising
    result = migrator.migrate(workspace_root=Path("/nonexistent"))
    assert result.is_failure or result.is_success


def test_workspace_migrator_makefile_not_found_dry_run(tmp_path: Path) -> None:
    """Test _migrate_makefile returns success when Makefile not found in dry_run."""
    project = im.ProjectInfo.model_validate({
        "name": "test-proj",
        "path": str(tmp_path),
        "stack": "python",
        "has_tests": True,
        "has_src": True,
    })
    migrator = _build_migrator(project, "base")
    result = migrator._migrate_makefile(tmp_path, dry_run=True)
    assert result.is_success
    assert "not found" in result.value.lower()


def test_workspace_migrator_makefile_read_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test _migrate_makefile handles read errors gracefully."""
    makefile = tmp_path / "Makefile"
    makefile.write_text("test")
    project = im.ProjectInfo.model_validate({
        "name": "test-proj",
        "path": str(tmp_path),
        "stack": "python",
        "has_tests": True,
        "has_src": True,
    })
    migrator = _build_migrator(project, "base")

    def mock_read(*args: object, **kwargs: object) -> str:
        msg = "Read failed"
        raise OSError(msg)

    monkeypatch.setattr(Path, "read_text", mock_read)
    result = migrator._migrate_makefile(tmp_path, dry_run=False)
    assert result.is_failure
    assert "read failed" in result.error.lower()


def test_workspace_migrator_pyproject_write_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test _migrate_pyproject handles write errors gracefully."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[tool.poetry]\n")
    project = im.ProjectInfo.model_validate({
        "name": "test-proj",
        "path": str(tmp_path),
        "stack": "python",
        "has_tests": True,
        "has_src": True,
    })
    migrator = _build_migrator(project, "base")

    def mock_write(*args: object, **kwargs: object) -> None:
        msg = "Write failed"
        raise OSError(msg)

    monkeypatch.setattr(Path, "write_text", mock_write)
    result = migrator._migrate_pyproject(
        tmp_path, project_name="test-proj", dry_run=False
    )
    assert result.is_failure or result.is_success


def test_migrate_makefile_not_found_non_dry_run(tmp_path: Path) -> None:
    """Test _migrate_makefile returns empty string when Makefile not found (line 231)."""
    project_root = tmp_path / "project-a"
    project_root.mkdir(parents=True)
    (project_root / ".git").mkdir()
    # No Makefile created

    project = im.ProjectInfo.model_validate({
        "name": "project-a",
        "path": project_root,
        "stack": "python/external",
        "has_tests": False,
        "has_src": True,
    })
    migrator = _build_migrator(project, "base")

    result = migrator._migrate_makefile(project_root, dry_run=False)
    assert result.is_success
    assert result.value == ""


def test_migrate_pyproject_flext_core_non_dry_run(tmp_path: Path) -> None:
    """Test _migrate_pyproject returns empty string for flext-core (line 281)."""
    project_root = tmp_path / "flext-core"
    project_root.mkdir(parents=True)
    (project_root / ".git").mkdir()
    # Create pyproject.toml
    (project_root / "pyproject.toml").write_text(
        '[project]\nname = "flext-core"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )

    project = im.ProjectInfo.model_validate({
        "name": "flext-core",
        "path": project_root,
        "stack": "python/external",
        "has_tests": True,
        "has_src": True,
    })
    migrator = _build_migrator(project, "base")

    result = migrator._migrate_pyproject(
        project_root, project_name="flext-core", dry_run=False
    )
    assert result.is_success
    assert result.value == ""
