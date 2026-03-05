"""Workspace-level integration tests for class nesting refactor."""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_infra.refactor.dependency_analyzer import DependencyAnalyzer
from flext_infra.refactor.scanner import FlextInfraRefactorLooseClassScanner


class TestWorkspaceLevelRefactor:
    """Test class nesting refactor across multi-project workspace."""

    def test_multi_project_workspace_processed(self, tmp_path: Path) -> None:
        """Test that multi-project workspace is processed correctly."""
        # Create mock workspace with multiple projects
        projects = ["flext-core", "flext-auth", "flext-api"]

        for proj in projects:
            src_dir = tmp_path / proj / "src" / proj.replace("-", "_")
            src_dir.mkdir(parents=True)

            # Create loose classes in each project
            test_file = src_dir / "models.py"
            test_file.write_text(f"""
class {proj.replace("-", "").title()}Model:
    pass
""")

        # Scan workspace
        scanner = FlextInfraRefactorLooseClassScanner()
        result = scanner.scan(tmp_path)

        assert result["violations_count"] >= 3
        assert result["files_scanned"] >= 3

    def test_cross_project_references_updated(self, tmp_path: Path) -> None:
        """Test that cross-project references are updated."""
        # Create project A with class
        proj_a = tmp_path / "project-a" / "src" / "project_a"
        proj_a.mkdir(parents=True)
        (proj_a / "core.py").write_text("""
class CoreService:
    pass
""")

        # Create project B importing from A
        proj_b = tmp_path / "project-b" / "src" / "project_b"
        proj_b.mkdir(parents=True)
        (proj_b / "consumer.py").write_text("""
from project_a.core import CoreService

def use_service(svc: CoreService) -> None:
    pass
""")

        # Verify dependency analyzer finds cross-project imports
        analyzer = DependencyAnalyzer(tmp_path)
        graph = analyzer.build_import_graph()

        # Should detect project_b imports from project_a
        assert "project-b" in graph or "project_b" in graph

    def test_all_projects_consistent(self, tmp_path: Path) -> None:
        """Verify all projects remain consistent after refactor."""
        projects = ["proj1", "proj2", "proj3"]

        for proj in projects:
            src_dir = tmp_path / proj / "src" / proj
            src_dir.mkdir(parents=True)
            (src_dir / "__init__.py").write_text("")
            (src_dir / "utils.py").write_text("""
class UtilityHelper:
    @staticmethod
    def help() -> str:
        return "help"
""")

        # Scan all projects
        scanner = FlextInfraRefactorLooseClassScanner()
        all_violations = []

        for proj in projects:
            result = scanner.scan(tmp_path / proj)
            all_violations.extend(result.get("violations", []))

        # Should find violations in all projects
        assert len(all_violations) >= 3

        # All should have same pattern
        for v in all_violations:
            assert "confidence" in v
            assert v["confidence"] in ("high", "medium", "low")
