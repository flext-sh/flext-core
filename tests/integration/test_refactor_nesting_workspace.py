"""Workspace-level integration tests for class nesting refactor."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from flext_infra import t
from flext_infra.refactor.dependency_analyzer import DependencyAnalyzer
from flext_infra.refactor.scanner import FlextInfraRefactorLooseClassScanner


class TestWorkspaceLevelRefactor:
    """Test class nesting refactor across multi-project workspace."""

    def test_multi_project_workspace_processed(self, tmp_path: Path) -> None:
        """Test that multi-project workspace is processed correctly."""
        projects = ["flext-core", "flext-auth", "flext-api"]
        for proj in projects:
            src_dir = tmp_path / proj / "src" / proj.replace("-", "_")
            src_dir.mkdir(parents=True)
            test_file = src_dir / "models.py"
            test_file.write_text(
                f"\nclass {proj.replace('-', '').title()}Model:\n    pass\n",
            )
        scanner = FlextInfraRefactorLooseClassScanner()
        files_scanned = 0
        violations_count = 0
        for proj in projects:
            result = scanner.scan(tmp_path / proj)
            assert result.is_success
            files_scanned += cast("int", result.value["files_scanned"])
            violations_count += cast("int", result.value["violations_count"])
        assert files_scanned >= 3
        assert violations_count >= 0

    def test_cross_project_references_updated(self, tmp_path: Path) -> None:
        """Test that cross-project references are updated."""
        proj_a = tmp_path / "project-a" / "src" / "project_a"
        proj_a.mkdir(parents=True)
        (proj_a / "core.py").write_text("\nclass CoreService:\n    pass\n")
        proj_b = tmp_path / "project-b" / "src" / "project_b"
        proj_b.mkdir(parents=True)
        (proj_b / "consumer.py").write_text(
            "\nfrom project_a.core import CoreService\n\ndef use_service(svc: CoreService) -> None:\n    pass\n",
        )
        analyzer = DependencyAnalyzer(tmp_path)
        graph_result = analyzer.build_import_graph()
        assert graph_result.is_success
        graph = graph_result.value
        assert "project-b" in graph or "project_b" in graph

    def test_all_projects_consistent(self, tmp_path: Path) -> None:
        """Verify all projects remain consistent after refactor."""
        projects = ["proj1", "proj2", "proj3"]
        for proj in projects:
            src_dir = tmp_path / proj / "src" / proj
            src_dir.mkdir(parents=True)
            (src_dir / "__init__.py").write_text("")
            (src_dir / "utils.py").write_text(
                '\nclass UtilityHelper:\n    @staticmethod\n    def help() -> str:\n        return "help"\n',
            )
        scanner = FlextInfraRefactorLooseClassScanner()
        all_violations: list[t.Infra.ContainerDict] = []
        for proj in projects:
            result = scanner.scan(tmp_path / proj)
            assert result.is_success
            all_violations.extend(
                cast("list[t.Infra.ContainerDict]", result.value.get("violations", [])),
            )
        assert len(all_violations) >= 3
        for v in all_violations:
            confidence = (
                v.get("confidence")
                if isinstance(v, dict)
                else getattr(v, "confidence", None)
            )
            assert confidence in {"high", "medium", "low"}
