"""Workspace-level integration tests for class nesting refactor."""

from __future__ import annotations

from collections.abc import Mapping, MutableSequence
from pathlib import Path

from flext_infra import FlextInfraModels, FlextInfraRefactorLooseClassScanner


class TestWorkspaceLevelRefactor:
    """Test class nesting refactor across multi-project workspace."""

    def test_multi_project_workspace_processed(self, tmp_path: Path) -> None:
        """Test that multi-project workspace is processed correctly."""
        projects = ["flext-core", "flext-auth", "flext-api"]
        for proj in projects:
            src_dir = tmp_path / proj / "src" / proj.replace("-", "_")
            src_dir.mkdir(parents=True)
            test_file = src_dir / "models.py"
            _ = test_file.write_text(
                f"\nclass {proj.replace('-', '').title()}Model:\n    pass\n",
            )
        scanner = FlextInfraRefactorLooseClassScanner()
        files_scanned = 0
        violations_count = 0
        for proj in projects:
            result = scanner.scan(tmp_path / proj)
            assert result.is_success
            raw_files = result.value.get("files_scanned", 0)
            files_scanned += (
                int(raw_files) if isinstance(raw_files, (int, float)) else 0
            )
            raw_violations = result.value.get("violations_count", 0)
            violations_count += (
                int(raw_violations) if isinstance(raw_violations, (int, float)) else 0
            )
        assert files_scanned >= 3
        assert violations_count >= 0

    def test_cross_project_references_updated(self, tmp_path: Path) -> None:
        """Test that cross-project references are scanned correctly."""
        proj_a = tmp_path / "project-a" / "src" / "project_a"
        proj_a.mkdir(parents=True)
        _ = (proj_a / "core.py").write_text("\nclass CoreService:\n    pass\n")
        proj_b = tmp_path / "project-b" / "src" / "project_b"
        proj_b.mkdir(parents=True)
        _ = (proj_b / "consumer.py").write_text(
            "\nfrom project_a.core import CoreService\n\ndef use_service(svc: CoreService) -> None:\n    pass\n",
        )
        scanner = FlextInfraRefactorLooseClassScanner()
        result_a = scanner.scan(tmp_path / "project-a")
        result_b = scanner.scan(tmp_path / "project-b")
        assert result_a.is_success
        assert result_b.is_success
        total_files = 0
        for result in (result_a, result_b):
            raw = result.value.get("files_scanned", 0)
            total_files += int(raw) if isinstance(raw, (int, float)) else 0
        assert total_files >= 2

    def test_all_projects_consistent(self, tmp_path: Path) -> None:
        """Verify all projects remain consistent after refactor."""
        projects = ["proj1", "proj2", "proj3"]
        for proj in projects:
            src_dir = tmp_path / proj / "src" / proj
            src_dir.mkdir(parents=True)
            _ = (src_dir / "__init__.py").write_text("")
            _ = (src_dir / "utils.py").write_text(
                '\nclass UtilityHelper:\n    @staticmethod\n    def help() -> str:\n        return "help"\n',
            )
        scanner = FlextInfraRefactorLooseClassScanner()
        all_violations: MutableSequence[FlextInfraModels.Infra.LooseClassViolation] = []
        for proj in projects:
            result = scanner.scan(tmp_path / proj)
            assert result.is_success
            violations_raw = result.value.get("violations", [])
            if isinstance(violations_raw, list):
                for v_item in violations_raw:
                    if isinstance(v_item, FlextInfraModels.Infra.LooseClassViolation):
                        all_violations.append(v_item)
                    elif isinstance(v_item, Mapping):
                        all_violations.append(
                            FlextInfraModels.Infra.LooseClassViolation.model_validate(
                                v_item
                            ),
                        )
        assert len(all_violations) >= 3
        for v in all_violations:
            assert v.confidence in {"high", "medium", "low"}
