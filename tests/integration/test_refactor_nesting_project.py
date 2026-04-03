"""Project-level integration tests for class nesting refactor."""

from __future__ import annotations

from pathlib import Path

from flext_infra import (
    FlextInfraClassNestingRefactorRule as ClassNestingRefactorRule,
    m,
    u,
)


def _apply_rule(
    workspace_root: Path,
    file_path: Path,
    rule: ClassNestingRefactorRule,
    *,
    dry_run: bool,
) -> m.Infra.Result:
    rope_project = u.Infra.init_rope_project(workspace_root)
    try:
        resource = u.Infra.get_resource_from_path(rope_project, file_path)
        if resource is None:
            raise FileNotFoundError(file_path)
        return rule.apply(rope_project, resource, dry_run=dry_run)
    finally:
        rope_project.close()


class TestProjectLevelRefactor:
    """Test class nesting refactor across a project."""

    def test_project_processes_without_errors(self, tmp_path: Path) -> None:
        """Test that full project processes without errors."""
        src_dir = tmp_path / "src" / "test_project"
        src_dir.mkdir(parents=True)
        test_file = src_dir / "dispatcher.py"
        test_file.write_text(
            "\nclass TimeoutEnforcer:\n    pass\n\nclass RateLimiter:\n    pass\n",
        )
        config_file = tmp_path / "mappings.yml"
        config_file.write_text(
            "\nclass_nesting:\n  - loose_name: TimeoutEnforcer\n    current_file: src/test_project/dispatcher.py\n    target_namespace: FlextDispatcher\n    target_name: TimeoutEnforcer\n    confidence: high\n  - loose_name: RateLimiter\n    current_file: src/test_project/dispatcher.py\n    target_namespace: FlextDispatcher\n    target_name: RateLimiter\n    confidence: high\n",
        )
        rule = ClassNestingRefactorRule(config_file)
        result = _apply_rule(tmp_path, test_file, rule, dry_run=True)
        assert result.success
        assert result.modified is True

    def test_no_type_errors_introduced(self, tmp_path: Path) -> None:
        """Verify no type errors are introduced by refactoring."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        test_file = src_dir / "test.py"
        test_file.write_text(
            "\nfrom typing import Optional\n\nclass Helper:\n    def process(self, x: Optional[int] = None) -> int:\n        return x or 0\n",
        )
        config_file = tmp_path / "mappings.yml"
        config_file.write_text(
            "\nclass_nesting:\n  - loose_name: Helper\n    current_file: src/test.py\n    target_namespace: FlextUtilities\n    target_name: Helper\n    confidence: high\n",
        )
        rule = ClassNestingRefactorRule(config_file)
        result = _apply_rule(tmp_path, test_file, rule, dry_run=True)
        assert result.success
        assert result.refactored_code is not None
        assert (
            "Optional[int]" in result.refactored_code or "int" in result.refactored_code
        )
