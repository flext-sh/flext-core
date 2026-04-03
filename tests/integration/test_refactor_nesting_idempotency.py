"""Idempotency tests for class nesting refactor."""

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


class TestIdempotency:
    """Test that running refactor multiple times is idempotent."""

    def test_first_run_produces_changes(self, tmp_path: Path) -> None:
        """First run should produce changes."""
        test_file = tmp_path / "test.py"
        test_file.write_text("\nclass TimeoutEnforcer:\n    pass\n")
        config_file = tmp_path / "mappings.yml"
        config_file.write_text(
            "\nclass_nesting:\n  - loose_name: TimeoutEnforcer\n    current_file: test.py\n    target_namespace: FlextDispatcher\n    target_name: TimeoutEnforcer\n    confidence: high\n",
        )
        rule = ClassNestingRefactorRule(config_file)
        result = _apply_rule(tmp_path, test_file, rule, dry_run=False)
        assert result.modified is True
        assert result.refactored_code is not None
        assert "class FlextDispatcher:" in result.refactored_code

    def test_second_run_produces_no_changes(self, tmp_path: Path) -> None:
        """Second run on already-refactored code should produce no changes."""
        test_file = tmp_path / "test.py"
        test_file.write_text("\nclass TimeoutEnforcer:\n    pass\n")
        config_file = tmp_path / "mappings.yml"
        config_file.write_text(
            "\nclass_nesting:\n  - loose_name: TimeoutEnforcer\n    current_file: test.py\n    target_namespace: FlextDispatcher\n    target_name: TimeoutEnforcer\n    confidence: high\n",
        )
        rule = ClassNestingRefactorRule(config_file)
        result1 = _apply_rule(tmp_path, test_file, rule, dry_run=False)
        assert result1.refactored_code is not None
        test_file.write_text(result1.refactored_code)
        result2 = _apply_rule(tmp_path, test_file, rule, dry_run=True)
        assert result2.success

    def test_third_run_produces_no_changes(self, tmp_path: Path) -> None:
        """Third run should also produce no changes."""
        test_file = tmp_path / "test.py"
        test_file.write_text("\nclass TimeoutEnforcer:\n    pass\n")
        config_file = tmp_path / "mappings.yml"
        config_file.write_text(
            "\nclass_nesting:\n  - loose_name: TimeoutEnforcer\n    current_file: test.py\n    target_namespace: FlextDispatcher\n    target_name: TimeoutEnforcer\n    confidence: high\n",
        )
        rule = ClassNestingRefactorRule(config_file)
        for _ in range(3):
            result = _apply_rule(tmp_path, test_file, rule, dry_run=False)
            if result.modified and result.refactored_code is not None:
                test_file.write_text(result.refactored_code)
        final_result = _apply_rule(tmp_path, test_file, rule, dry_run=True)
        assert final_result.success
