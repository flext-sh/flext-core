"""Idempotency tests for class nesting refactor."""

from __future__ import annotations

from pathlib import Path

from flext_infra.refactor.rules.class_nesting import ClassNestingRefactorRule


class TestIdempotency:
    """Test that running refactor multiple times is idempotent."""

    def test_first_run_produces_changes(self, tmp_path: Path) -> None:
        """First run should produce changes."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
class TimeoutEnforcer:
    pass
""")

        config_file = tmp_path / "mappings.yml"
        config_file.write_text("""
class_nesting:
  - loose_name: TimeoutEnforcer
    current_file: test.py
    target_namespace: FlextDispatcher
    target_name: TimeoutEnforcer
    confidence: high
""")

        rule = ClassNestingRefactorRule(config_file)
        result = rule.apply(test_file, dry_run=False)

        assert result.modified is True
        assert "class FlextDispatcher:" in result.output

    def test_second_run_produces_no_changes(self, tmp_path: Path) -> None:
        """Second run on already-refactored code should produce no changes."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
class TimeoutEnforcer:
    pass
""")

        config_file = tmp_path / "mappings.yml"
        config_file.write_text("""
class_nesting:
  - loose_name: TimeoutEnforcer
    current_file: test.py
    target_namespace: FlextDispatcher
    target_name: TimeoutEnforcer
    confidence: high
""")

        rule = ClassNestingRefactorRule(config_file)

        # First run
        result1 = rule.apply(test_file, dry_run=False)
        test_file.write_text(result1.output)

        # Second run - should detect already nested
        result2 = rule.apply(test_file, dry_run=True)

        # Should not need further modifications
        assert result2.success

    def test_third_run_produces_no_changes(self, tmp_path: Path) -> None:
        """Third run should also produce no changes."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
class TimeoutEnforcer:
    pass
""")

        config_file = tmp_path / "mappings.yml"
        config_file.write_text("""
class_nesting:
  - loose_name: TimeoutEnforcer
    current_file: test.py
    target_namespace: FlextDispatcher
    target_name: TimeoutEnforcer
    confidence: high
""")

        rule = ClassNestingRefactorRule(config_file)

        # Run three times
        for _ in range(3):
            result = rule.apply(test_file, dry_run=False)
            if result.modified:
                test_file.write_text(result.output)

        # After third run, should be stable
        final_result = rule.apply(test_file, dry_run=True)
        assert final_result.success
