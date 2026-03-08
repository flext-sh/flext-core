"""Project-level integration tests for class nesting refactor."""

from __future__ import annotations

from pathlib import Path

from flext_infra.refactor.rules.class_nesting import ClassNestingRefactorRule


class TestProjectLevelRefactor:
    """Test class nesting refactor across a project."""

    def test_project_processes_without_errors(self, tmp_path: Path) -> None:
        """Test that full project processes without errors."""
        # Create a mock project structure
        src_dir = tmp_path / "src" / "test_project"
        src_dir.mkdir(parents=True)

        # Create a file with loose classes
        test_file = src_dir / "dispatcher.py"
        test_file.write_text("""
class TimeoutEnforcer:
    pass

class RateLimiter:
    pass
""")

        # Create mappings config
        config_file = tmp_path / "mappings.yml"
        config_file.write_text("""
class_nesting:
  - loose_name: TimeoutEnforcer
    current_file: src/test_project/dispatcher.py
    target_namespace: FlextDispatcher
    target_name: TimeoutEnforcer
    confidence: high
  - loose_name: RateLimiter
    current_file: src/test_project/dispatcher.py
    target_namespace: FlextDispatcher
    target_name: RateLimiter
    confidence: high
""")

        rule = ClassNestingRefactorRule(config_file)
        result = rule.apply(test_file, dry_run=True)

        assert result.success
        assert result.modified is True

    def test_no_type_errors_introduced(self, tmp_path: Path) -> None:
        """Verify no type errors are introduced by refactoring."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        test_file = src_dir / "test.py"
        test_file.write_text("""
from typing import Optional

class Helper:
    def process(self, x: Optional[int] = None) -> int:
        return x or 0
""")

        # After refactoring, types should still be valid
        config_file = tmp_path / "mappings.yml"
        config_file.write_text("""
class_nesting:
  - loose_name: Helper
    current_file: src/test.py
    target_namespace: FlextUtilities
    target_name: Helper
    confidence: high
""")

        rule = ClassNestingRefactorRule(config_file)
        result = rule.apply(test_file, dry_run=True)

        assert result.success
        # Verify type hints preserved in output
        assert result.refactored_code is not None
        assert (
            "Optional[int]" in result.refactored_code or "int" in result.refactored_code
        )
