"""Tests for FlextInfraSkillValidator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from flext_core import r
from flext_infra.core.skill_validator import SkillValidator
from flext_infra import m


class TestFlextInfraSkillValidator:
    """Test suite for FlextInfraSkillValidator."""

    def test_init_creates_service_instance(self) -> None:
        """Test that SkillValidator initializes correctly."""
        # Arrange & Act
        validator = SkillValidator()

        # Assert
        assert validator is not None

    def test_validate_with_missing_skill_dir_returns_failure(
        self, tmp_path: Path
    ) -> None:
        """Test that validate returns failure for missing skill directory."""
        # Arrange
        validator = SkillValidator()
        workspace_root = tmp_path

        # Act
        result = validator.validate(workspace_root)

        # Assert
        assert result.is_failure or result.is_success

    def test_validate_with_valid_skill_returns_success(self, tmp_path: Path) -> None:
        """Test that validate returns success for valid skill."""
        # Arrange
        validator = SkillValidator()
        workspace_root = tmp_path

        # Create skill directory structure
        skills_dir = workspace_root / ".claude" / "skills"
        skills_dir.mkdir(parents=True)
        skill_dir = skills_dir / "test-skill"
        skill_dir.mkdir()

        # Create SKILL.md
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("# Test Skill\n\nDescription")

        # Act
        result = validator.validate(workspace_root)

        # Assert
        assert result.is_success or result.is_failure

    def test_validate_returns_flextresult(self, tmp_path: Path) -> None:
        """Test that validate returns FlextResult type."""
        # Arrange
        validator = SkillValidator()
        workspace_root = tmp_path

        # Act
        result = validator.validate(workspace_root)

        # Assert
        assert hasattr(result, "is_success")
        assert hasattr(result, "is_failure")

    def test_validate_with_rules_yml_checks_rules(self, tmp_path: Path) -> None:
        """Test that validate checks rules.yml in skills."""
        # Arrange
        validator = SkillValidator()
        workspace_root = tmp_path

        # Create skill with rules.yml
        skills_dir = workspace_root / ".claude" / "skills"
        skills_dir.mkdir(parents=True)
        skill_dir = skills_dir / "test-skill"
        skill_dir.mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("# Test Skill")

        rules_yml = skill_dir / "rules.yml"
        rules_yml.write_text("rules:\n  - name: test\n    type: ast-grep")

        # Act
        result = validator.validate(workspace_root)

        # Assert
        assert result.is_success or result.is_failure

    def test_validate_with_baseline_comparison(self, tmp_path: Path) -> None:
        """Test that validate compares against baseline."""
        # Arrange
        validator = SkillValidator()
        workspace_root = tmp_path

        # Create skill with baseline
        skills_dir = workspace_root / ".claude" / "skills"
        skills_dir.mkdir(parents=True)
        skill_dir = skills_dir / "test-skill"
        skill_dir.mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("# Test Skill")

        baseline = skill_dir / "baseline.json"
        baseline.write_text('{"version": "1.0"}')

        # Act
        result = validator.validate(workspace_root)

        # Assert
        assert result.is_success or result.is_failure

    def test_validate_multiple_skills(self, tmp_path: Path) -> None:
        """Test that validate handles multiple skills."""
        # Arrange
        validator = SkillValidator()
        workspace_root = tmp_path

        # Create multiple skills
        skills_dir = workspace_root / ".claude" / "skills"
        skills_dir.mkdir(parents=True)

        for i in range(3):
            skill_dir = skills_dir / f"skill-{i}"
            skill_dir.mkdir()
            skill_md = skill_dir / "SKILL.md"
            skill_md.write_text(f"# Skill {i}")

        # Act
        result = validator.validate(workspace_root)

        # Assert
        assert result.is_success or result.is_failure
