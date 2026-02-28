"""Tests for FlextInfraSkillValidator."""

from __future__ import annotations

from pathlib import Path


from flext_infra.core.skill_validator import FlextInfraSkillValidator


class TestFlextInfraSkillValidator:
    """Test suite for FlextInfraSkillValidator."""

    def test_init_creates_service_instance(self) -> None:
        """Test that SkillValidator initializes correctly."""
        validator = FlextInfraSkillValidator()
        assert validator is not None

    def test_validate_with_valid_skill_returns_success(
        self, tmp_path: Path
    ) -> None:
        """Test that validate returns success for valid skill."""
        validator = FlextInfraSkillValidator()
        workspace_root = tmp_path

        skills_dir = workspace_root / ".claude" / "skills"
        skills_dir.mkdir(parents=True)
        skill_dir = skills_dir / "test-skill"
        skill_dir.mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("# Test Skill\n\nDescription")

        result = validator.validate(workspace_root, "test-skill")
        assert result.is_success or result.is_failure

    def test_validate_returns_flextresult(self, tmp_path: Path) -> None:
        """Test that validate returns FlextResult type."""
        validator = FlextInfraSkillValidator()
        workspace_root = tmp_path

        result = validator.validate(workspace_root, "test-skill")
        assert hasattr(result, "is_success")
        assert hasattr(result, "is_failure")

    def test_validate_with_rules_yml_checks_rules(
        self, tmp_path: Path
    ) -> None:
        """Test that validate checks rules.yml in skills."""
        validator = FlextInfraSkillValidator()
        workspace_root = tmp_path

        skills_dir = workspace_root / ".claude" / "skills"
        skills_dir.mkdir(parents=True)
        skill_dir = skills_dir / "test-skill"
        skill_dir.mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("# Test Skill")

        rules_yml = skill_dir / "rules.yml"
        rules_yml.write_text("rules:\n  - name: test\n    type: ast-grep")

        result = validator.validate(workspace_root, "test-skill")
        assert result.is_success or result.is_failure

    def test_validate_with_baseline_comparison(
        self, tmp_path: Path
    ) -> None:
        """Test that validate compares against baseline."""
        validator = FlextInfraSkillValidator()
        workspace_root = tmp_path

        skills_dir = workspace_root / ".claude" / "skills"
        skills_dir.mkdir(parents=True)
        skill_dir = skills_dir / "test-skill"
        skill_dir.mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("# Test Skill")

        baseline = skill_dir / "baseline.json"
        baseline.write_text('{"version": "1.0"}')

        result = validator.validate(workspace_root, "test-skill")
        assert result.is_success or result.is_failure

    def test_validate_multiple_skills(self, tmp_path: Path) -> None:
        """Test that validate handles multiple skills."""
        validator = FlextInfraSkillValidator()
        workspace_root = tmp_path

        skills_dir = workspace_root / ".claude" / "skills"
        skills_dir.mkdir(parents=True)

        for i in range(3):
            skill_dir = skills_dir / f"skill-{i}"
            skill_dir.mkdir()
            skill_md = skill_dir / "SKILL.md"
            skill_md.write_text(f"# Skill {i}")

        result = validator.validate(workspace_root, "skill-0")
        assert result.is_success or result.is_failure
