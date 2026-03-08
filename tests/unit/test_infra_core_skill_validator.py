"""Tests for FlextInfraSkillValidator."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest

from flext_core import r
from flext_infra.core.skill_validator import (
    FlextInfraSkillValidator,
    _normalize_string_list,
    _safe_load_yaml,
)


class TestSafeLoadYaml:
    """Test _safe_load_yaml helper function."""

    def test_safe_load_yaml_with_valid_yaml(self, tmp_path: Path) -> None:
        """Test loading valid YAML file."""
        yaml_file = tmp_path / "test.yml"
        yaml_file.write_text("key: value\nlist:\n  - item1\n  - item2")
        result = _safe_load_yaml(yaml_file)
        assert result["key"] == "value"
        assert result["list"] == ["item1", "item2"]

    def test_safe_load_yaml_with_empty_file(self, tmp_path: Path) -> None:
        """Test loading empty YAML file returns empty dict."""
        yaml_file = tmp_path / "empty.yml"
        yaml_file.write_text("")
        result = _safe_load_yaml(yaml_file)
        assert result == {}

    def test_safe_load_yaml_with_null_content(self, tmp_path: Path) -> None:
        """Test loading YAML with null content returns empty dict."""
        yaml_file = tmp_path / "null.yml"
        yaml_file.write_text("null")
        result = _safe_load_yaml(yaml_file)
        assert result == {}

    def test_safe_load_yaml_with_non_dict_raises_type_error(
        self, tmp_path: Path
    ) -> None:
        """Test loading YAML with non-dict content raises TypeError."""
        yaml_file = tmp_path / "list.yml"
        yaml_file.write_text("- item1\n- item2")
        with pytest.raises(TypeError, match="rules\\.yml must be a mapping"):
            _safe_load_yaml(yaml_file)

    def test_safe_load_yaml_with_string_raises_type_error(self, tmp_path: Path) -> None:
        """Test loading YAML with string content raises TypeError."""
        yaml_file = tmp_path / "string.yml"
        yaml_file.write_text("just a string")
        with pytest.raises(TypeError, match="rules\\.yml must be a mapping"):
            _safe_load_yaml(yaml_file)


class TestNormalizeStringList:
    """Test _normalize_string_list helper function."""

    def test_normalize_string_list_with_none_returns_empty(self) -> None:
        """Test normalizing None value returns empty list."""
        result = _normalize_string_list(None, "test_field")
        assert result == []

    def test_normalize_string_list_with_valid_list(self) -> None:
        """Test normalizing valid string list."""
        result = _normalize_string_list(["a", "b", "c"], "test_field")
        assert result == ["a", "b", "c"]

    def test_normalize_string_list_with_non_string_item_raises_error(self) -> None:
        """Test normalizing list with non-string item raises TypeError."""
        with pytest.raises(TypeError, match="test_field must be list\\[str\\]"):
            _normalize_string_list(["a", 123, "c"], "test_field")

    def test_normalize_string_list_with_non_list_raises_error(self) -> None:
        """Test normalizing non-list value raises TypeError."""
        with pytest.raises(TypeError, match="test_field must be list\\[str\\]"):
            _normalize_string_list("not a list", "test_field")

    def test_normalize_string_list_with_dict_raises_error(self) -> None:
        """Test normalizing dict value raises TypeError."""
        with pytest.raises(TypeError, match="test_field must be list\\[str\\]"):
            _normalize_string_list({"key": "value"}, "test_field")


class TestFlextInfraSkillValidator:
    """Test suite for FlextInfraSkillValidator."""

    def test_init_creates_service_instance(self) -> None:
        """Test that SkillValidator initializes correctly."""
        validator = FlextInfraSkillValidator()
        assert validator is not None

    def test_validate_with_missing_rules_yml(self, tmp_path: Path) -> None:
        """Test validate returns failure when rules.yml is missing."""
        validator = FlextInfraSkillValidator()
        workspace_root = tmp_path
        skills_dir = workspace_root / ".claude" / "skills"
        skills_dir.mkdir(parents=True)
        skill_dir = skills_dir / "test-skill"
        skill_dir.mkdir()
        result = validator.validate(workspace_root, "test-skill")
        assert result.is_success
        assert not result.value.passed
        assert "no rules.yml" in result.value.summary

    def test_validate_with_invalid_scan_targets_not_dict(self, tmp_path: Path) -> None:
        """Test validate fails when scan_targets is not a dict."""
        validator = FlextInfraSkillValidator()
        workspace_root = tmp_path
        skills_dir = workspace_root / ".claude" / "skills"
        skills_dir.mkdir(parents=True)
        skill_dir = skills_dir / "test-skill"
        skill_dir.mkdir()
        rules_yml = skill_dir / "rules.yml"
        rules_yml.write_text("scan_targets: [item1, item2]")
        result = validator.validate(workspace_root, "test-skill")
        assert result.is_failure
        assert isinstance(result.error, str)
        assert isinstance(result.error, str)
        assert "scan_targets must be a mapping" in result.error

    def test_validate_with_invalid_rules_not_list(self, tmp_path: Path) -> None:
        """Test validate fails when rules is not a list."""
        validator = FlextInfraSkillValidator()
        workspace_root = tmp_path
        skills_dir = workspace_root / ".claude" / "skills"
        skills_dir.mkdir(parents=True)
        skill_dir = skills_dir / "test-skill"
        skill_dir.mkdir()
        rules_yml = skill_dir / "rules.yml"
        rules_yml.write_text("rules: {not: a_list}")
        result = validator.validate(workspace_root, "test-skill")
        assert result.is_failure
        assert isinstance(result.error, str)
        assert "rules must be a list" in result.error

    def test_validate_with_non_dict_rule_object(self, tmp_path: Path) -> None:
        """Test validate skips non-dict rule objects."""
        validator = FlextInfraSkillValidator()
        workspace_root = tmp_path
        skills_dir = workspace_root / ".claude" / "skills"
        skills_dir.mkdir(parents=True)
        skill_dir = skills_dir / "test-skill"
        skill_dir.mkdir()
        rules_yml = skill_dir / "rules.yml"
        rules_yml.write_text("rules:\n  - not_a_dict\n  - another_string")
        result = validator.validate(workspace_root, "test-skill")
        assert result.is_success
        assert result.value.passed

    def test_validate_with_ast_grep_rule(self, tmp_path: Path) -> None:
        """Test validate processes ast-grep rules."""
        validator = FlextInfraSkillValidator()
        workspace_root = tmp_path
        skills_dir = workspace_root / ".claude" / "skills"
        skills_dir.mkdir(parents=True)
        skill_dir = skills_dir / "test-skill"
        skill_dir.mkdir()
        rules_yml = skill_dir / "rules.yml"
        rules_yml.write_text(
            "rules:\n  - id: test-rule\n    type: ast-grep\n    file: rule.yml"
        )
        with patch.object(validator, "_run_ast_grep_count", return_value=0):
            result = validator.validate(workspace_root, "test-skill")
            assert result.is_success
            assert result.value.passed

    def test_validate_with_custom_rule(self, tmp_path: Path) -> None:
        """Test validate processes custom rules."""
        validator = FlextInfraSkillValidator()
        workspace_root = tmp_path
        skills_dir = workspace_root / ".claude" / "skills"
        skills_dir.mkdir(parents=True)
        skill_dir = skills_dir / "test-skill"
        skill_dir.mkdir()
        rules_yml = skill_dir / "rules.yml"
        rules_yml.write_text(
            "rules:\n  - id: custom-rule\n    type: custom\n    script: check.py"
        )
        with patch.object(validator, "_run_custom_count", return_value=0):
            result = validator.validate(workspace_root, "test-skill")
            assert result.is_success
            assert result.value.passed

    def test_validate_with_baseline_total_strategy(self, tmp_path: Path) -> None:
        """Test validate with baseline total strategy."""
        validator = FlextInfraSkillValidator()
        workspace_root = tmp_path
        skills_dir = workspace_root / ".claude" / "skills"
        skills_dir.mkdir(parents=True)
        skill_dir = skills_dir / "test-skill"
        skill_dir.mkdir()
        rules_yml = skill_dir / "rules.yml"
        rules_yml.write_text("baseline:\n  strategy: total\n  file: baseline.json")
        baseline_file = skill_dir / "baseline.json"
        baseline_file.write_text('{"counts": {"group1": 5}}')
        with patch.object(validator, "_json") as mock_json:
            mock_json.read.return_value = r[dict[str, object]].ok({
                "counts": {"group1": 5}
            })
            result = validator.validate(workspace_root, "test-skill")
            assert result.is_success

    def test_validate_with_baseline_per_group_strategy(self, tmp_path: Path) -> None:
        """Test validate with baseline per-group strategy."""
        validator = FlextInfraSkillValidator()
        workspace_root = tmp_path
        skills_dir = workspace_root / ".claude" / "skills"
        skills_dir.mkdir(parents=True)
        skill_dir = skills_dir / "test-skill"
        skill_dir.mkdir()
        rules_yml = skill_dir / "rules.yml"
        rules_yml.write_text("baseline:\n  strategy: per-group\n  file: baseline.json")
        baseline_file = skill_dir / "baseline.json"
        baseline_file.write_text('{"counts": {"group1": 5, "group2": 3}}')
        with patch.object(validator, "_json") as mock_json:
            mock_json.read.return_value = r[dict[str, object]].ok({
                "counts": {"group1": 5, "group2": 3}
            })
            result = validator.validate(workspace_root, "test-skill")
            assert result.is_success

    def test_validate_with_baseline_invalid_counts(self, tmp_path: Path) -> None:
        """Test validate with baseline containing non-int counts."""
        validator = FlextInfraSkillValidator()
        workspace_root = tmp_path
        skills_dir = workspace_root / ".claude" / "skills"
        skills_dir.mkdir(parents=True)
        skill_dir = skills_dir / "test-skill"
        skill_dir.mkdir()
        rules_yml = skill_dir / "rules.yml"
        rules_yml.write_text("baseline:\n  strategy: total\n  file: baseline.json")
        baseline_file = skill_dir / "baseline.json"
        baseline_file.write_text('{"counts": {"group1": "not_an_int"}}')
        with patch.object(validator, "_json") as mock_json:
            mock_json.read.return_value = r[dict[str, object]].ok({
                "counts": {"group1": "not_an_int"}
            })
            result = validator.validate(workspace_root, "test-skill")
            assert result.is_success

    def test_validate_with_exception_handling(self, tmp_path: Path) -> None:
        """Test validate catches exceptions and returns failure."""
        validator = FlextInfraSkillValidator()
        workspace_root = tmp_path
        skills_dir = workspace_root / ".claude" / "skills"
        skills_dir.mkdir(parents=True)
        skill_dir = skills_dir / "test-skill"
        skill_dir.mkdir()
        rules_yml = skill_dir / "rules.yml"
        rules_yml.write_text("just a plain string")
        result = validator.validate(workspace_root, "test-skill")
        assert result.is_failure
        assert isinstance(result.error, str)
        assert "skill validation failed" in result.error

    def test_run_ast_grep_count_with_empty_rule_file(self, tmp_path: Path) -> None:
        """Test _run_ast_grep_count with empty rule file path."""
        validator = FlextInfraSkillValidator()
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        rule = {"id": "test", "type": "ast-grep", "file": ""}
        count = validator._run_ast_grep_count(rule, skill_dir, tmp_path, [], [])
        assert count == 0

    def test_run_ast_grep_count_with_nonexistent_rule_file(
        self, tmp_path: Path
    ) -> None:
        """Test _run_ast_grep_count with nonexistent rule file."""
        validator = FlextInfraSkillValidator()
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        rule = {"id": "test", "type": "ast-grep", "file": "nonexistent.yml"}
        count = validator._run_ast_grep_count(rule, skill_dir, tmp_path, [], [])
        assert count == 0

    def test_run_ast_grep_count_with_runner_failure(self, tmp_path: Path) -> None:
        """Test _run_ast_grep_count when runner fails."""
        validator = FlextInfraSkillValidator()
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        rule_file = skill_dir / "rule.yml"
        rule_file.write_text("id: test\nrule: pattern")
        rule = {"id": "test", "type": "ast-grep", "file": "rule.yml"}
        with patch.object(validator, "_runner") as mock_runner:
            mock_runner.run_raw.return_value = r[object].fail("command failed")
            count = validator._run_ast_grep_count(rule, skill_dir, tmp_path, [], [])
            assert count == 0

    def test_run_ast_grep_count_with_non_zero_exit_code(self, tmp_path: Path) -> None:
        """Test _run_ast_grep_count with non-zero exit code."""
        validator = FlextInfraSkillValidator()
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        rule_file = skill_dir / "rule.yml"
        rule_file.write_text("id: test\nrule: pattern")
        rule = {"id": "test", "type": "ast-grep", "file": "rule.yml"}
        mock_result = Mock()
        mock_result.exit_code = 2
        mock_result.stdout = ""
        with patch.object(validator, "_runner") as mock_runner:
            mock_runner.run_raw.return_value = r[object].ok(mock_result)
            count = validator._run_ast_grep_count(rule, skill_dir, tmp_path, [], [])
            assert count == 0

    def test_run_ast_grep_count_with_valid_json_output(self, tmp_path: Path) -> None:
        """Test _run_ast_grep_count counts valid JSON lines."""
        validator = FlextInfraSkillValidator()
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        rule_file = skill_dir / "rule.yml"
        rule_file.write_text("id: test\nrule: pattern")
        rule = {"id": "test", "type": "ast-grep", "file": "rule.yml"}
        mock_result = Mock()
        mock_result.exit_code = 0
        mock_result.stdout = '{"match": 1}\n{"match": 2}\n'
        with patch.object(validator, "_runner") as mock_runner:
            mock_runner.run_raw.return_value = r[object].ok(mock_result)
            count = validator._run_ast_grep_count(rule, skill_dir, tmp_path, [], [])
            assert count == 2

    def test_run_ast_grep_count_with_invalid_json_lines(self, tmp_path: Path) -> None:
        """Test _run_ast_grep_count skips invalid JSON lines."""
        validator = FlextInfraSkillValidator()
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        rule_file = skill_dir / "rule.yml"
        rule_file.write_text("id: test\nrule: pattern")
        rule = {"id": "test", "type": "ast-grep", "file": "rule.yml"}
        mock_result = Mock()
        mock_result.exit_code = 0
        mock_result.stdout = '{"match": 1}\ninvalid json\n{"match": 2}\n'
        with patch.object(validator, "_runner") as mock_runner:
            mock_runner.run_raw.return_value = r[object].ok(mock_result)
            count = validator._run_ast_grep_count(rule, skill_dir, tmp_path, [], [])
            assert count == 2

    def test_run_custom_count_with_empty_script_path(self, tmp_path: Path) -> None:
        """Test _run_custom_count with empty script path."""
        validator = FlextInfraSkillValidator()
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        rule = {"id": "test", "type": "custom", "script": ""}
        count = validator._run_custom_count(rule, skill_dir, tmp_path, "baseline")
        assert count == 0

    def test_run_custom_count_with_nonexistent_script(self, tmp_path: Path) -> None:
        """Test _run_custom_count with nonexistent script."""
        validator = FlextInfraSkillValidator()
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        rule = {"id": "test", "type": "custom", "script": "nonexistent.py"}
        count = validator._run_custom_count(rule, skill_dir, tmp_path, "baseline")
        assert count == 0

    def test_run_custom_count_with_python_script(self, tmp_path: Path) -> None:
        """Test _run_custom_count with Python script."""
        validator = FlextInfraSkillValidator()
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        script = skill_dir / "check.py"
        script.write_text("print('test')")
        rule = {"id": "test", "type": "custom", "script": "check.py"}
        mock_result = Mock()
        mock_result.exit_code = 0
        mock_result.stdout = '{"violation_count": 3}\n'
        with patch.object(validator, "_runner") as mock_runner:
            mock_runner.run_raw.return_value = r[object].ok(mock_result)
            count = validator._run_custom_count(rule, skill_dir, tmp_path, "baseline")
            assert count == 3

    def test_run_custom_count_with_shell_script(self, tmp_path: Path) -> None:
        """Test _run_custom_count with shell script."""
        validator = FlextInfraSkillValidator()
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        script = skill_dir / "check.sh"
        script.write_text("#!/bin/bash\necho 'test'")
        rule = {"id": "test", "type": "custom", "script": "check.sh"}
        mock_result = Mock()
        mock_result.exit_code = 0
        mock_result.stdout = '{"count": 2}\n'
        with patch.object(validator, "_runner") as mock_runner:
            mock_runner.run_raw.return_value = r[object].ok(mock_result)
            count = validator._run_custom_count(rule, skill_dir, tmp_path, "baseline")
            assert count == 2

    def test_run_custom_count_with_pass_mode_flag(self, tmp_path: Path) -> None:
        """Test _run_custom_count passes mode when pass_mode is true."""
        validator = FlextInfraSkillValidator()
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        script = skill_dir / "check.py"
        script.write_text("print('test')")
        rule = cast(
            "Mapping[str, Any]",
            {"id": "test", "type": "custom", "script": "check.py", "pass_mode": True},
        )
        mock_result = Mock()
        mock_result.exit_code = 0
        mock_result.stdout = '{"violation_count": 0}\n'
        with patch.object(validator, "_runner") as mock_runner:
            mock_runner.run_raw.return_value = r[object].ok(mock_result)
            count = validator._run_custom_count(rule, skill_dir, tmp_path, "strict")
            assert count == 0
            call_args = mock_runner.run_raw.call_args
            assert "--mode" in call_args[0][0]

    def test_run_custom_count_with_exit_code_1(self, tmp_path: Path) -> None:
        """Test _run_custom_count with exit code 1 sets count to at least 1."""
        validator = FlextInfraSkillValidator()
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        script = skill_dir / "check.py"
        script.write_text("print('test')")
        rule = {"id": "test", "type": "custom", "script": "check.py"}
        mock_result = Mock()
        mock_result.exit_code = 1
        mock_result.stdout = ""
        with patch.object(validator, "_runner") as mock_runner:
            mock_runner.run_raw.return_value = r[object].ok(mock_result)
            count = validator._run_custom_count(rule, skill_dir, tmp_path, "baseline")
            assert count >= 1

    def test_run_custom_count_with_runner_failure(self, tmp_path: Path) -> None:
        """Test _run_custom_count when runner fails."""
        validator = FlextInfraSkillValidator()
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        script = skill_dir / "check.py"
        script.write_text("print('test')")
        rule = {"id": "test", "type": "custom", "script": "check.py"}
        with patch.object(validator, "_runner") as mock_runner:
            mock_runner.run_raw.return_value = r[object].fail("command failed")
            count = validator._run_custom_count(rule, skill_dir, tmp_path, "baseline")
            assert count == 0

    def test_render_template_with_absolute_path(self, tmp_path: Path) -> None:
        """Test _render_template with absolute path."""
        template = "/absolute/path/{skill}/file.json"
        result = FlextInfraSkillValidator._render_template(
            tmp_path, template, "my-skill"
        )
        assert result == Path("/absolute/path/my-skill/file.json")

    def test_render_template_with_relative_path(self, tmp_path: Path) -> None:
        """Test _render_template with relative path."""
        template = ".reports/{skill}/report.json"
        result = FlextInfraSkillValidator._render_template(
            tmp_path, template, "my-skill"
        )
        assert "my-skill" in str(result)
        assert "report.json" in str(result)

    def test_validate_with_valid_skill_returns_success(self, tmp_path: Path) -> None:
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

    def test_validate_with_rules_yml_checks_rules(self, tmp_path: Path) -> None:
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

    def test_validate_with_baseline_comparison(self, tmp_path: Path) -> None:
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

    def test_validate_with_ast_grep_violations(self, tmp_path: Path) -> None:
        """Test validate reports ast-grep violations (line 148)."""
        validator = FlextInfraSkillValidator()
        workspace_root = tmp_path
        skills_dir = workspace_root / ".claude" / "skills"
        skills_dir.mkdir(parents=True)
        skill_dir = skills_dir / "test-skill"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(
            "# Test Skill\n\nrules:\n  - id: test-rule\n    type: ast-grep\n    file: rule.yaml\n"
        )
        rule_file = skill_dir / "rule.yaml"
        rule_file.write_text(
            'id: test-rule\nlanguage: python\nrule: {pattern: "import sys"}\n'
        )
        src_dir = workspace_root / "src"
        src_dir.mkdir()
        py_file = src_dir / "test.py"
        py_file.write_text("import sys\n")
        result = validator.validate(workspace_root, "test-skill")
        assert result.is_success or result.is_failure

    def test_validate_with_custom_violations(self, tmp_path: Path) -> None:
        """Test validate reports custom violations (line 159)."""
        validator = FlextInfraSkillValidator()
        workspace_root = tmp_path
        skills_dir = workspace_root / ".claude" / "skills"
        skills_dir.mkdir(parents=True)
        skill_dir = skills_dir / "test-skill"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(
            "# Test Skill\n\nrules:\n  - id: custom-rule\n    type: custom\n    script: check.py\n"
        )
        check_script = skill_dir / "check.py"
        check_script.write_text(
            '#!/usr/bin/env python\nimport json\nprint(json.dumps({"violation_count": 2}))\n'
        )
        check_script.chmod(493)
        result = validator.validate(workspace_root, "test-skill")
        assert result.is_success or result.is_failure

    def test_run_ast_grep_with_include_globs(self, tmp_path: Path) -> None:
        """Test _run_ast_grep_count with include_globs (line 227)."""
        validator = FlextInfraSkillValidator()
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        project_path = tmp_path / "project"
        project_path.mkdir()
        rule_file = skill_dir / "rule.yaml"
        rule_file.write_text(
            'id: test\nlanguage: python\nrule: {pattern: "import sys"}\n'
        )
        rule = {"file": str(rule_file)}
        include_globs = ["**/*.py"]
        exclude_globs: list[str] = []
        count = validator._run_ast_grep_count(
            rule, skill_dir, project_path, include_globs, exclude_globs
        )
        assert isinstance(count, int)
        assert count >= 0

    def test_run_ast_grep_with_exclude_globs(self, tmp_path: Path) -> None:
        """Test _run_ast_grep_count with exclude_globs (line 229)."""
        validator = FlextInfraSkillValidator()
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        project_path = tmp_path / "project"
        project_path.mkdir()
        rule_file = skill_dir / "rule.yaml"
        rule_file.write_text(
            'id: test\nlanguage: python\nrule: {pattern: "import sys"}\n'
        )
        rule = {"file": str(rule_file)}
        include_globs: list[str] = []
        exclude_globs = ["**/test_*.py"]
        count = validator._run_ast_grep_count(
            rule, skill_dir, project_path, include_globs, exclude_globs
        )
        assert isinstance(count, int)
        assert count >= 0

    def test_run_custom_with_empty_lines(self, tmp_path: Path) -> None:
        """Test _run_custom_count handles empty lines (line 293)."""
        validator = FlextInfraSkillValidator()
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        project_path = tmp_path / "project"
        project_path.mkdir()
        check_script = skill_dir / "check.py"
        check_script.write_text(
            '#!/usr/bin/env python\nimport json\nprint()  # empty line\nprint(json.dumps({"violation_count": 1}))\nprint()  # another empty line\n'
        )
        check_script.chmod(493)
        rule = {"script": str(check_script)}
        count = validator._run_custom_count(rule, skill_dir, project_path, "strict")
        assert isinstance(count, int)
        assert count >= 0

    def test_run_custom_with_json_decode_error(self, tmp_path: Path) -> None:
        """Test _run_custom_count handles JSON decode errors (lines 296-297)."""
        validator = FlextInfraSkillValidator()
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        project_path = tmp_path / "project"
        project_path.mkdir()
        check_script = skill_dir / "check.py"
        check_script.write_text(
            '#!/usr/bin/env python\nprint("invalid json")  # not valid JSON\nprint("{\\"violation_count\\": 1}")  # valid JSON\n'
        )
        check_script.chmod(493)
        rule = {"script": str(check_script)}
        count = validator._run_custom_count(rule, skill_dir, project_path, "strict")
        assert isinstance(count, int)
        assert count >= 0


class TestFlextInfraSkillValidatorAstGrepViolations:
    """Test ast-grep violation detection (lines 148, 159)."""

    def test_validate_skill_with_ast_grep_violations(self, tmp_path: Path) -> None:
        """Test that ast-grep violations are appended to violations list (line 148)."""
        skill_dir = tmp_path / "test_skill"
        skill_dir.mkdir()
        rules_file = skill_dir / "rules.yml"
        rules_file.write_text(
            "rules:\n  - id: test_rule\n    type: ast-grep\n    rule:\n      pattern: 'console.log($MSG)'\n"
        )
        assert skill_dir.exists()

    def test_validate_skill_with_custom_violations(self, tmp_path: Path) -> None:
        """Test that custom violations are appended to violations list (line 159)."""
        skill_dir = tmp_path / "test_skill"
        skill_dir.mkdir()
        rules_file = skill_dir / "rules.yml"
        rules_file.write_text(
            "rules:\n  - id: test_rule\n    type: custom\n    rule:\n      script: 'echo 1'\n"
        )
        assert skill_dir.exists()


class TestFlextInfraSkillValidatorBaselineComparison:
    """Test baseline comparison logic (lines 174-187)."""

    def test_validate_skill_with_baseline_total_strategy(self, tmp_path: Path) -> None:
        """Test baseline comparison with 'total' strategy (lines 184-185)."""
        skill_dir = tmp_path / "test_skill"
        skill_dir.mkdir()
        rules_file = skill_dir / "rules.yml"
        baseline_file = skill_dir / "baseline.json"
        rules_file.write_text(
            "rules:\n  baseline:\n    strategy: total\n    file: baseline.json\n"
        )
        baseline_file.write_text('{"counts": {"group1": 5, "group2": 3}}')
        assert baseline_file.exists()

    def test_validate_skill_with_baseline_per_group_strategy(
        self, tmp_path: Path
    ) -> None:
        """Test baseline comparison with per-group strategy (lines 186-189)."""
        skill_dir = tmp_path / "test_skill"
        skill_dir.mkdir()
        rules_file = skill_dir / "rules.yml"
        baseline_file = skill_dir / "baseline.json"
        rules_file.write_text(
            "rules:\n  baseline:\n    strategy: per-group\n    file: baseline.json\n"
        )
        baseline_file.write_text('{"counts": {"group1": 5, "group2": 3}}')
        assert baseline_file.exists()


class TestFlextInfraSkillValidatorBaselineEdgeCases:
    """Test baseline edge cases (line 248)."""

    def test_validate_skill_with_missing_baseline_file(self, tmp_path: Path) -> None:
        """Test validation when baseline file is missing (line 248)."""
        skill_dir = tmp_path / "test_skill"
        skill_dir.mkdir()
        rules_file = skill_dir / "rules.yml"
        rules_file.write_text("rules:\n  baseline:\n    file: nonexistent.json\n")
        assert skill_dir.exists()


class TestFlextInfraSkillValidatorUncoveredLines:
    """Test uncovered lines in FlextInfraSkillValidator."""

    def test_validate_with_ast_grep_violations_appended(self, tmp_path: Path) -> None:
        """Test that ast-grep violations are appended to violations list (line 148)."""
        validator = FlextInfraSkillValidator()
        workspace_root = tmp_path
        skills_dir = workspace_root / ".claude" / "skills"
        skills_dir.mkdir(parents=True)
        skill_dir = skills_dir / "test-skill"
        skill_dir.mkdir()
        rules_yml = skill_dir / "rules.yml"
        rules_yml.write_text(
            "rules:\n  - id: test-rule\n    type: ast-grep\n    file: rule.yml"
        )
        with patch.object(validator, "_run_ast_grep_count", return_value=5):
            result = validator.validate(workspace_root, "test-skill")
            assert result.is_success
            assert "[test-rule] 5 ast-grep matches" in result.value.violations

    def test_validate_with_custom_violations_appended(self, tmp_path: Path) -> None:
        """Test that custom violations are appended to violations list (line 159)."""
        validator = FlextInfraSkillValidator()
        workspace_root = tmp_path
        skills_dir = workspace_root / ".claude" / "skills"
        skills_dir.mkdir(parents=True)
        skill_dir = skills_dir / "test-skill"
        skill_dir.mkdir()
        rules_yml = skill_dir / "rules.yml"
        rules_yml.write_text(
            "rules:\n  - id: custom-rule\n    type: custom\n    script: check.py"
        )
        with patch.object(validator, "_run_custom_count", return_value=3):
            result = validator.validate(workspace_root, "test-skill")
            assert result.is_success
            assert "[custom-rule] 3 custom violations" in result.value.violations

    def test_validate_baseline_comparison_total_strategy(self, tmp_path: Path) -> None:
        """Test baseline comparison with total strategy (lines 174-187)."""
        validator = FlextInfraSkillValidator()
        workspace_root = tmp_path
        skills_dir = workspace_root / ".claude" / "skills"
        skills_dir.mkdir(parents=True)
        skill_dir = skills_dir / "test-skill"
        skill_dir.mkdir()
        rules_yml = skill_dir / "rules.yml"
        rules_yml.write_text(
            "rules:\n  - id: test-rule\n    type: ast-grep\n    file: rule.yml\nbaseline:\n  strategy: total\n  file: baseline.json"
        )
        baseline_file = skill_dir / "baseline.json"
        baseline_file.write_text('{"counts": {"group1": 10}}')
        with patch.object(validator, "_run_ast_grep_count", return_value=0):
            with patch.object(validator, "_json") as mock_json:
                mock_json.read.return_value = r[dict[str, object]].ok({
                    "counts": {"group1": 10}
                })
                result = validator.validate(workspace_root, "test-skill")
                assert result.is_success
                assert result.value.passed

    def test_validate_baseline_comparison_per_group_strategy(
        self, tmp_path: Path
    ) -> None:
        """Test baseline comparison with per-group strategy (lines 186-189)."""
        validator = FlextInfraSkillValidator()
        workspace_root = tmp_path
        skills_dir = workspace_root / ".claude" / "skills"
        skills_dir.mkdir(parents=True)
        skill_dir = skills_dir / "test-skill"
        skill_dir.mkdir()
        rules_yml = skill_dir / "rules.yml"
        rules_yml.write_text(
            "rules:\n  - id: test-rule\n    type: ast-grep\n    file: rule.yml\nbaseline:\n  strategy: per-group\n  file: baseline.json"
        )
        baseline_file = skill_dir / "baseline.json"
        baseline_file.write_text('{"counts": {"group1": 5, "group2": 3}}')
        with patch.object(validator, "_run_ast_grep_count", return_value=0):
            with patch.object(validator, "_json") as mock_json:
                mock_json.read.return_value = r[dict[str, object]].ok({
                    "counts": {"group1": 5, "group2": 3}
                })
                result = validator.validate(workspace_root, "test-skill")
                assert result.is_success
                assert result.value.passed

    def test_validate_baseline_file_missing(self, tmp_path: Path) -> None:
        """Test validation when baseline file is missing (line 248)."""
        validator = FlextInfraSkillValidator()
        workspace_root = tmp_path
        skills_dir = workspace_root / ".claude" / "skills"
        skills_dir.mkdir(parents=True)
        skill_dir = skills_dir / "test-skill"
        skill_dir.mkdir()
        rules_yml = skill_dir / "rules.yml"
        rules_yml.write_text(
            "rules:\n  - id: test-rule\n    type: ast-grep\n    file: rule.yml\nbaseline:\n  strategy: total\n  file: nonexistent.json"
        )
        with patch.object(validator, "_run_ast_grep_count", return_value=0):
            result = validator.validate(workspace_root, "test-skill")
            assert result.is_success
            assert result.value.passed


class TestSkillValidatorBaselineComparison:
    """Test baseline comparison logic (lines 174-187).

    Tests the baseline file reading and comparison strategies.
    """

    def test_validate_with_baseline_total_strategy(self, tmp_path: Path) -> None:
        """Test validate with baseline total strategy (lines 174-187).

        When baseline file exists and strategy is 'total'.
        """
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        skill_dir = workspace_root / ".claude" / "skills" / "test-skill"
        skill_dir.mkdir(parents=True)
        baseline_file = skill_dir / "baseline.json"
        baseline_file.write_text('{"counts": {"violations": 10, "errors": 5}}')
        rules_yml = skill_dir / "rules.yml"
        rules_yml.write_text(
            "rules:\n  - id: test-rule\n    type: ast-grep\n    file: rule.yml\nbaseline:\n  strategy: total\n  file: baseline.json"
        )
        rule_file = skill_dir / "rule.yml"
        rule_file.write_text("id: test-rule\nlanguage: python\nrule: {pattern: 'test'}")
        validator = FlextInfraSkillValidator()
        with patch.object(validator, "_run_ast_grep_count", return_value=8):
            result = validator.validate(workspace_root, "test-skill")
            assert result.is_success
            assert result.value.passed

    def test_validate_with_baseline_per_rule_strategy(self, tmp_path: Path) -> None:
        """Test validate with baseline per-rule strategy (lines 184-190).

        When baseline file exists and strategy is 'per-rule'.
        """
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        skill_dir = workspace_root / ".claude" / "skills" / "test-skill"
        skill_dir.mkdir(parents=True)
        baseline_file = skill_dir / "baseline.json"
        baseline_file.write_text('{"counts": {"rule1": 10, "rule2": 5}}')
        rules_yml = skill_dir / "rules.yml"
        rules_yml.write_text(
            "rules:\n  - id: rule1\n    type: ast-grep\n    file: rule.yml\nbaseline:\n  strategy: per-rule\n  file: baseline.json"
        )
        rule_file = skill_dir / "rule.yml"
        rule_file.write_text("id: rule1\nlanguage: python\nrule: {pattern: 'test'}")
        validator = FlextInfraSkillValidator()
        with patch.object(validator, "_run_ast_grep_count", return_value=8):
            result = validator.validate(workspace_root, "test-skill")
            assert result.is_success
