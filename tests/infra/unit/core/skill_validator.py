"""Tests for FlextInfraSkillValidator — core validation and helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_infra.core.skill_validator import FlextInfraSkillValidator
from flext_tests import tm

_safe_load_yaml = FlextInfraSkillValidator._safe_load_yaml
_normalize_string_list = FlextInfraSkillValidator._normalize_string_list


class TestSafeLoadYaml:
    """Test _safe_load_yaml helper function."""

    def test_valid_yaml(self, tmp_path: Path) -> None:
        """Valid YAML file loads correctly."""
        f = tmp_path / "test.yml"
        f.write_text("key: value\nlist:\n  - item1\n  - item2")
        result = _safe_load_yaml(f)
        assert result.get("key") == "value"
        list_value = result.get("list")
        assert isinstance(list_value, list)
        assert list_value == ["item1", "item2"]

    def test_empty_and_null(self, tmp_path: Path) -> None:
        """Empty/null YAML returns empty dict."""
        (tmp_path / "empty.yml").write_text("")
        assert dict(_safe_load_yaml(tmp_path / "empty.yml")) == {}
        (tmp_path / "null.yml").write_text("null")
        assert dict(_safe_load_yaml(tmp_path / "null.yml")) == {}

    def test_non_dict_raises_type_error(self, tmp_path: Path) -> None:
        """Non-dict YAML raises TypeError."""
        (tmp_path / "list.yml").write_text("- item1\n- item2")
        with pytest.raises(TypeError, match="rules\\.yml must be a mapping"):
            _safe_load_yaml(tmp_path / "list.yml")
        (tmp_path / "str.yml").write_text("just a string")
        with pytest.raises(TypeError, match="rules\\.yml must be a mapping"):
            _safe_load_yaml(tmp_path / "str.yml")


class TestNormalizeStringList:
    """Test _normalize_string_list helper function."""

    def test_none_returns_empty(self) -> None:
        """None returns empty list."""
        tm.that(_normalize_string_list(None, "test"), eq=[])

    def test_valid_list(self) -> None:
        """Valid string list passes through."""
        tm.that(_normalize_string_list(["a", "b", "c"], "test"), eq=["a", "b", "c"])

    def test_invalid_input_raises(self) -> None:
        """Non-string items and non-list values raise TypeError."""
        with pytest.raises(TypeError, match="test_field must be list\\[str\\]"):
            _normalize_string_list(["a", 123, "c"], "test_field")
        with pytest.raises(TypeError, match="test_field must be list\\[str\\]"):
            _normalize_string_list("not a list", "test_field")
        with pytest.raises(TypeError, match="test_field must be list\\[str\\]"):
            _normalize_string_list({"key": "value"}, "test_field")


class TestSkillValidatorCore:
    """Core validation tests for FlextInfraSkillValidator."""

    def test_validate_missing_rules_yml(self, tmp_path: Path) -> None:
        """Missing rules.yml returns not-passed report."""
        validator = FlextInfraSkillValidator()
        skills = tmp_path / ".claude" / "skills" / "test-skill"
        skills.mkdir(parents=True)
        report = tm.ok(validator.validate(tmp_path, "test-skill"))
        tm.that(report.passed, eq=False)
        tm.that(report.summary, contains="no rules.yml")

    def test_validate_invalid_scan_targets(self, tmp_path: Path) -> None:
        """Non-dict scan_targets returns failure."""
        validator = FlextInfraSkillValidator()
        skill = tmp_path / ".claude" / "skills" / "test-skill"
        skill.mkdir(parents=True)
        (skill / "rules.yml").write_text("scan_targets: [item1, item2]")
        tm.fail(validator.validate(tmp_path, "test-skill"), has="scan_targets")

    def test_validate_invalid_rules_not_list(self, tmp_path: Path) -> None:
        """Non-list rules returns failure."""
        validator = FlextInfraSkillValidator()
        skill = tmp_path / ".claude" / "skills" / "test-skill"
        skill.mkdir(parents=True)
        (skill / "rules.yml").write_text("rules: {not: a_list}")
        tm.fail(validator.validate(tmp_path, "test-skill"), has="rules must be a list")

    def test_validate_non_dict_rule_skipped(self, tmp_path: Path) -> None:
        """Non-dict rule objects are skipped."""
        validator = FlextInfraSkillValidator()
        skill = tmp_path / ".claude" / "skills" / "test-skill"
        skill.mkdir(parents=True)
        (skill / "rules.yml").write_text("rules:\n  - not_a_dict\n  - another_string")
        report = tm.ok(validator.validate(tmp_path, "test-skill"))
        tm.that(report.passed, eq=True)

    def test_validate_exception_returns_failure(self, tmp_path: Path) -> None:
        """Invalid YAML content returns failure."""
        validator = FlextInfraSkillValidator()
        skill = tmp_path / ".claude" / "skills" / "test-skill"
        skill.mkdir(parents=True)
        (skill / "rules.yml").write_text("just a plain string")
        tm.fail(
            validator.validate(tmp_path, "test-skill"), has="skill validation failed"
        )


class TestSkillValidatorRenderTemplate:
    """Tests for _render_template static method."""

    def test_absolute_path(self, tmp_path: Path) -> None:
        """Absolute path template resolves correctly."""
        result = FlextInfraSkillValidator._render_template(
            tmp_path,
            "/absolute/path/{skill}/file.json",
            "my-skill",
        )
        tm.that(str(result), eq="/absolute/path/my-skill/file.json")

    def test_relative_path(self, tmp_path: Path) -> None:
        """Relative path template resolves with skill name."""
        result = FlextInfraSkillValidator._render_template(
            tmp_path,
            ".reports/{skill}/report.json",
            "my-skill",
        )
        tm.that("my-skill" in str(result), eq=True)
        tm.that("report.json" in str(result), eq=True)


class TestSkillValidatorAstGrepCount:
    """Tests for _run_ast_grep_count and _run_custom_count methods."""

    def test_empty_or_missing_rule_file(self, tmp_path: Path) -> None:
        """Empty/nonexistent rule file returns 0."""
        v = FlextInfraSkillValidator()
        skill = tmp_path / "skill"
        skill.mkdir()
        empty = {"id": "t", "type": "ast-grep", "file": ""}
        missing = {"id": "t", "type": "ast-grep", "file": "nonexistent.yml"}
        tm.that(v._run_ast_grep_count(empty, skill, tmp_path, [], []), eq=0)
        tm.that(v._run_ast_grep_count(missing, skill, tmp_path, [], []), eq=0)

    def test_with_include_globs(self, tmp_path: Path) -> None:
        """Include globs are passed to runner."""
        v = FlextInfraSkillValidator()
        skill = tmp_path / "skill"
        skill.mkdir()
        rule_file = skill / "rule.yaml"
        rule_file.write_text("id: test\nlanguage: python\nrule: {pattern: 'test'}")
        count = v._run_ast_grep_count(
            {"file": str(rule_file)},
            skill,
            tmp_path,
            ["**/*.py"],
            [],
        )
        assert isinstance(count, int)

    def test_custom_count_empty_or_missing(self, tmp_path: Path) -> None:
        """Empty/nonexistent custom script returns 0."""
        v = FlextInfraSkillValidator()
        skill = tmp_path / "skill"
        skill.mkdir()
        empty = {"id": "t", "type": "custom", "script": ""}
        missing = {"id": "t", "type": "custom", "script": "nonexistent.py"}
        tm.that(v._run_custom_count(empty, skill, tmp_path, "baseline"), eq=0)
        tm.that(v._run_custom_count(missing, skill, tmp_path, "baseline"), eq=0)


__all__: list[str] = []
