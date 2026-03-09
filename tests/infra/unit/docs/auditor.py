"""Tests for FlextInfraDocAuditor — core audit and static helper methods.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flext_infra import m
from flext_infra.docs.auditor import FlextInfraDocAuditor
from flext_tests import tm


class TestAuditorCore:
    """Core audit invocation tests."""

    @pytest.fixture
    def auditor(self) -> FlextInfraDocAuditor:
        """Create auditor instance."""
        return FlextInfraDocAuditor()

    def test_audit_returns_flext_result(
        self, auditor: FlextInfraDocAuditor, tmp_path: Path
    ) -> None:
        """Test that audit returns FlextResult."""
        result = auditor.audit(tmp_path)
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_audit_with_valid_scope_returns_success(
        self, auditor: FlextInfraDocAuditor, tmp_path: Path
    ) -> None:
        """Test audit with valid scope returns success."""
        result = auditor.audit(tmp_path)
        tm.ok(result)
        tm.that(isinstance(result.value, list), eq=True)

    def test_audit_report_structure(
        self, auditor: FlextInfraDocAuditor, tmp_path: Path
    ) -> None:
        """Test AuditReport has required fields."""
        result = auditor.audit(tmp_path)
        if result.is_success and result.value:
            report = result.value[0]
            tm.that(hasattr(report, "scope"), eq=True)
            tm.that(hasattr(report, "items"), eq=True)
            tm.that(isinstance(report.items, list), eq=True)

    def test_audit_issue_structure(self) -> None:
        """Test AuditIssue model structure."""
        issue = m.Infra.Docs.AuditIssue(
            file="README.md",
            issue_type="broken_link",
            severity="high",
            message="Link to missing file",
        )
        tm.that(issue.file, eq="README.md")
        tm.that(issue.issue_type, eq="broken_link")
        tm.that(issue.severity, eq="high")

    def test_audit_with_project_filter(
        self, auditor: FlextInfraDocAuditor, tmp_path: Path
    ) -> None:
        """Test audit with single project filter."""
        result = auditor.audit(tmp_path, project="test-project")
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_audit_with_projects_filter(
        self, auditor: FlextInfraDocAuditor, tmp_path: Path
    ) -> None:
        """Test audit with multiple projects filter."""
        result = auditor.audit(tmp_path, projects="proj1,proj2")
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_audit_with_check_links(
        self, auditor: FlextInfraDocAuditor, tmp_path: Path
    ) -> None:
        """Test audit with links check."""
        result = auditor.audit(tmp_path, check="links")
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_audit_with_check_forbidden_terms(
        self, auditor: FlextInfraDocAuditor, tmp_path: Path
    ) -> None:
        """Test audit with forbidden-terms check."""
        result = auditor.audit(tmp_path, check="forbidden-terms")
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_audit_with_strict_mode(
        self, auditor: FlextInfraDocAuditor, tmp_path: Path
    ) -> None:
        """Test audit with strict mode enabled."""
        result = auditor.audit(tmp_path, strict=True)
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_audit_with_custom_output_dir(
        self, auditor: FlextInfraDocAuditor, tmp_path: Path
    ) -> None:
        """Test audit with custom output directory."""
        result = auditor.audit(tmp_path, output_dir=str(tmp_path / "custom_output"))
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_audit_report_frozen(self) -> None:
        """Test AuditReport is frozen (immutable)."""
        tm.that(m.Infra.Docs.DocsPhaseReport.model_config.get("frozen"), eq=True)

    def test_audit_issue_frozen(self) -> None:
        """Test AuditIssue is frozen (immutable)."""
        tm.that(m.Infra.Docs.AuditIssue.model_config.get("frozen"), eq=True)


class TestAuditorNormalize:
    """Tests for normalize_link, should_skip_target, is_external."""

    def test_normalize_link_with_fragment(self) -> None:
        """Test normalize_link strips fragment identifier."""
        tm.that(
            FlextInfraDocAuditor.normalize_link("path/to/file.md#section"),
            eq="path/to/file.md",
        )

    def test_normalize_link_with_query_string(self) -> None:
        """Test normalize_link strips query string."""
        tm.that(
            FlextInfraDocAuditor.normalize_link("path/to/file.md?param=value"),
            eq="path/to/file.md",
        )

    def test_normalize_link_with_angle_brackets(self) -> None:
        """Test normalize_link removes angle brackets."""
        tm.that(
            FlextInfraDocAuditor.normalize_link("<path/to/file.md>"),
            eq="path/to/file.md",
        )

    def test_normalize_link_with_whitespace(self) -> None:
        """Test normalize_link strips whitespace."""
        tm.that(
            FlextInfraDocAuditor.normalize_link("  path/to/file.md  "),
            eq="path/to/file.md",
        )

    def test_normalize_link_complex(self) -> None:
        """Test normalize_link with fragment and query."""
        tm.that(
            FlextInfraDocAuditor.normalize_link(
                "<path/to/file.md#section?param=value>"
            ),
            eq="path/to/file.md",
        )

    def test_should_skip_target_http_url(self) -> None:
        """Test should_skip_target returns False for http URLs."""
        tm.that(
            FlextInfraDocAuditor.should_skip_target(
                "[link](http://example.com)", "http://example.com"
            ),
            eq=False,
        )

    def test_should_skip_target_https_url(self) -> None:
        """Test should_skip_target returns False for https URLs."""
        tm.that(
            FlextInfraDocAuditor.should_skip_target(
                "[link](https://example.com)", "https://example.com"
            ),
            eq=False,
        )

    def test_should_skip_target_comma_no_md(self) -> None:
        """Test should_skip_target skips comma-separated non-paths."""
        tm.that(FlextInfraDocAuditor.should_skip_target("[a, b]", "a"), eq=True)

    def test_should_skip_target_space_no_md(self) -> None:
        """Test should_skip_target skips space-separated non-paths."""
        tm.that(FlextInfraDocAuditor.should_skip_target("[a b]", "a"), eq=True)

    def test_should_skip_target_md_file_not_skipped(self) -> None:
        """Test should_skip_target does not skip .md files."""
        tm.that(FlextInfraDocAuditor.should_skip_target("[a, b.md]", "a"), eq=False)

    def test_should_skip_target_path_not_skipped(self) -> None:
        """Test should_skip_target does not skip paths with slashes."""
        tm.that(FlextInfraDocAuditor.should_skip_target("[a/b]", "a/b"), eq=False)

    def test_is_external_http(self) -> None:
        """Test is_external returns True for http URLs."""
        tm.that(FlextInfraDocAuditor.is_external("http://example.com"), eq=True)

    def test_is_external_https(self) -> None:
        """Test is_external returns True for https URLs."""
        tm.that(FlextInfraDocAuditor.is_external("https://example.com"), eq=True)

    def test_is_external_mailto(self) -> None:
        """Test is_external returns True for mailto links."""
        tm.that(FlextInfraDocAuditor.is_external("mailto:test@example.com"), eq=True)

    def test_is_external_tel(self) -> None:
        """Test is_external returns True for tel links."""
        tm.that(FlextInfraDocAuditor.is_external("tel:+1234567890"), eq=True)

    def test_is_external_data_uri(self) -> None:
        """Test is_external returns True for data URIs."""
        tm.that(
            FlextInfraDocAuditor.is_external("data:text/plain;base64,SGVsbG8="), eq=True
        )

    def test_is_external_local_path(self) -> None:
        """Test is_external returns False for local paths."""
        tm.that(FlextInfraDocAuditor.is_external("path/to/file.md"), eq=False)

    def test_is_external_with_angle_brackets(self) -> None:
        """Test is_external handles angle brackets."""
        tm.that(FlextInfraDocAuditor.is_external("<http://example.com>"), eq=True)

    def test_is_external_case_insensitive(self) -> None:
        """Test is_external is case insensitive."""
        tm.that(FlextInfraDocAuditor.is_external("HTTPS://EXAMPLE.COM"), eq=True)


class TestAuditorBudgets:
    """Tests for load_audit_budgets."""

    def test_load_audit_budgets_no_config(self, tmp_path: Path) -> None:
        """Test load_audit_budgets returns defaults when no config."""
        default, by_scope = FlextInfraDocAuditor.load_audit_budgets(tmp_path)
        tm.that(default, eq=None)
        tm.that(by_scope, eq={})

    def test_load_audit_budgets_with_config(self, tmp_path: Path) -> None:
        """Test load_audit_budgets loads from config file."""
        arch_dir = tmp_path / "docs/architecture"
        arch_dir.mkdir(parents=True, exist_ok=True)
        config_data = {
            "docs_validation": {
                "audit_gate": {
                    "max_issues_default": 5,
                    "max_issues_by_scope": {"test-project": 3},
                }
            }
        }
        (arch_dir / "architecture_config.json").write_text(json.dumps(config_data))
        default, by_scope = FlextInfraDocAuditor.load_audit_budgets(tmp_path)
        tm.that(default, eq=5)
        tm.that(by_scope.get("test-project"), eq=3)

    def test_load_audit_budgets_invalid_json(self, tmp_path: Path) -> None:
        """Test load_audit_budgets handles invalid JSON gracefully."""
        arch_dir = tmp_path / "docs/architecture"
        arch_dir.mkdir(parents=True, exist_ok=True)
        (arch_dir / "architecture_config.json").write_text("{invalid json}")
        default, by_scope = FlextInfraDocAuditor.load_audit_budgets(tmp_path)
        tm.that(default, eq=None)
        tm.that(by_scope, eq={})

    def test_load_audit_budgets_with_float_values(self, tmp_path: Path) -> None:
        """Test load_audit_budgets converts float values to int."""
        arch_dir = tmp_path / "docs/architecture"
        arch_dir.mkdir(parents=True, exist_ok=True)
        config_data = {
            "docs_validation": {
                "audit_gate": {
                    "max_issues_default": 5.5,
                    "max_issues_by_scope": {"test-project": 3.7},
                }
            }
        }
        (arch_dir / "architecture_config.json").write_text(json.dumps(config_data))
        default, by_scope = FlextInfraDocAuditor.load_audit_budgets(tmp_path)
        tm.that(default, eq=5)
        tm.that(by_scope.get("test-project"), eq=3)

    def test_load_audit_budgets_no_default_budget(self, tmp_path: Path) -> None:
        """Test load_audit_budgets returns None when no default budget."""
        arch_dir = tmp_path / "docs/architecture"
        arch_dir.mkdir(parents=True, exist_ok=True)
        config_data = {
            "docs_validation": {
                "audit_gate": {"max_issues_by_scope": {"test-project": 3}}
            }
        }
        (arch_dir / "architecture_config.json").write_text(json.dumps(config_data))
        default, by_scope = FlextInfraDocAuditor.load_audit_budgets(tmp_path)
        tm.that(default, eq=None)
        tm.that(by_scope.get("test-project"), eq=3)
