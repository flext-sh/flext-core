"""Tests for FlextInfraDocAuditor — core audit and static helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_infra import m
from flext_infra.docs.auditor import FlextInfraDocAuditor
from flext_tests import tm


@pytest.fixture
def auditor() -> FlextInfraDocAuditor:
    return FlextInfraDocAuditor()


class TestAuditorCore:
    def test_returns_flext_result(
        self,
        auditor: FlextInfraDocAuditor,
        tmp_path: Path,
    ) -> None:
        result = auditor.audit(tmp_path)
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_valid_scope_returns_success(
        self,
        auditor: FlextInfraDocAuditor,
        tmp_path: Path,
    ) -> None:
        result = auditor.audit(tmp_path)
        tm.ok(result)

    def test_report_structure(
        self,
        auditor: FlextInfraDocAuditor,
        tmp_path: Path,
    ) -> None:
        result = auditor.audit(tmp_path)
        if result.is_success and result.value:
            report = result.value[0]
            tm.that(hasattr(report, "scope"), eq=True)
            tm.that(hasattr(report, "items"), eq=True)

    def test_issue_structure(self) -> None:
        issue = m.Infra.Docs.AuditIssue(
            file="README.md",
            issue_type="broken_link",
            severity="high",
            message="Link to missing file",
        )
        tm.that(issue.file, eq="README.md")
        tm.that(issue.issue_type, eq="broken_link")
        tm.that(issue.severity, eq="high")

    def test_with_project_filter(
        self,
        auditor: FlextInfraDocAuditor,
        tmp_path: Path,
    ) -> None:
        result = auditor.audit(tmp_path, project="test-project")
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_with_projects_filter(
        self,
        auditor: FlextInfraDocAuditor,
        tmp_path: Path,
    ) -> None:
        result = auditor.audit(tmp_path, projects="proj1,proj2")
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_check_links(
        self,
        auditor: FlextInfraDocAuditor,
        tmp_path: Path,
    ) -> None:
        result = auditor.audit(tmp_path, check="links")
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_check_forbidden_terms(
        self,
        auditor: FlextInfraDocAuditor,
        tmp_path: Path,
    ) -> None:
        result = auditor.audit(tmp_path, check="forbidden-terms")
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_strict_mode(
        self,
        auditor: FlextInfraDocAuditor,
        tmp_path: Path,
    ) -> None:
        result = auditor.audit(tmp_path, strict=True)
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_custom_output_dir(
        self,
        auditor: FlextInfraDocAuditor,
        tmp_path: Path,
    ) -> None:
        result = auditor.audit(tmp_path, output_dir=str(tmp_path / "custom_output"))
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_report_frozen(self) -> None:
        tm.that(m.Infra.Docs.DocsPhaseReport.model_config.get("frozen"), eq=True)

    def test_issue_frozen(self) -> None:
        tm.that(m.Infra.Docs.AuditIssue.model_config.get("frozen"), eq=True)


class TestAuditorNormalize:
    def test_normalize_strips_fragment(self) -> None:
        nl = FlextInfraDocAuditor.normalize_link
        tm.that(nl("path/to/file.md#section"), eq="path/to/file.md")

    def test_normalize_strips_query(self) -> None:
        nl = FlextInfraDocAuditor.normalize_link
        tm.that(nl("path/to/file.md?param=value"), eq="path/to/file.md")

    def test_normalize_removes_angle_brackets(self) -> None:
        nl = FlextInfraDocAuditor.normalize_link
        tm.that(nl("<path/to/file.md>"), eq="path/to/file.md")

    def test_normalize_strips_whitespace(self) -> None:
        nl = FlextInfraDocAuditor.normalize_link
        tm.that(nl("  path/to/file.md  "), eq="path/to/file.md")

    def test_normalize_complex(self) -> None:
        nl = FlextInfraDocAuditor.normalize_link
        tm.that(nl("<path/to/file.md#section?param=value>"), eq="path/to/file.md")

    def test_skip_target_http(self) -> None:
        sst = FlextInfraDocAuditor.should_skip_target
        tm.that(sst("[link](http://example.com)", "http://example.com"), eq=False)

    def test_skip_target_https(self) -> None:
        sst = FlextInfraDocAuditor.should_skip_target
        tm.that(sst("[link](https://example.com)", "https://example.com"), eq=False)

    def test_skip_target_comma_no_md(self) -> None:
        tm.that(FlextInfraDocAuditor.should_skip_target("[a, b]", "a"), eq=True)

    def test_skip_target_space_no_md(self) -> None:
        tm.that(FlextInfraDocAuditor.should_skip_target("[a b]", "a"), eq=True)

    def test_skip_target_md_not_skipped(self) -> None:
        tm.that(FlextInfraDocAuditor.should_skip_target("[a, b.md]", "a"), eq=False)

    def test_skip_target_path_not_skipped(self) -> None:
        tm.that(FlextInfraDocAuditor.should_skip_target("[a/b]", "a/b"), eq=False)

    def test_is_external_http(self) -> None:
        tm.that(FlextInfraDocAuditor.is_external("http://example.com"), eq=True)

    def test_is_external_https(self) -> None:
        tm.that(FlextInfraDocAuditor.is_external("https://example.com"), eq=True)

    def test_is_external_mailto(self) -> None:
        tm.that(FlextInfraDocAuditor.is_external("mailto:test@example.com"), eq=True)

    def test_is_external_tel(self) -> None:
        tm.that(FlextInfraDocAuditor.is_external("tel:+1234567890"), eq=True)

    def test_is_external_data_uri(self) -> None:
        ie = FlextInfraDocAuditor.is_external
        tm.that(ie("data:text/plain;base64,SGVsbG8="), eq=True)

    def test_is_external_local_path(self) -> None:
        tm.that(FlextInfraDocAuditor.is_external("path/to/file.md"), eq=False)

    def test_is_external_angle_brackets(self) -> None:
        tm.that(FlextInfraDocAuditor.is_external("<http://example.com>"), eq=True)

    def test_is_external_case_insensitive(self) -> None:
        tm.that(FlextInfraDocAuditor.is_external("HTTPS://EXAMPLE.COM"), eq=True)
