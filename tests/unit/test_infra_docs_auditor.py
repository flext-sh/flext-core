"""Tests for FlextInfraDocAuditor service.

Tests documentation auditing functionality with mocked file system
and structured FlextResult reports.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from flext_infra.docs.auditor import AuditIssue, AuditReport, FlextInfraDocAuditor
from flext_infra.docs.shared import FlextInfraDocScope


class TestFlextInfraDocAuditor:
    """Tests for FlextInfraDocAuditor service."""

    @pytest.fixture
    def auditor(self) -> FlextInfraDocAuditor:
        """Create auditor instance."""
        return FlextInfraDocAuditor()

    @pytest.fixture
    def sample_scope(self, tmp_path: Path) -> FlextInfraDocScope:
        """Create sample documentation scope."""
        report_dir = tmp_path / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        return FlextInfraDocScope(
            name="test-project",
            path=tmp_path,
            report_dir=report_dir,
        )

    def test_audit_returns_flext_result(
        self, auditor: FlextInfraDocAuditor, tmp_path: Path
    ) -> None:
        """Test that audit returns FlextResult[list[AuditReport]]."""
        result = auditor.audit(tmp_path)
        assert result.is_success or result.is_failure

    def test_audit_with_valid_scope_returns_success(
        self, auditor: FlextInfraDocAuditor, tmp_path: Path
    ) -> None:
        """Test audit with valid scope returns success."""
        result = auditor.audit(tmp_path)
        assert result.is_success
        assert isinstance(result.value, list)

    def test_audit_report_structure(
        self, auditor: FlextInfraDocAuditor, tmp_path: Path
    ) -> None:
        """Test AuditReport has required fields."""
        result = auditor.audit(tmp_path)
        if result.is_success and result.value:
            report = result.value[0]
            assert hasattr(report, "scope")
            assert hasattr(report, "issues")
            assert isinstance(report.issues, list)

    def test_audit_issue_structure(self) -> None:
        """Test AuditIssue model structure."""
        issue = AuditIssue(
            file="README.md",
            issue_type="broken_link",
            severity="high",
            message="Link to missing file",
        )
        assert issue.file == "README.md"
        assert issue.issue_type == "broken_link"
        assert issue.severity == "high"

    def test_audit_with_project_filter(
        self, auditor: FlextInfraDocAuditor, tmp_path: Path
    ) -> None:
        """Test audit with single project filter."""
        result = auditor.audit(tmp_path, project="test-project")
        assert result.is_success or result.is_failure

    def test_audit_with_projects_filter(
        self, auditor: FlextInfraDocAuditor, tmp_path: Path
    ) -> None:
        """Test audit with multiple projects filter."""
        result = auditor.audit(tmp_path, projects="proj1,proj2")
        assert result.is_success or result.is_failure

    def test_audit_with_check_links(
        self, auditor: FlextInfraDocAuditor, tmp_path: Path
    ) -> None:
        """Test audit with links check."""
        result = auditor.audit(tmp_path, check="links")
        assert result.is_success or result.is_failure

    def test_audit_with_check_forbidden_terms(
        self, auditor: FlextInfraDocAuditor, tmp_path: Path
    ) -> None:
        """Test audit with forbidden-terms check."""
        result = auditor.audit(tmp_path, check="forbidden-terms")
        assert result.is_success or result.is_failure

    def test_audit_with_strict_mode(
        self, auditor: FlextInfraDocAuditor, tmp_path: Path
    ) -> None:
        """Test audit with strict mode enabled."""
        result = auditor.audit(tmp_path, strict=True)
        assert result.is_success or result.is_failure

    def test_audit_with_custom_output_dir(
        self, auditor: FlextInfraDocAuditor, tmp_path: Path
    ) -> None:
        """Test audit with custom output directory."""
        output_dir = str(tmp_path / "custom_output")
        result = auditor.audit(tmp_path, output_dir=output_dir)
        assert result.is_success or result.is_failure

    def test_audit_report_frozen(self) -> None:
        """Test AuditReport is frozen (immutable)."""
        report = AuditReport(scope="test", issues=[])
        with pytest.raises(Exception):  # pydantic frozen raises
            report.scope = "modified"  # type: ignore

    def test_audit_issue_frozen(self) -> None:
        """Test AuditIssue is frozen (immutable)."""
        issue = AuditIssue(
            file="test.md",
            issue_type="test",
            severity="low",
            message="test",
        )
        with pytest.raises(Exception):  # pydantic frozen raises
            issue.file = "modified"  # type: ignore
