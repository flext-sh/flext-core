"""Tests for FlextInfraDependencyDetectionService."""

from __future__ import annotations

from flext_infra.deps.detection import (
    FlextInfraDependencyDetectionService,
    dm,
)


class TestFlextInfraDependencyDetectionModels:
    """Test FlextInfraDependencyDetectionModels namespace."""

    def test_deptry_issue_groups_creation(self) -> None:
        """Test DeptryIssueGroups model creation."""
        groups = dm.DeptryIssueGroups()
        assert groups.dep001 == []
        assert groups.dep002 == []
        assert groups.dep003 == []
        assert groups.dep004 == []

    def test_deptry_report_creation(self) -> None:
        """Test DeptryReport model creation."""
        report = dm.DeptryReport()
        assert report.missing == []
        assert report.unused == []
        assert report.transitive == []
        assert report.dev_in_runtime == []
        assert report.raw_count == 0

    def test_project_dependency_report_creation(self) -> None:
        """Test ProjectDependencyReport model creation."""
        deptry = dm.DeptryReport()
        report = dm.ProjectDependencyReport(project="test-project", deptry=deptry)
        assert report.project == "test-project"
        assert report.deptry == deptry

    def test_typings_report_creation(self) -> None:
        """Test TypingsReport model creation."""
        report = dm.TypingsReport()
        assert report.required_packages == []
        assert report.hinted == []
        assert report.missing_modules == []
        assert report.current == []
        assert report.to_add == []
        assert report.to_remove == []
        assert report.limits_applied is False
        assert report.python_version is None


class TestFlextInfraDependencyDetectionService:
    """Test FlextInfraDependencyDetectionService."""

    def test_service_initialization(self) -> None:
        """Test service initializes without errors."""
        service = FlextInfraDependencyDetectionService()
        assert service is not None

    def test_default_module_to_types_package_mapping(self) -> None:
        """Test default module to types package mapping exists."""
        service = FlextInfraDependencyDetectionService()
        assert "yaml" in service.DEFAULT_MODULE_TO_TYPES_PACKAGE
        assert service.DEFAULT_MODULE_TO_TYPES_PACKAGE["yaml"] == "types-pyyaml"
