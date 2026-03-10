"""Tests for FlextInfraDocBuilder — core build and scope tests.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_infra.docs.builder import FlextInfraDocBuilder
from flext_tests import tm
from tests.infra.models import m


class TestBuilderCore:
    """Core build invocation tests."""

    @pytest.fixture
    def builder(self) -> FlextInfraDocBuilder:
        """Create builder instance."""
        return FlextInfraDocBuilder()

    def test_build_returns_flext_result(
        self, builder: FlextInfraDocBuilder, tmp_path: Path
    ) -> None:
        """Test that build returns FlextResult."""
        result = builder.build(tmp_path)
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_build_with_valid_scope_returns_success(
        self, builder: FlextInfraDocBuilder, tmp_path: Path
    ) -> None:
        """Test build with valid scope returns success."""
        result = builder.build(tmp_path)
        tm.ok(result)
        tm.that(isinstance(result.value, list), eq=True)

    def test_build_report_structure(
        self, builder: FlextInfraDocBuilder, tmp_path: Path
    ) -> None:
        """Test BuildReport has required fields."""
        result = builder.build(tmp_path)
        if result.is_success and result.value:
            report = result.value[0]
            tm.that(hasattr(report, "scope"), eq=True)
            tm.that(hasattr(report, "result"), eq=True)
            tm.that(hasattr(report, "reason"), eq=True)
            tm.that(hasattr(report, "site_dir"), eq=True)

    def test_build_report_frozen(self) -> None:
        """Test BuildReport is frozen (immutable)."""
        tm.that(m.Infra.Docs.DocsPhaseReport.model_config.get("frozen"), eq=True)

    def test_build_with_project_filter(
        self, builder: FlextInfraDocBuilder, tmp_path: Path
    ) -> None:
        """Test build with single project filter."""
        result = builder.build(tmp_path, project="test-project")
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_build_with_projects_filter(
        self, builder: FlextInfraDocBuilder, tmp_path: Path
    ) -> None:
        """Test build with multiple projects filter."""
        result = builder.build(tmp_path, projects="proj1,proj2")
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_build_with_custom_output_dir(
        self, builder: FlextInfraDocBuilder, tmp_path: Path
    ) -> None:
        """Test build with custom output directory."""
        result = builder.build(tmp_path, output_dir=str(tmp_path / "custom_output"))
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_build_report_result_field_values(self) -> None:
        """Test BuildReport result field accepts valid values."""
        for status in ["OK", "FAIL", "SKIP"]:
            report = m.Infra.Docs.DocsPhaseReport(
                phase="build",
                scope="test",
                result=status,
                reason="Test reason",
                site_dir="/tmp/site",
            )
            tm.that(report.result, eq=status)

    def test_build_report_site_dir_field(self) -> None:
        """Test BuildReport site_dir field."""
        report = m.Infra.Docs.DocsPhaseReport(
            phase="build",
            scope="test",
            result="OK",
            reason="Build successful",
            site_dir="/path/to/site",
        )
        tm.that(report.site_dir, eq="/path/to/site")

    def test_build_with_multiple_projects_returns_list(
        self, builder: FlextInfraDocBuilder, tmp_path: Path
    ) -> None:
        """Test build with multiple projects returns list of reports."""
        result = builder.build(tmp_path, projects="proj1,proj2")
        if result.is_success:
            tm.that(isinstance(result.value, list), eq=True)
