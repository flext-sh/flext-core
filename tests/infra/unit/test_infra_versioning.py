"""Tests for FlextInfraVersioningService.

Tests cover semantic version parsing, bumping, and branch tag extraction.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_infra import FlextInfraUtilitiesVersioning
from flext_tests import tm


@pytest.fixture
def service() -> FlextInfraUtilitiesVersioning:
    """Create a versioning service instance."""
    return FlextInfraUtilitiesVersioning()


class TestParseSemver:
    """Tests for semantic version parsing."""

    def test_valid_version(self, service: FlextInfraUtilitiesVersioning) -> None:
        tm.ok(service.parse_semver("1.2.3"), eq=(1, 2, 3))

    def test_with_dev_suffix(self, service: FlextInfraUtilitiesVersioning) -> None:
        tm.ok(service.parse_semver("1.2.3-dev"), eq=(1, 2, 3))

    def test_zero_version(self, service: FlextInfraUtilitiesVersioning) -> None:
        tm.ok(service.parse_semver("0.0.0"), eq=(0, 0, 0))

    def test_large_numbers(self, service: FlextInfraUtilitiesVersioning) -> None:
        tm.ok(service.parse_semver("999.888.777"), eq=(999, 888, 777))

    def test_invalid_format(self, service: FlextInfraUtilitiesVersioning) -> None:
        tm.fail(service.parse_semver("1.2"), has="invalid semver")

    def test_non_numeric(self, service: FlextInfraUtilitiesVersioning) -> None:
        tm.fail(service.parse_semver("a.b.c"), has="invalid semver")

    def test_extra_suffix(self, service: FlextInfraUtilitiesVersioning) -> None:
        tm.fail(service.parse_semver("1.2.3-beta"), has="invalid semver")

    def test_result_type(self, service: FlextInfraUtilitiesVersioning) -> None:
        result = service.parse_semver("1.2.3")
        value = tm.ok(result, is_=tuple)
        tm.that(len(value), eq=3)


class TestBumpVersion:
    """Tests for version bumping."""

    def test_bump_major(self, service: FlextInfraUtilitiesVersioning) -> None:
        tm.ok(service.bump_version("1.2.3", "major"), eq="2.0.0")

    def test_bump_minor(self, service: FlextInfraUtilitiesVersioning) -> None:
        tm.ok(service.bump_version("1.2.3", "minor"), eq="1.3.0")

    def test_bump_patch(self, service: FlextInfraUtilitiesVersioning) -> None:
        tm.ok(service.bump_version("1.2.3", "patch"), eq="1.2.4")

    def test_bump_from_zero(self, service: FlextInfraUtilitiesVersioning) -> None:
        tm.ok(service.bump_version("0.0.0", "major"), eq="1.0.0")

    def test_invalid_bump_type(self, service: FlextInfraUtilitiesVersioning) -> None:
        tm.fail(service.bump_version("1.2.3", "invalid"), has="invalid bump type")

    def test_invalid_version(self, service: FlextInfraUtilitiesVersioning) -> None:
        tm.fail(service.bump_version("not.a.version", "major"), has="invalid semver")

    def test_result_type(self, service: FlextInfraUtilitiesVersioning) -> None:
        tm.ok(service.bump_version("1.2.3", "major"), is_=str)


class TestReleaseTagFromBranch:
    """Tests for branch tag extraction."""

    def test_release_pattern(self, service: FlextInfraUtilitiesVersioning) -> None:
        tm.ok(service.release_tag_from_branch("release/1.2.3"), eq="v1.2.3")

    def test_dev_pattern(self, service: FlextInfraUtilitiesVersioning) -> None:
        tm.ok(service.release_tag_from_branch("1.2.3-dev"), eq="v1.2.3")

    def test_invalid_pattern(self, service: FlextInfraUtilitiesVersioning) -> None:
        tm.fail(
            service.release_tag_from_branch("feature/my-feature"),
            has="does not match release pattern",
        )

    def test_empty_string(self, service: FlextInfraUtilitiesVersioning) -> None:
        tm.fail(
            service.release_tag_from_branch(""),
            has="does not match release pattern",
        )

    def test_result_type(self, service: FlextInfraUtilitiesVersioning) -> None:
        tm.ok(service.release_tag_from_branch("release/1.2.3"), is_=str)


class TestWorkspaceVersion:
    """Tests for workspace version reading and replacement."""

    def test_current_version_success(
        self, service: FlextInfraUtilitiesVersioning, tmp_path: Path
    ) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[project]\nversion = "1.2.3"\n', encoding="utf-8")
        tm.ok(service.current_workspace_version(tmp_path), eq="1.2.3")

    def test_current_version_missing_file(
        self, service: FlextInfraUtilitiesVersioning, tmp_path: Path
    ) -> None:
        tm.fail(service.current_workspace_version(tmp_path))

    def test_current_version_missing_project_table(
        self, service: FlextInfraUtilitiesVersioning, tmp_path: Path
    ) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[tool]\nname = "test"\n', encoding="utf-8")
        tm.fail(service.current_workspace_version(tmp_path), has="version not found")

    def test_current_version_missing_version_field(
        self, service: FlextInfraUtilitiesVersioning, tmp_path: Path
    ) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[project]\nname = "test"\n', encoding="utf-8")
        tm.fail(service.current_workspace_version(tmp_path), has="version not found")

    def test_current_version_empty_version(
        self, service: FlextInfraUtilitiesVersioning, tmp_path: Path
    ) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[project]\nversion = ""\n', encoding="utf-8")
        tm.fail(service.current_workspace_version(tmp_path), has="version not found")

    def test_replace_version_success(
        self, service: FlextInfraUtilitiesVersioning, tmp_path: Path
    ) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[project]\nversion = "1.0.0"\n', encoding="utf-8")
        tm.ok(service.replace_project_version(tmp_path, "2.0.0"), eq=True)
        tm.ok(service.current_workspace_version(tmp_path), eq="2.0.0")

    def test_replace_version_missing_file(
        self, service: FlextInfraUtilitiesVersioning, tmp_path: Path
    ) -> None:
        tm.fail(service.replace_project_version(tmp_path, "2.0.0"))

    def test_replace_version_missing_project_table(
        self, service: FlextInfraUtilitiesVersioning, tmp_path: Path
    ) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[tool]\nname = "test"\n', encoding="utf-8")
        tm.fail(
            service.replace_project_version(tmp_path, "2.0.0"),
            has="missing [project] table",
        )
