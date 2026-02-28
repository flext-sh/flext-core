"""Tests for FlextInfraVersioningService.

Tests cover semantic version parsing, bumping, and branch tag extraction.
"""

from __future__ import annotations

import pytest
from flext_core import r
from flext_infra import FlextInfraVersioningService


class TestFlextInfraVersioningService:
    """Test suite for FlextInfraVersioningService."""

    @pytest.fixture
    def service(self) -> FlextInfraVersioningService:
        """Create a versioning service instance."""
        return FlextInfraVersioningService()

    def test_parse_semver_valid_version(
        self,
        service: FlextInfraVersioningService,
    ) -> None:
        """Test parsing a valid semantic version."""
        result = service.parse_semver("1.2.3")

        assert result.is_success
        assert result.value == (1, 2, 3)

    def test_parse_semver_with_dev_suffix(
        self,
        service: FlextInfraVersioningService,
    ) -> None:
        """Test parsing semantic version with -dev suffix."""
        result = service.parse_semver("1.2.3-dev")

        assert result.is_success
        assert result.value == (1, 2, 3)

    def test_parse_semver_zero_version(
        self,
        service: FlextInfraVersioningService,
    ) -> None:
        """Test parsing version with zeros."""
        result = service.parse_semver("0.0.0")

        assert result.is_success
        assert result.value == (0, 0, 0)

    def test_parse_semver_large_numbers(
        self,
        service: FlextInfraVersioningService,
    ) -> None:
        """Test parsing version with large numbers."""
        result = service.parse_semver("999.888.777")

        assert result.is_success
        assert result.value == (999, 888, 777)

    def test_parse_semver_invalid_format(
        self,
        service: FlextInfraVersioningService,
    ) -> None:
        """Test parsing invalid version format."""
        result = service.parse_semver("1.2")

        assert result.is_failure
        assert result.error and "invalid semver" in result.error

    def test_parse_semver_non_numeric(
        self,
        service: FlextInfraVersioningService,
    ) -> None:
        """Test parsing version with non-numeric parts."""
        result = service.parse_semver("a.b.c")

        assert result.is_failure
        assert result.error and "invalid semver" in result.error

    def test_parse_semver_extra_suffix(
        self,
        service: FlextInfraVersioningService,
    ) -> None:
        """Test parsing version with invalid suffix."""
        result = service.parse_semver("1.2.3-beta")

        assert result.is_failure
        assert result.error and "invalid semver" in result.error

    def test_bump_version_major(
        self,
        service: FlextInfraVersioningService,
    ) -> None:
        """Test bumping major version."""
        result = service.bump_version("1.2.3", "major")

        assert result.is_success
        assert result.value == "2.0.0"

    def test_bump_version_minor(
        self,
        service: FlextInfraVersioningService,
    ) -> None:
        """Test bumping minor version."""
        result = service.bump_version("1.2.3", "minor")

        assert result.is_success
        assert result.value == "1.3.0"

    def test_bump_version_patch(
        self,
        service: FlextInfraVersioningService,
    ) -> None:
        """Test bumping patch version."""
        result = service.bump_version("1.2.3", "patch")

        assert result.is_success
        assert result.value == "1.2.4"

    def test_bump_version_from_zero(
        self,
        service: FlextInfraVersioningService,
    ) -> None:
        """Test bumping from 0.0.0."""
        result = service.bump_version("0.0.0", "major")

        assert result.is_success
        assert result.value == "1.0.0"

    def test_bump_version_invalid_bump_type(
        self,
        service: FlextInfraVersioningService,
    ) -> None:
        """Test bumping with invalid bump type."""
        result = service.bump_version("1.2.3", "invalid")

        assert result.is_failure
        assert result.error and "invalid bump type" in result.error

    def test_bump_version_invalid_version(
        self,
        service: FlextInfraVersioningService,
    ) -> None:
        """Test bumping invalid version string."""
        result = service.bump_version("not.a.version", "major")

        assert result.is_failure
        assert result.error and "invalid semver" in result.error

    def test_release_tag_from_branch_release_pattern(
        self,
        service: FlextInfraVersioningService,
    ) -> None:
        """Test extracting tag from release branch."""
        result = service.release_tag_from_branch("release/1.2.3")

        assert result.is_success
        assert result.value == "v1.2.3"

    def test_release_tag_from_branch_dev_pattern(
        self,
        service: FlextInfraVersioningService,
    ) -> None:
        """Test extracting tag from dev branch."""
        result = service.release_tag_from_branch("1.2.3-dev")

        assert result.is_success
        assert result.value == "v1.2.3"

    def test_release_tag_from_branch_invalid_pattern(
        self,
        service: FlextInfraVersioningService,
    ) -> None:
        """Test extracting tag from invalid branch name."""
        result = service.release_tag_from_branch("feature/my-feature")

        assert result.is_failure
        assert result.error and "does not match release pattern" in result.error

    def test_release_tag_from_branch_empty_string(
        self,
        service: FlextInfraVersioningService,
    ) -> None:
        """Test extracting tag from empty branch name."""
        result = service.release_tag_from_branch("")

        assert result.is_failure
        assert result.error and "does not match release pattern" in result.error

    def test_parse_semver_result_type(
        self,
        service: FlextInfraVersioningService,
    ) -> None:
        """Test that parse_semver returns properly typed FlextResult."""
        result = service.parse_semver("1.2.3")

        assert isinstance(result, type(r[tuple[int, int, int]].ok((0, 0, 0))))
        assert result.is_success
        assert isinstance(result.value, tuple)
        assert len(result.value) == 3

    def test_bump_version_result_type(
        self,
        service: FlextInfraVersioningService,
    ) -> None:
        """Test that bump_version returns properly typed FlextResult."""
        result = service.bump_version("1.2.3", "major")

        assert isinstance(result, type(r[str].ok("")))
        assert result.is_success
        assert isinstance(result.value, str)

    def test_release_tag_from_branch_result_type(
        self,
        service: FlextInfraVersioningService,
    ) -> None:
        """Test that release_tag_from_branch returns properly typed FlextResult."""
        result = service.release_tag_from_branch("release/1.2.3")

        assert isinstance(result, type(r[str].ok("")))
        assert result.is_success
        assert isinstance(result.value, str)
