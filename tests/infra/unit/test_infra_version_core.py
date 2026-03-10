"""Tests for flext_infra.__version__ — FlextInfraVersion class methods.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_infra.__version__ import FlextInfraVersion


class TestFlextInfraVersionClass:
    """Test FlextInfraVersion class attributes and methods."""

    def test_flext_infra_version_has_version_attribute(self) -> None:
        """Test that FlextInfraVersion has __version__ attribute."""
        assert hasattr(FlextInfraVersion, "__version__")

    def test_flext_infra_version_has_version_info_attribute(self) -> None:
        """Test that FlextInfraVersion has __version_info__ attribute."""
        assert hasattr(FlextInfraVersion, "__version_info__")

    def test_flext_infra_version_has_title_attribute(self) -> None:
        """Test that FlextInfraVersion has __title__ attribute."""
        assert hasattr(FlextInfraVersion, "__title__")

    def test_flext_infra_version_has_description_attribute(self) -> None:
        """Test that FlextInfraVersion has __description__ attribute."""
        assert hasattr(FlextInfraVersion, "__description__")

    def test_flext_infra_version_has_author_attribute(self) -> None:
        """Test that FlextInfraVersion has __author__ attribute."""
        assert hasattr(FlextInfraVersion, "__author__")

    def test_flext_infra_version_has_author_email_attribute(self) -> None:
        """Test that FlextInfraVersion has __author_email__ attribute."""
        assert hasattr(FlextInfraVersion, "__author_email__")

    def test_flext_infra_version_has_license_attribute(self) -> None:
        """Test that FlextInfraVersion has __license__ attribute."""
        assert hasattr(FlextInfraVersion, "__license__")

    def test_flext_infra_version_has_url_attribute(self) -> None:
        """Test that FlextInfraVersion has __url__ attribute."""
        assert hasattr(FlextInfraVersion, "__url__")

    def test_get_version_string_returns_string(self) -> None:
        """Test that get_version_string() returns a string."""
        result = FlextInfraVersion.get_version_string()
        assert isinstance(result, str)

    def test_get_version_string_is_not_empty(self) -> None:
        """Test that get_version_string() returns non-empty string."""
        result = FlextInfraVersion.get_version_string()
        assert len(result) > 0

    def test_get_version_string_matches_version_attribute(self) -> None:
        """Test that get_version_string() matches __version__ attribute."""
        result = FlextInfraVersion.get_version_string()
        assert result == FlextInfraVersion.__version__

    def test_get_version_info_returns_tuple(self) -> None:
        """Test that get_version_info() returns a tuple."""
        result = FlextInfraVersion.get_version_info()
        assert isinstance(result, tuple)

    def test_get_version_info_is_not_empty(self) -> None:
        """Test that get_version_info() returns non-empty tuple."""
        result = FlextInfraVersion.get_version_info()
        assert len(result) > 0

    def test_get_version_info_matches_version_info_attribute(self) -> None:
        """Test that get_version_info() matches __version_info__ attribute."""
        result = FlextInfraVersion.get_version_info()
        assert result == FlextInfraVersion.__version_info__

    def test_get_version_info_contains_integers_or_strings(self) -> None:
        """Test that get_version_info() contains integers or strings."""
        result = FlextInfraVersion.get_version_info()
        for part in result:
            assert isinstance(part, (int, str))

    def test_is_version_at_least_with_current_version(self) -> None:
        """Test that is_version_at_least() returns True for current version."""
        version_info = FlextInfraVersion.get_version_info()
        major = version_info[0]
        minor = version_info[1] if len(version_info) > 1 else 0
        patch = version_info[2] if len(version_info) > 2 else 0
        if isinstance(major, int) and isinstance(minor, int) and isinstance(patch, int):
            result = FlextInfraVersion.is_version_at_least(major, minor, patch)
            assert result is True

    def test_is_version_at_least_with_lower_version(self) -> None:
        """Test that is_version_at_least() returns True for lower version."""
        result = FlextInfraVersion.is_version_at_least(0, 0, 0)
        assert result is True

    def test_is_version_at_least_with_higher_version(self) -> None:
        """Test that is_version_at_least() returns False for higher version."""
        result = FlextInfraVersion.is_version_at_least(999, 999, 999)
        assert result is False

    def test_is_version_at_least_with_major_only(self) -> None:
        """Test that is_version_at_least() works with major version only."""
        result = FlextInfraVersion.is_version_at_least(0)
        assert isinstance(result, bool)

    def test_is_version_at_least_with_major_and_minor(self) -> None:
        """Test that is_version_at_least() works with major and minor."""
        result = FlextInfraVersion.is_version_at_least(0, 0)
        assert isinstance(result, bool)

    def test_is_version_at_least_returns_bool(self) -> None:
        """Test that is_version_at_least() returns a boolean."""
        result = FlextInfraVersion.is_version_at_least(0, 0, 0)
        assert isinstance(result, bool)
