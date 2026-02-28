"""Tests for flext_infra.__version__ module.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_infra.__version__ import (
    FlextInfraVersion,
    __author__,
    __author_email__,
    __description__,
    __license__,
    __title__,
    __url__,
    __version__,
    __version_info__,
)


class TestFlextInfraVersion:
    """Test FlextInfraVersion class."""

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

        # Ensure major, minor, patch are integers
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

    def test_get_package_info_returns_mapping(self) -> None:
        """Test that get_package_info() returns a mapping."""
        result = FlextInfraVersion.get_package_info()
        assert isinstance(result, dict)

    def test_get_package_info_has_name_key(self) -> None:
        """Test that get_package_info() has 'name' key."""
        result = FlextInfraVersion.get_package_info()
        assert "name" in result

    def test_get_package_info_has_version_key(self) -> None:
        """Test that get_package_info() has 'version' key."""
        result = FlextInfraVersion.get_package_info()
        assert "version" in result

    def test_get_package_info_has_description_key(self) -> None:
        """Test that get_package_info() has 'description' key."""
        result = FlextInfraVersion.get_package_info()
        assert "description" in result

    def test_get_package_info_has_author_key(self) -> None:
        """Test that get_package_info() has 'author' key."""
        result = FlextInfraVersion.get_package_info()
        assert "author" in result

    def test_get_package_info_has_author_email_key(self) -> None:
        """Test that get_package_info() has 'author_email' key."""
        result = FlextInfraVersion.get_package_info()
        assert "author_email" in result

    def test_get_package_info_has_license_key(self) -> None:
        """Test that get_package_info() has 'license' key."""
        result = FlextInfraVersion.get_package_info()
        assert "license" in result

    def test_get_package_info_has_url_key(self) -> None:
        """Test that get_package_info() has 'url' key."""
        result = FlextInfraVersion.get_package_info()
        assert "url" in result

    def test_get_package_info_all_values_are_strings(self) -> None:
        """Test that get_package_info() all values are strings."""
        result = FlextInfraVersion.get_package_info()
        for value in result.values():
            assert isinstance(value, str)

    def test_module_level_version_is_string(self) -> None:
        """Test that module-level __version__ is a string."""
        assert isinstance(__version__, str)

    def test_module_level_version_info_is_tuple(self) -> None:
        """Test that module-level __version_info__ is a tuple."""
        assert isinstance(__version_info__, tuple)

    def test_module_level_title_is_string(self) -> None:
        """Test that module-level __title__ is a string."""
        assert isinstance(__title__, str)

    def test_module_level_description_is_string(self) -> None:
        """Test that module-level __description__ is a string."""
        assert isinstance(__description__, str)

    def test_module_level_author_is_string(self) -> None:
        """Test that module-level __author__ is a string."""
        assert isinstance(__author__, str)

    def test_module_level_author_email_is_string(self) -> None:
        """Test that module-level __author_email__ is a string."""
        assert isinstance(__author_email__, str)

    def test_module_level_license_is_string(self) -> None:
        """Test that module-level __license__ is a string."""
        assert isinstance(__license__, str)

    def test_module_level_url_is_string(self) -> None:
        """Test that module-level __url__ is a string."""
        assert isinstance(__url__, str)

    def test_module_level_version_matches_class_version(self) -> None:
        """Test that module-level __version__ matches class version."""
        assert __version__ == FlextInfraVersion.__version__

    def test_module_level_version_info_matches_class_version_info(self) -> None:
        """Test that module-level __version_info__ matches class version_info."""
        assert __version_info__ == FlextInfraVersion.__version_info__
