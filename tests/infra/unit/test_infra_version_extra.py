"""Tests for flext_infra.__version__ — package info and module-level exports.

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


class TestFlextInfraVersionPackageInfo:
    """Test FlextInfraVersion.get_package_info() method."""

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


class TestFlextInfraVersionModuleLevel:
    """Test module-level version exports."""

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
