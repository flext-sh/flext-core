"""Test version module functionality.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextVersionManager, __version__


class TestVersion:
    """Test version information."""

    def test_version_exists(self) -> None:
        """Test that version is defined."""
        assert __version__ is not None
        assert isinstance(__version__, str)
        assert len(__version__) > 0

    def test_version_format(self) -> None:
        """Test version follows semantic versioning."""
        parts = __version__.split(".")
        assert len(parts) >= 2, f"Expected {len(parts)} >= {2} # At least major.minor"

        # First two parts should be numeric
        assert parts[0].isdigit()
        assert parts[1].isdigit()

    def test_get_version_function(self) -> None:
        """Test get_version_string function."""
        version = FlextVersionManager.get_version_string()
        assert __version__ in version, (
            f"Expected {__version__} in {version} # Version string contains the version"
        )
        assert isinstance(version, str)

    def test_get_version_info_function(self) -> None:
        """Test get_version_info function."""
        version_info = FlextVersionManager.get_version_info()
        assert hasattr(version_info, "major")
        assert hasattr(version_info, "minor")
        assert hasattr(version_info, "patch")
        assert isinstance(version_info.major, int)
        assert isinstance(version_info.minor, int)
        assert isinstance(version_info.patch, int)

    def test_get_version_tuple(self) -> None:
        """Test get_version_tuple function."""
        version_tuple = FlextVersionManager.get_version_tuple()
        assert isinstance(version_tuple, tuple)
        assert len(version_tuple) == 3
        assert all(isinstance(v, int) for v in version_tuple)

    def test_get_available_features(self) -> None:
        """Test get_available_features function."""
        features = FlextVersionManager.get_available_features()
        assert isinstance(features, list)
        # Should have some features available
        assert len(features) > 0

    def test_compare_versions(self) -> None:
        """Test compare_versions function."""
        # Test equal versions
        result = FlextVersionManager.compare_versions("1.0.0", "1.0.0")
        assert result == 0

        # Test version1 > version2
        result = FlextVersionManager.compare_versions("1.1.0", "1.0.0")
        assert result > 0

        # Test version1 < version2
        result = FlextVersionManager.compare_versions("1.0.0", "1.1.0")
        assert result < 0

    def test_validate_version_format(self) -> None:
        """Test validate_version_format function."""
        # Test valid version formats
        assert FlextVersionManager.validate_version_format("1.0.0") is True
        assert FlextVersionManager.validate_version_format("0.9.0") is True
        assert FlextVersionManager.validate_version_format("10.20.30") is True

        # Test invalid version formats
        assert FlextVersionManager.validate_version_format("1.0") is False
        assert FlextVersionManager.validate_version_format("1.0.0.0") is False
        assert FlextVersionManager.validate_version_format("invalid") is False
        assert FlextVersionManager.validate_version_format("") is False

        # Test additional edge cases to cover missing lines
        assert FlextVersionManager.validate_version_format(None) is False  # Line 138
        assert FlextVersionManager.validate_version_format("1.a.0") is False  # Line 145
        assert (
            FlextVersionManager.validate_version_format("1.-1.0") is False
        )  # Line 147
        assert (
            FlextVersionManager.validate_version_format("a.b.c") is False
        )  # Line 147-149

    def test_compatibility_result_creation(self) -> None:
        """Test CompatibilityResult creation to cover missing init lines."""
        # Create a CompatibilityResult to cover lines 74-78
        check = FlextVersionManager.CompatibilityResult(
            is_compatible=True,
            current_version=(1, 0, 0),
            required_version=(1, 0, 0),
            error_message="No error",
            recommendations=["Keep using current version"],
        )
        assert check.is_compatible is True
        assert check.current_version == (1, 0, 0)
        assert check.required_version == (1, 0, 0)
        assert check.error_message == "No error"
        assert check.recommendations == ["Keep using current version"]
