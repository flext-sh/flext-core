"""Test version module functionality."""

from __future__ import annotations

from flext_core import (
    __version__,
    check_python_compatibility,
    compare_versions,
    get_available_features,
    get_version_info,
    get_version_string,
    get_version_tuple,
    is_feature_available,
    validate_version_format,
)


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
        version = get_version_string()
        assert __version__ in version, (
            f"Expected {__version__} in {version} # Version string contains the version"
        )
        assert isinstance(version, str)

    def test_get_version_info_function(self) -> None:
        """Test get_version_info function."""
        version_info = get_version_info()
        assert hasattr(version_info, "major")
        assert hasattr(version_info, "minor")
        assert hasattr(version_info, "patch")
        assert isinstance(version_info.major, int)
        assert isinstance(version_info.minor, int)
        assert isinstance(version_info.patch, int)

    def test_get_version_tuple(self) -> None:
        """Test get_version_tuple function."""
        version_tuple = get_version_tuple()
        assert isinstance(version_tuple, tuple)
        assert len(version_tuple) == 3
        assert all(isinstance(v, int) for v in version_tuple)

    def test_check_python_compatibility(self) -> None:
        """Test check_python_compatibility function."""
        compatibility = check_python_compatibility()
        assert hasattr(compatibility, "is_compatible")
        assert hasattr(compatibility, "current_version")
        assert hasattr(compatibility, "required_version")
        assert isinstance(compatibility.is_compatible, bool)
        # Should be compatible since we're running the tests
        assert compatibility.is_compatible is True

    def test_is_feature_available(self) -> None:
        """Test is_feature_available function."""
        # Test with known features
        result = is_feature_available("pydantic_validation")
        assert isinstance(result, bool)

        # Test with unknown feature
        result = is_feature_available("unknown_feature")
        assert isinstance(result, bool)

    def test_get_available_features(self) -> None:
        """Test get_available_features function."""
        features = get_available_features()
        assert isinstance(features, list)
        # Should have some features available
        assert len(features) > 0

    def test_compare_versions(self) -> None:
        """Test compare_versions function."""
        # Test equal versions
        result = compare_versions("1.0.0", "1.0.0")
        assert result == 0

        # Test version1 > version2
        result = compare_versions("1.1.0", "1.0.0")
        assert result > 0

        # Test version1 < version2
        result = compare_versions("1.0.0", "1.1.0")
        assert result < 0

    def test_validate_version_format(self) -> None:
        """Test validate_version_format function."""
        # Test valid version formats
        assert validate_version_format("1.0.0") is True
        assert validate_version_format("0.9.0") is True
        assert validate_version_format("10.20.30") is True

        # Test invalid version formats
        assert validate_version_format("1.0") is False
        assert validate_version_format("1.0.0.0") is False
        assert validate_version_format("invalid") is False
        assert validate_version_format("") is False
