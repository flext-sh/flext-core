"""Test version module functionality."""

from __future__ import annotations

from flext_core import __version__
from flext_core.version import flext_get_version_info, get_version_string


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
        assert len(parts) >= 2  # At least major.minor

        # First two parts should be numeric
        assert parts[0].isdigit()
        assert parts[1].isdigit()

    def test_get_version_function(self) -> None:
        """Test get_version_string function."""
        version = get_version_string()
        assert __version__ in version  # Version string contains the version
        assert isinstance(version, str)

    def test_flext_get_version_info_function(self) -> None:
        """Test flext_get_version_info function."""
        version_info = flext_get_version_info()
        assert hasattr(version_info, "major")
        assert hasattr(version_info, "minor")
        assert hasattr(version_info, "patch")
        assert isinstance(version_info.major, int)
        assert isinstance(version_info.minor, int)
        assert isinstance(version_info.patch, int)
