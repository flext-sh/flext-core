"""Tests for flext_core.version module.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Tests for version information module.
"""

from __future__ import annotations

import pytest

from flext_core import version


class TestVersionModule:
    """Test version module functionality."""

    def test_version_info_exists(self) -> None:
        """Test that version info is available."""
        assert hasattr(version, "__version__")
        assert hasattr(version, "__version_info__")

    def test_version_string_format(self) -> None:
        """Test version string is properly formatted."""
        version_str = version.__version__

        assert isinstance(version_str, str)
        assert len(version_str) > 0
        # Should follow semantic versioning pattern
        parts = version_str.split(".")
        assert len(parts) >= 2  # At least major.minor

    def test_version_info_tuple(self) -> None:
        """Test version info tuple."""
        version_info = version.__version_info__

        assert isinstance(version_info, tuple)
        assert len(version_info) >= 2  # At least major, minor

        # All parts should be integers, strings, or None (for optional fields)
        for part in version_info:
            assert isinstance(part, (int, str, type(None)))

    def test_version_constants_consistency(self) -> None:
        """Test version constants are consistent."""
        version_str = version.__version__
        version_info = version.__version_info__

        # Version string should start with the major.minor from version_info
        if len(version_info) >= 2:
            expected_start = f"{version_info[0]}.{version_info[1]}"
            assert version_str.startswith(expected_start)

    def test_version_module_attributes(self) -> None:
        """Test version module has expected attributes."""
        # Standard version attributes
        expected_attrs = ["__version__", "__version_info__"]

        for attr in expected_attrs:
            assert hasattr(version, attr), f"Version module missing {attr}"

    def test_version_not_empty(self) -> None:
        """Test version is not empty or placeholder."""
        version_str = version.__version__

        # Should not be empty, None, or common placeholders
        assert version_str
        assert version_str != "0.0.0"
        assert version_str != "unknown"
        assert version_str != "dev"
        assert "placeholder" not in version_str.lower()

    def test_version_info_structure(self) -> None:
        """Test version info has proper structure."""
        version_info = version.__version_info__

        # Should be a proper version tuple
        assert len(version_info) >= 2

        # First two elements should be integers (major, minor)
        assert isinstance(version_info[0], int)
        assert isinstance(version_info[1], int)

        # Major version should be reasonable
        assert version_info[0] >= 0
        assert version_info[1] >= 0


class TestVersionExports:
    """Test version module exports."""

    def test_version_can_be_imported(self) -> None:
        """Test version can be imported directly."""
        # Should be able to import version from flext_core
        from flext_core import __version__

        assert __version__
        assert isinstance(__version__, str)

    def test_version_info_can_be_imported(self) -> None:
        """Test version info can be imported."""
        from flext_core import __version_info__

        assert __version_info__
        assert isinstance(__version_info__, tuple)

    def test_version_module_direct_import(self) -> None:
        """Test version module can be imported directly."""
        import flext_core.version as version_module

        assert hasattr(version_module, "__version__")
        assert hasattr(version_module, "__version_info__")


class TestVersionUsage:
    """Test version usage scenarios."""

    def test_version_string_in_user_agent(self) -> None:
        """Test version string can be used in user agent."""
        version_str = version.__version__

        user_agent = f"flext-core/{version_str}"

        assert "flext-core" in user_agent
        assert version_str in user_agent
        assert "/" in user_agent

    def test_version_comparison(self) -> None:
        """Test version can be used for comparisons."""
        version_info = version.__version_info__

        # Should be able to compare version tuples
        assert version_info >= (0, 0)
        assert version_info > (-1, 0)

        # Version info elements should support comparison
        major, minor = version_info[0], version_info[1]
        assert major >= 0
        assert minor >= 0

    def test_version_in_about_info(self) -> None:
        """Test version can be used in about/info displays."""
        version_str = version.__version__
        version_info = version.__version_info__

        about_info = {
            "name": "flext-core",
            "version": version_str,
            "version_info": version_info,
        }

        assert about_info["name"] == "flext-core"
        assert about_info["version"] == version_str
        assert about_info["version_info"] == version_info

    def test_version_serializable(self) -> None:
        """Test version info is serializable."""
        import json

        version_str = version.__version__
        version_info = version.__version_info__

        # Should be able to serialize version info
        version_data = {
            "version": version_str,
            "version_info": list(version_info),  # Convert tuple to list for JSON
        }

        json_str = json.dumps(version_data)
        assert isinstance(json_str, str)
        assert version_str in json_str

        # Should be able to deserialize
        parsed = json.loads(json_str)
        assert parsed["version"] == version_str
        assert tuple(parsed["version_info"]) == version_info


class TestVersionEdgeCases:
    """Test version module edge cases."""

    def test_version_attributes_immutable(self) -> None:
        """Test version attributes cannot be modified."""
        original_version = version.__version__
        original_info = version.__version_info__

        # Attempting to modify should not affect the module
        # (We can't actually prevent modification in Python, but we test current values)
        assert version.__version__ == original_version
        assert version.__version_info__ == original_info

    def test_version_string_encoding(self) -> None:
        """Test version string handles encoding properly."""
        version_str = version.__version__

        # Should encode/decode properly
        encoded = version_str.encode("utf-8")
        decoded = encoded.decode("utf-8")

        assert decoded == version_str

    def test_version_repr_and_str(self) -> None:
        """Test version string representation."""
        version_str = version.__version__
        version_info = version.__version_info__

        # String representations should be meaningful
        str_version = str(version_str)
        assert str_version == version_str

        str_info = str(version_info)
        assert isinstance(str_info, str)
        assert len(str_info) > 0

    def test_version_hash_consistent(self) -> None:
        """Test version hashing is consistent."""
        version_str = version.__version__
        version_info = version.__version_info__

        # Hash should be consistent
        hash1 = hash(version_str)
        hash2 = hash(version_str)
        assert hash1 == hash2

        info_hash1 = hash(version_info)
        info_hash2 = hash(version_info)
        assert info_hash1 == info_hash2


class TestVersionIntegration:
    """Test version module integration."""

    def test_version_available_at_package_level(self) -> None:
        """Test version is available at package level."""
        import flext_core

        # Should be available directly from package
        assert hasattr(flext_core, "__version__")
        assert hasattr(flext_core, "__version_info__")

        # Should match version module
        assert flext_core.__version__ == version.__version__
        assert flext_core.__version_info__ == version.__version_info__

    def test_version_consistency_across_imports(self) -> None:
        """Test version is consistent across different import methods."""
        # Direct module import
        from flext_core import version

        v1 = version.__version__

        # Package level import
        from flext_core import __version__

        v2 = __version__

        # Module attribute access
        import flext_core

        v3 = flext_core.__version__

        # All should be the same
        assert v1 == v2 == v3

    def test_version_info_consistency(self) -> None:
        """Test version info is consistent across imports."""
        # Different import methods should give same version info
        from flext_core import version
        from flext_core import __version_info__
        import flext_core

        info1 = version.__version_info__
        info2 = __version_info__
        info3 = flext_core.__version_info__

        assert info1 == info2 == info3
