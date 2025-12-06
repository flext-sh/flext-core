"""Tests for flext_core.__version__ module.

Tests cover FlextVersion class methods and module-level exports,
including version string, version info tuple, version comparison,
package info dictionary, and PackageNotFoundError fallback handling.
"""

from __future__ import annotations

import importlib
import sys
from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

from flext_core import __version__, __version_info__
from flext_core.__version__ import FlextVersion


class TestFlextVersion:
    """Test FlextVersion class methods and functionality."""

    def test_get_version_string(self) -> None:
        """Test get_version_string returns version string."""
        version = FlextVersion.get_version_string()
        assert isinstance(version, str)
        assert len(version) > 0

    def test_get_version_info(self) -> None:
        """Test get_version_info returns version tuple."""
        version_info = FlextVersion.get_version_info()
        assert isinstance(version_info, tuple)
        assert len(version_info) > 0

    def test_is_version_at_least(self) -> None:
        """Test is_version_at_least version comparison."""
        # Should always be at least 0.0.0
        assert FlextVersion.is_version_at_least(0, 0, 0) is True
        # Test with current version components
        current_version_info = FlextVersion.get_version_info()
        current_major = (
            current_version_info[0] if isinstance(current_version_info[0], int) else 0
        )
        assert FlextVersion.is_version_at_least(current_major - 1, 0, 0) is True

    def test_get_package_info(self) -> None:
        """Test get_package_info returns package metadata dictionary."""
        info = FlextVersion.get_package_info()
        assert isinstance(info, dict)
        assert "name" in info
        assert "version" in info
        assert "description" in info
        assert "author" in info
        assert "author_email" in info
        assert "license" in info
        assert "url" in info

    def test_module_level_exports(self) -> None:
        """Test module-level version exports."""
        assert isinstance(__version__, str)
        assert isinstance(__version_info__, tuple)

    def test_package_not_found_error_fallback(self) -> None:
        """Test PackageNotFoundError fallback handling.

        Note: Module-level exception handlers are difficult to test as they
        execute at import time. This test verifies the fallback mechanism
        by temporarily removing the module from cache and reimporting with
        mocked metadata that raises PackageNotFoundError.
        """
        # Save original module reference for restoration
        module_name = "flext_core.__version__"
        original_module = sys.modules.get(module_name)

        # Remove module from cache to allow fresh import
        if module_name in sys.modules:
            del sys.modules[module_name]

        try:
            with patch(
                "importlib.metadata.metadata",
                side_effect=PackageNotFoundError("test"),
            ):
                # Import module fresh to trigger exception handler
                version_module = importlib.import_module(module_name)

                # Verify fallback values are set
                assert version_module.__version__ == "0.0.0-dev"
                assert version_module.__title__ == "flext-core"
        finally:
            # Restore original module state
            if module_name in sys.modules:
                del sys.modules[module_name]
            if original_module is not None:
                sys.modules[module_name] = original_module
            else:
                # Reimport if no original existed
                importlib.import_module(module_name)
