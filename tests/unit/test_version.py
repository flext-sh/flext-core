"""Comprehensive tests for FlextVersionManager - Version Management.

This module tests the version management functionality provided by FlextVersionManager.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time

from flext_core import FlextVersionManager


class TestFlextVersionManager:
    """Test suite for FlextVersionManager version management."""

    def test_version_manager_initialization(self) -> None:
        """Test version manager initialization."""
        version_manager = FlextVersionManager()

        assert version_manager.VERSION_MAJOR >= 0
        assert version_manager.VERSION_MINOR >= 0
        assert version_manager.VERSION_PATCH >= 0

    def test_version_constants(self) -> None:
        """Test version constants."""
        assert FlextVersionManager.VERSION_MAJOR >= 0
        assert FlextVersionManager.VERSION_MINOR >= 0
        assert FlextVersionManager.VERSION_PATCH >= 0
        assert FlextVersionManager.SEMVER_PARTS_COUNT == 3

    def test_release_information(self) -> None:
        """Test release information."""
        assert FlextVersionManager.RELEASE_NAME is not None
        assert FlextVersionManager.RELEASE_DATE is not None
        assert FlextVersionManager.BUILD_TYPE is not None

    def test_python_version_constraints(self) -> None:
        """Test Python version constraints."""
        min_version = FlextVersionManager.MIN_PYTHON_VERSION
        max_version = FlextVersionManager.MAX_PYTHON_VERSION

        assert len(min_version) == 3
        assert len(max_version) == 3
        assert min_version[0] >= 3
        assert max_version[0] >= 3

    def test_get_version_tuple(self) -> None:
        """Test get_version_tuple method."""
        version_tuple = FlextVersionManager.get_version_tuple()

        assert isinstance(version_tuple, tuple)
        assert len(version_tuple) == 3
        assert all(isinstance(v, int) for v in version_tuple)

    def test_get_version_info(self) -> None:
        """Test get_version_info method."""
        version_info = FlextVersionManager.get_version_info()

        assert hasattr(version_info, "major")
        assert hasattr(version_info, "minor")
        assert hasattr(version_info, "patch")
        assert isinstance(version_info.major, int)
        assert isinstance(version_info.minor, int)
        assert isinstance(version_info.patch, int)

    def test_get_version_string(self) -> None:
        """Test get_version_string method."""
        version_string = FlextVersionManager.get_version_string()

        assert isinstance(version_string, str)
        assert "." in version_string
        parts = version_string.split(".")
        assert len(parts) >= 3

    def test_get_available_features(self) -> None:
        """Test get_available_features method."""
        features = FlextVersionManager.get_available_features()

        assert isinstance(features, list)
        assert all(isinstance(feature, str) for feature in features)

    def test_compare_versions(self) -> None:
        """Test compare_versions method."""
        # Test version comparison
        result = FlextVersionManager.compare_versions("1.0.0", "1.0.1")
        assert result < 0  # 1.0.0 < 1.0.1

        result = FlextVersionManager.compare_versions("1.0.1", "1.0.0")
        assert result > 0  # 1.0.1 > 1.0.0

        result = FlextVersionManager.compare_versions("1.0.0", "1.0.0")
        assert result == 0  # 1.0.0 == 1.0.0

    def test_validate_version_format(self) -> None:
        """Test validate_version_format method."""
        # Valid versions
        assert FlextVersionManager.validate_version_format("1.0.0")
        assert FlextVersionManager.validate_version_format("0.0.0")
        assert FlextVersionManager.validate_version_format("999.999.999")

        # Invalid versions
        assert not FlextVersionManager.validate_version_format("1.0")
        assert not FlextVersionManager.validate_version_format("1.0.0.0")
        assert not FlextVersionManager.validate_version_format("invalid")
        assert not FlextVersionManager.validate_version_format("")

    def test_version_manager_performance(self) -> None:
        """Test version manager performance."""
        start_time = time.time()

        # Test performance of version operations
        for i in range(1000):
            FlextVersionManager.get_version_string()
            FlextVersionManager.get_version_tuple()
            FlextVersionManager.validate_version_format(f"{i % 10}.{i % 10}.{i % 10}")

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete quickly (less than 1 second)
        assert execution_time < 1.0

    def test_version_manager_type_safety(self) -> None:
        """Test type safety of version manager."""
        # Test type safety
        version_tuple = FlextVersionManager.get_version_tuple()
        assert isinstance(version_tuple, tuple)

        version_string = FlextVersionManager.get_version_string()
        assert isinstance(version_string, str)

        features = FlextVersionManager.get_available_features()
        assert isinstance(features, list)
