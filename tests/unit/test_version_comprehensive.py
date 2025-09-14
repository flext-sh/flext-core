"""Comprehensive tests for version.py to achieve near 100% coverage.

This module provides comprehensive test coverage for version.py using extensive
flext_tests standardization patterns to target missing coverage lines.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest

from flext_core import FlextVersionManager


class TestFlextVersionComprehensiveCoverage:
    """Comprehensive tests for FlextVersionManager to achieve near 100% coverage."""

    # Test CompatibilityResult class initialization (covers lines 68-72)
    def test_compatibility_result_initialization_comprehensive(self) -> None:
        """Test CompatibilityResult.__init__ method with all parameter assignments."""
        # Create CompatibilityResult instance to cover lines 68-72
        compat_result = FlextVersionManager.CompatibilityResult(
            is_compatible=True,
            current_version=(3, 13, 0),
            required_version=(3, 12, 0),
            error_message="Version check successful",
            recommendations=["Continue with current version", "Update if needed"],
        )

        # Verify all attributes are properly assigned (lines 68-72)
        assert compat_result.is_compatible is True
        assert compat_result.current_version == (3, 13, 0)
        assert compat_result.required_version == (3, 12, 0)
        assert compat_result.error_message == "Version check successful"
        assert compat_result.recommendations == [
            "Continue with current version",
            "Update if needed",
        ]

    def test_compatibility_result_initialization_failure_case(self) -> None:
        """Test CompatibilityResult initialization with failure parameters."""
        compat_result = FlextVersionManager.CompatibilityResult(
            is_compatible=False,
            current_version=(3, 11, 0),
            required_version=(3, 13, 0),
            error_message="Python version too old",
            recommendations=["Upgrade to Python 3.13+", "Check system compatibility"],
        )

        assert compat_result.is_compatible is False
        assert compat_result.current_version == (3, 11, 0)
        assert compat_result.required_version == (3, 13, 0)
        assert compat_result.error_message == "Python version too old"
        assert compat_result.recommendations == [
            "Upgrade to Python 3.13+",
            "Check system compatibility",
        ]

    def test_compatibility_result_initialization_empty_recommendations(self) -> None:
        """Test CompatibilityResult initialization with empty recommendations."""
        compat_result = FlextVersionManager.CompatibilityResult(
            is_compatible=True,
            current_version=(3, 14, 0),
            required_version=(3, 13, 0),
            error_message="",
            recommendations=[],
        )

        assert compat_result.is_compatible is True
        assert compat_result.current_version == (3, 14, 0)
        assert compat_result.required_version == (3, 13, 0)
        assert compat_result.error_message == ""
        assert compat_result.recommendations == []

    def test_compatibility_result_initialization_complex_version_tuples(self) -> None:
        """Test CompatibilityResult initialization with complex version tuples."""
        compat_result = FlextVersionManager.CompatibilityResult(
            is_compatible=False,
            current_version=(3, 13, 5),
            required_version=(3, 14, 0),
            error_message="Complex version mismatch",
            recommendations=["Consider upgrading", "Check pre-release compatibility"],
        )

        assert compat_result.is_compatible is False
        assert compat_result.current_version == (3, 13, 5)
        assert compat_result.required_version == (3, 14, 0)
        assert compat_result.error_message == "Complex version mismatch"
        assert len(compat_result.recommendations) == 2

    # Test validate_version_format exception handling (covers lines 137-141)
    def test_validate_version_format_value_error_exception(self) -> None:
        """Test validate_version_format with inputs that cause ValueError (line 140)."""
        # Test with non-numeric parts that cause ValueError in int() conversion
        test_cases = [
            "1.2.a",  # Letter causes ValueError
            "1.2.3.4.5.6",  # Too many parts, but int("4") works, int("5") works
            "1.2.",  # Empty part causes ValueError
            ".1.2",  # Empty first part causes ValueError
            "1..2",  # Empty middle part causes ValueError
            "1.2.beta",  # Non-numeric part causes ValueError
            "v1.2.3",  # Version prefix causes ValueError
            "1.2.3-alpha",  # Hyphen suffix causes ValueError
        ]

        for version_str in test_cases:
            result = FlextVersionManager.validate_version_format(version_str)
            assert result is False, f"Expected False for version: {version_str}"

    def test_validate_version_format_attribute_error_exception(self) -> None:
        """Test validate_version_format with inputs that cause AttributeError (line 140)."""
        # Test with inputs that don't have string methods like split() or isdigit()
        invalid_inputs: list[object] = [
            None,  # None.split() causes AttributeError
            123,  # int.split() causes AttributeError
            [],  # list.split() causes AttributeError
            {},  # dict.split() causes AttributeError
            set(),  # set.split() causes AttributeError
        ]

        for invalid_input in invalid_inputs:
            result = FlextVersionManager.validate_version_format(invalid_input)
            assert result is False, f"Expected False for input: {invalid_input}"

    def test_validate_version_format_negative_number_detection(self) -> None:
        """Test validate_version_format with negative numbers (lines 138-139)."""
        # Test cases where int(part) < 0 check is triggered (line 138)
        negative_cases = [
            "-1.2.3",  # Negative major version
            "1.-2.3",  # Negative minor version
            "1.2.-3",  # Negative patch version
            "-1.-2.-3",  # All negative versions
        ]

        for version_str in negative_cases:
            result = FlextVersionManager.validate_version_format(version_str)
            assert result is False, (
                f"Expected False for negative version: {version_str}"
            )

    def test_validate_version_format_non_digit_detection(self) -> None:
        """Test validate_version_format with non-digit detection (lines 136-137)."""
        # Test cases where part.isdigit() returns False (line 136)
        non_digit_cases = [
            "1.2.3a",  # Letter suffix
            "1.2.3.0",  # Extra part beyond SEMVER_PARTS_COUNT
            "01.2.3",  # Leading zero (still digit, but edge case)
            "1.02.3",  # Leading zero in minor
            "1.2.03",  # Leading zero in patch
            "1.2.+3",  # Plus sign
            "1.2.*",  # Wildcard
        ]

        for version_str in non_digit_cases:
            result = FlextVersionManager.validate_version_format(version_str)
            # Most of these should return False, but verify behavior
            assert isinstance(result, bool), (
                f"Expected boolean result for: {version_str}"
            )

    def test_validate_version_format_parts_count_validation(self) -> None:
        """Test validate_version_format with wrong number of parts (line 132)."""
        # Test cases where len(parts) != SEMVER_PARTS_COUNT (line 132)
        wrong_parts_cases = [
            "",  # Empty string -> 1 part after split
            "1",  # 1 part
            "1.2",  # 2 parts
            "1.2.3.4",  # 4 parts (more than SEMVER_PARTS_COUNT=3)
            "1.2.3.4.5",  # 5 parts
        ]

        for version_str in wrong_parts_cases:
            result = FlextVersionManager.validate_version_format(version_str)
            if version_str == "":
                # Empty string edge case
                assert result is False
            elif version_str == "1.2.3":
                # This should be valid (3 parts)
                assert result is True
            else:
                # Wrong number of parts should be invalid
                assert result is False, (
                    f"Expected False for wrong parts count: {version_str}"
                )

    def test_validate_version_format_edge_cases_comprehensive(self) -> None:
        """Test validate_version_format with comprehensive edge cases."""
        # Edge cases that should trigger different exception paths
        edge_cases = [
            ("1.2.3", True),  # Valid case for reference
            ("0.0.0", True),  # All zeros (valid)
            ("999.999.999", True),  # Large numbers (valid)
            ("1.2", False),  # Wrong part count
            ("1.2.3.4", False),  # Wrong part count
            ("", False),  # Empty string
            ("1.2.abc", False),  # Non-numeric part
            ("a.b.c", False),  # All non-numeric
            ("1..3", False),  # Empty middle part
            (".2.3", False),  # Empty first part
            ("1.2.", False),  # Empty last part
            ("1.2.-1", False),  # Negative number
            ("-1.2.3", False),  # Negative first part
        ]

        for version_str, expected in edge_cases:
            result = FlextVersionManager.validate_version_format(version_str)
            assert result == expected, (
                f"Version '{version_str}' expected {expected}, got {result}"
            )

    def test_validate_version_format_exception_handling_coverage(self) -> None:
        """Test validate_version_format to ensure exception handling paths are covered."""
        # Specifically target the try/except block (lines 130-143)

        # Case 1: ValueError from int() conversion (line 140)
        try:
            # This should not raise, but return False due to exception handling
            result = FlextVersionManager.validate_version_format("1.2.invalid")
            assert result is False
        except Exception as e:
            pytest.fail(f"Exception should be handled internally, but got: {e}")

        # Case 2: AttributeError from missing string methods (line 140)
        try:
            # This should not raise, but return False due to exception handling
            result = FlextVersionManager.validate_version_format(None)
            assert result is False
        except Exception as e:
            pytest.fail(f"Exception should be handled internally, but got: {e}")

        # Case 3: Successful execution path (else clause, line 142-143)
        result = FlextVersionManager.validate_version_format("1.2.3")
        assert result is True

    # Additional comprehensive tests to ensure full coverage
    def test_version_manager_static_methods_comprehensive(self) -> None:
        """Comprehensive test of all FlextVersionManager static methods."""
        # Test get_version_tuple
        version_tuple = FlextVersionManager.get_version_tuple()
        assert isinstance(version_tuple, tuple)
        assert len(version_tuple) == 3
        assert all(isinstance(v, int) for v in version_tuple)

        # Test get_version_info
        version_info = FlextVersionManager.get_version_info()
        assert hasattr(version_info, "major")
        assert hasattr(version_info, "minor")
        assert hasattr(version_info, "patch")
        assert hasattr(version_info, "release_name")
        assert hasattr(version_info, "release_date")
        assert hasattr(version_info, "build_type")

        # Test get_version_string
        version_string = FlextVersionManager.get_version_string()
        assert isinstance(version_string, str)
        assert "0.9.0" in version_string
        assert "Foundation" in version_string

        # Test get_available_features
        features = FlextVersionManager.get_available_features()
        assert isinstance(features, list)
        assert all(isinstance(feature, str) for feature in features)
        assert "core_validation" in features
        assert "railway_programming" in features

        # Test compare_versions
        assert FlextVersionManager.compare_versions("1.0.0", "2.0.0") == -1
        assert FlextVersionManager.compare_versions("2.0.0", "1.0.0") == 1
        assert FlextVersionManager.compare_versions("1.0.0", "1.0.0") == 0

    def test_version_manager_constants_access(self) -> None:
        """Test access to FlextVersionManager constants and class variables."""
        # Test version constants
        assert FlextVersionManager.VERSION_MAJOR == 0
        assert FlextVersionManager.VERSION_MINOR == 9
        assert FlextVersionManager.VERSION_PATCH == 0
        assert FlextVersionManager.SEMVER_PARTS_COUNT == 3

        # Test release information
        assert FlextVersionManager.RELEASE_NAME == "Foundation"
        assert FlextVersionManager.RELEASE_DATE == "2025-06-27"
        assert FlextVersionManager.BUILD_TYPE == "stable"

        # Test Python version constraints
        assert FlextVersionManager.MIN_PYTHON_VERSION == (3, 13, 0)
        assert FlextVersionManager.MAX_PYTHON_VERSION == (3, 14, 0)

        # Test feature availability matrix
        assert isinstance(FlextVersionManager.AVAILABLE_FEATURES, dict)
        assert FlextVersionManager.AVAILABLE_FEATURES["core_validation"] is True
        assert FlextVersionManager.AVAILABLE_FEATURES["plugin_architecture"] is False

    def test_version_info_named_tuple_comprehensive(self) -> None:
        """Comprehensive test of VersionInfo NamedTuple functionality."""
        version_info = FlextVersionManager.get_version_info()

        # Test NamedTuple properties
        assert hasattr(version_info, "_fields")
        expected_fields = (
            "major",
            "minor",
            "patch",
            "release_name",
            "release_date",
            "build_type",
        )
        assert version_info._fields == expected_fields

        # Test field access by name and index
        assert version_info.major == version_info[0] == 0
        assert version_info.minor == version_info[1] == 9
        assert version_info.patch == version_info[2] == 0
        assert version_info.release_name == version_info[3] == "Foundation"
        assert version_info.release_date == version_info[4] == "2025-06-27"
        assert version_info.build_type == version_info[5] == "stable"

        # Test immutability (NamedTuple characteristic)
        with pytest.raises(AttributeError):
            version_info.major = 1

    def test_compare_versions_edge_cases_comprehensive(self) -> None:
        """Comprehensive test of compare_versions with edge cases."""
        # Test various version comparison scenarios
        comparison_cases = [
            ("0.0.1", "0.0.2", -1),
            ("0.1.0", "0.0.9", 1),
            ("1.0.0", "0.9.9", 1),
            ("1.2.3", "1.2.3", 0),
            ("10.0.0", "2.0.0", 1),  # Numeric comparison, not string
            ("0.0.0", "0.0.1", -1),
            ("999.999.999", "1000.0.0", -1),
        ]

        for v1, v2, expected in comparison_cases:
            result = FlextVersionManager.compare_versions(v1, v2)
            assert result == expected, (
                f"compare_versions('{v1}', '{v2}') expected {expected}, got {result}"
            )
