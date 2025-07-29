"""FLEXT Core Version Module.

Comprehensive version management system for the FLEXT Core library providing
standardized versioning information, compatibility checking, and version-dependent
feature availability detection.

Architecture:
    - Semantic versioning compliance with SemVer 2.0.0 specification
    - Single source of truth for version information across FLEXT ecosystem
    - Version comparison utilities and feature availability detection
    - Python compatibility validation with structured error reporting
    - Build metadata and release information management

Version Components:
    - __version__: Primary semantic version string (MAJOR.MINOR.PATCH)
    - Version metadata: Release name, date, and build type information
    - Compatibility matrix: Python version ranges and feature availability
    - Utility functions: Version comparison and format validation

Dependencies:
    - Standard library only: sys, typing for minimal footprint
    - No external runtime dependencies

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from flext_core.types import TAnyList

# =============================================================================
# VERSION INFORMATION - Single source of truth for FLEXT Core version
# =============================================================================

__version__ = "0.8.0"

# Version metadata for programmatic access
VERSION_MAJOR = 0
VERSION_MINOR = 8
VERSION_PATCH = 0

# Semantic version format constants
SEMVER_PARTS_COUNT = 3  # major.minor.patch

# Release information
RELEASE_NAME = "Foundation"
RELEASE_DATE = "2025-06-27"
BUILD_TYPE = "stable"

# Compatibility information
MIN_PYTHON_VERSION = (3, 13, 0)
MAX_PYTHON_VERSION = (3, 14, 0)

# Feature availability matrix
AVAILABLE_FEATURES = {
    "core_validation": True,
    "dependency_injection": True,
    "domain_driven_design": True,
    "railway_programming": True,
    "enterprise_logging": True,
    "performance_tracking": True,
    "advanced_decorators": True,
    "type_safety": True,
    "configuration_management": True,
    "plugin_architecture": False,  # Future release
    "event_sourcing": False,  # Future release
    "distributed_tracing": False,  # Future release
}


# =============================================================================
# VERSION UTILITIES - Programmatic version handling and compatibility
# =============================================================================


class FlextVersionInfo(NamedTuple):
    """Structured version information for programmatic access.

    Provides structured access to version components enabling version
    comparison, compatibility checking, and feature availability detection
    with type safety and clear semantics.

    Components:
        major: Major version number (breaking changes)
        minor: Minor version number (backward-compatible features)
        patch: Patch version number (backward-compatible fixes)
        release_name: Human-readable release identifier
        release_date: Release date in ISO format
        build_type: Build type (stable, beta, alpha, dev)

    Usage:
        version = flext_get_version_info()
        if version.major >= 1:
            print("Using stable API")

        if version >= VersionInfo(0, 8, 0, "", "", ""):
            enable_advanced_features()
    """

    major: int
    minor: int
    patch: int
    release_name: str
    release_date: str
    build_type: str


class FlextCompatibilityResult(NamedTuple):
    """Result of compatibility checking operations.

    Provides structured result for compatibility validation enabling
    clear error reporting and graceful handling of compatibility issues.

    Components:
        is_compatible: Boolean indicating compatibility status
        current_version: Current version being checked
        required_version: Required version for compatibility
        error_message: Descriptive error message if incompatible
        recommendations: List of recommended actions for resolution

    Usage:
        result = check_python_compatibility()
        if not result.is_compatible:
            print(f"Error: {result.error_message}")
            for rec in result.recommendations:
                print(f"  - {rec}")
    """

    is_compatible: bool
    current_version: tuple[int, ...]
    required_version: tuple[int, ...]
    error_message: str
    recommendations: TAnyList


def get_version_tuple() -> tuple[int, int, int]:
    """Get version as tuple for programmatic comparison.

    Returns structured version tuple enabling version comparison operations
    and compatibility checking with clear semantics and type safety.

    Returns:
        Tuple containing (major, minor, patch) version components

    Usage:
        version = flext_get_version_tuple()
        if version >= (1, 0, 0):
            use_stable_api()
        elif version >= (0, 8, 0):
            use_beta_features()

    """
    return (VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH)


def get_version_info() -> FlextVersionInfo:
    """Get comprehensive version information.

    Returns complete version information including metadata enabling
    detailed version analysis, compatibility checking, and feature
    availability detection.

    Returns:
        VersionInfo with complete version and metadata information

    Usage:
        info = get_version_info()
        print(f"FLEXT Core {info.major}.{info.minor}.{info.patch}")
        print(f"Release: {info.release_name} ({info.release_date})")
        print(f"Build: {info.build_type}")

    """
    return FlextVersionInfo(
        major=VERSION_MAJOR,
        minor=VERSION_MINOR,
        patch=VERSION_PATCH,
        release_name=RELEASE_NAME,
        release_date=RELEASE_DATE,
        build_type=BUILD_TYPE,
    )


def get_version_string() -> str:
    """Get formatted version string for display.

    Returns human-readable version string suitable for logging,
    user interfaces, and documentation with consistent formatting.

    Returns:
        Formatted version string with release information

    Usage:
        version_str = get_version_string()
        print(f"Starting {version_str}")
        logger.info(f"Application version: {version_str}")

    """
    info = get_version_info()
    return f"{__version__} ({info.release_name})"


def check_python_compatibility() -> FlextCompatibilityResult:
    """Check Python version compatibility.

    Validates current Python version against FLEXT Core requirements
    providing detailed compatibility information and actionable
    recommendations for resolution.

    Returns:
        FlextCompatibilityResult with compatibility status and recommendations

    Usage:
        compatibility = check_python_compatibility()
        if not compatibility.is_compatible:
            raise FlextOperationError(
                "Python compatibility error",
                operation_context={
                    "error_message": compatibility.error_message,
                    "current_python": str(sys.version_info[:3]),
                    "required_python": "3.13+",
                },
            )

    """
    current_version = sys.version_info[:3]

    if current_version < MIN_PYTHON_VERSION:
        return FlextCompatibilityResult(
            is_compatible=False,
            current_version=current_version,
            required_version=MIN_PYTHON_VERSION,
            error_message=(
                f"Python {'.'.join(map(str, current_version))} is too old. "
                f"Minimum required: {'.'.join(map(str, MIN_PYTHON_VERSION))}"
            ),
            recommendations=[
                f"Upgrade Python to {'.'.join(map(str, MIN_PYTHON_VERSION))} or newer",
                "Use pyenv or conda to manage multiple Python versions",
                "Check FLEXT documentation for installation guides",
            ],
        )

    if current_version >= MAX_PYTHON_VERSION:
        return FlextCompatibilityResult(
            is_compatible=False,
            current_version=current_version,
            required_version=MAX_PYTHON_VERSION,
            error_message=(
                f"Python {'.'.join(map(str, current_version))} is too new. "
                f"Maximum supported: {'.'.join(map(str, MAX_PYTHON_VERSION))}"
            ),
            recommendations=[
                f"Use Python {'.'.join(map(str, MIN_PYTHON_VERSION))}"
                f" to {'.'.join(map(str, MAX_PYTHON_VERSION))}",
                "Check for newer FLEXT Core version with broader Python support",
                "Use pyenv or conda to install compatible Python version",
            ],
        )

    return FlextCompatibilityResult(
        is_compatible=True,
        current_version=current_version,
        required_version=MIN_PYTHON_VERSION,
        error_message="",
        recommendations=[],
    )


def is_feature_available(feature_name: str) -> bool:
    """Check if a specific feature is available in current version.

    Enables feature-based conditional logic allowing graceful handling
    of version-dependent functionality with clear feature names.

    Args:
        feature_name: Name of feature to check availability

    Returns:
        True if feature is available, False otherwise

    Usage:
        if is_feature_available("advanced_validation"):
            enable_advanced_validation()
        else:
            use_basic_validation()

    """
    return AVAILABLE_FEATURES.get(feature_name, False)


def get_available_features() -> TAnyList:
    """Get list of available features in current version.

    Returns list of feature names available in current version enabling
    dynamic feature discovery and capability reporting.

    Returns:
        List of available feature names

    Usage:
        features = get_available_features()
        print("Available features:")
        for feature in features:
            print(f"  - {feature}")

    """
    return [name for name, available in AVAILABLE_FEATURES.items() if available]


def compare_versions(version1: str, version2: str) -> int:
    """Compare two semantic version strings.

    Compares semantic version strings following SemVer 2.0.0 specification
    returning standard comparison result for version ordering.

    Args:
        version1: First version string to compare
        version2: Second version string to compare

    Returns:
        -1 if version1 < version2, 0 if equal, 1 if version1 > version2

    Usage:
        result = compare_versions("0.8.0", "1.0.0")
        if result < 0:
            print("First version is older")

    """

    def parse_version(version: str) -> tuple[int, ...]:
        """Parse version string into comparable tuple."""
        return tuple(int(part) for part in version.split("."))

    v1_tuple = parse_version(version1)
    v2_tuple = parse_version(version2)

    if v1_tuple < v2_tuple:
        return -1
    if v1_tuple > v2_tuple:
        return 1
    return 0


def validate_version_format(version: str) -> bool:
    """Validate semantic version string format.

    Validates version string against semantic versioning format requirements
    ensuring compliance with SemVer 2.0.0 specification.

    Args:
        version: Version string to validate

    Returns:
        True if version format is valid, False otherwise

    Usage:
        if validate_version_format("1.2.3"):
            process_version("1.2.3")
        else:
            raise FlextValidationError(
                "Invalid version format",
                validation_details={
                    "field": "version",
                    "value": version,
                    "rules": ["semver_format"],
                    "expected_format": "MAJOR.MINOR.PATCH (e.g., 1.2.3)",
                },
            )

    """
    try:
        parts = version.split(".")
        if len(parts) != SEMVER_PARTS_COUNT:
            return False

        for part in parts:
            if not part.isdigit():
                return False
            if int(part) < 0:
                return False
    except (ValueError, AttributeError):
        return False
    else:
        return True


# =============================================================================
# EXPORTS - Clean public API
# =============================================================================

__all__ = [
    "AVAILABLE_FEATURES",
    "BUILD_TYPE",
    "MAX_PYTHON_VERSION",
    "MIN_PYTHON_VERSION",
    "RELEASE_DATE",
    "RELEASE_NAME",
    "VERSION_MAJOR",
    "VERSION_MINOR",
    "VERSION_PATCH",
    "FlextCompatibilityResult",
    "FlextVersionInfo",
    "__version__",
    "check_python_compatibility",
    "compare_versions",
    "get_available_features",
    "get_version_info",
    "get_version_string",
    "get_version_tuple",
    "is_feature_available",
    "validate_version_format",
]
