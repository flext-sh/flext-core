"""FLEXT Core Version Module.

Comprehensive version management system for the FLEXT Core library providing
standardized versioning information, compatibility checking, and version-dependent
feature availability detection.

Architecture:
    - Semantic versioning compliance (MAJOR.MINOR.PATCH) following SemVer 2.0.0
    - Single source of truth for version information across the FLEXT ecosystem
    - Version comparison utilities for compatibility validation
    - Feature availability detection based on version constraints
    - Build metadata and pre-release identifier support
    - Integration with package management and distribution systems

Version Information Components:
    - Primary version: Standard semantic version string (MAJOR.MINOR.PATCH)
    - Version tuple: Structured version components for programmatic access
    - Version metadata: Build information, release date, and compatibility data
    - Feature flags: Version-dependent functionality availability
    - Compatibility matrix: Supported Python versions and dependency ranges
    - Release information: Change log references and migration guidance

Maintenance Guidelines:
    - Follow Semantic Versioning 2.0.0 specification strictly
    - Update version information in single location only
    - Maintain backward compatibility within MINOR version increments
    - Document breaking changes requiring MAJOR version increments
    - Update compatibility information with each release
    - Validate version format and component ranges

Design Decisions:
    - Single source of truth pattern preventing version inconsistencies
    - Semantic versioning for clear compatibility communication
    - Programmatic access through structured version components
    - Integration with Python packaging standards (PEP 440)
    - Feature availability detection for graceful degradation
    - Build metadata support for development and CI/CD workflows

Version Format Compliance:
    - SemVer 2.0.0: MAJOR.MINOR.PATCH format with optional pre-release and build metadata
    - PEP 440: Python-specific version identification and comparison
    - FLEXT conventions: Ecosystem-wide version coordination and compatibility
    - Release cycles: Regular release schedule with predictable version increments

Version Usage Patterns:
    - Package distribution: Version information for PyPI and package managers
    - API compatibility: Version-based feature detection and graceful degradation
    - Dependency management: Version constraints for compatible package ranges
    - Runtime validation: Version compatibility checking during initialization
    - Documentation: Version-specific documentation and migration guides

Compatibility Management:
    - Python version compatibility: Minimum and maximum supported Python versions
    - Dependency compatibility: Compatible ranges for required and optional dependencies
    - API compatibility: Breaking change tracking and migration paths
    - Feature deprecation: Version-based deprecation warnings and removal schedules
    - Cross-package compatibility: Version coordination across FLEXT ecosystem

Usage Patterns:
    # Basic version access
    from flext_core.version import __version__
    print(f"FLEXT Core v{__version__}")

    # Programmatic version handling
    version_tuple = get_version_tuple()
    if version_tuple >= (1, 0, 0):
        use_stable_api()

    # Feature availability checking
    if is_feature_available("advanced_validation"):
        enable_advanced_features()

    # Compatibility validation
    compatibility = check_python_compatibility()
    if not compatibility.is_compatible:
        raise RuntimeError(compatibility.error_message)

Dependencies:
    - Standard library: sys for Python version detection
    - typing: Type annotations and version comparison utilities
    - No external runtime dependencies for version management

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from typing import NamedTuple

# =============================================================================
# VERSION INFORMATION - Single source of truth for FLEXT Core version
# =============================================================================

__version__ = "0.8.0"

# Version metadata for programmatic access
VERSION_MAJOR = 0
VERSION_MINOR = 8
VERSION_PATCH = 0

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


class VersionInfo(NamedTuple):
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
        version = get_version_info()
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


class CompatibilityResult(NamedTuple):
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
    recommendations: list[str]


def get_version_tuple() -> tuple[int, int, int]:
    """Get version as tuple for programmatic comparison.

    Returns structured version tuple enabling version comparison operations
    and compatibility checking with clear semantics and type safety.

    Returns:
        Tuple containing (major, minor, patch) version components

    Usage:
        version = get_version_tuple()
        if version >= (1, 0, 0):
            use_stable_api()
        elif version >= (0, 8, 0):
            use_beta_features()

    """
    return (VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH)


def get_version_info() -> VersionInfo:
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
    return VersionInfo(
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


def check_python_compatibility() -> CompatibilityResult:
    """Check Python version compatibility.

    Validates current Python version against FLEXT Core requirements
    providing detailed compatibility information and actionable
    recommendations for resolution.

    Returns:
        CompatibilityResult with compatibility status and recommendations

    Usage:
        compatibility = check_python_compatibility()
        if not compatibility.is_compatible:
            raise RuntimeError(f"Python compatibility error: {compatibility.error_message}")

    """
    current_version = sys.version_info[:3]

    if current_version < MIN_PYTHON_VERSION:
        return CompatibilityResult(
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
        return CompatibilityResult(
            is_compatible=False,
            current_version=current_version,
            required_version=MAX_PYTHON_VERSION,
            error_message=(
                f"Python {'.'.join(map(str, current_version))} is too new. "
                f"Maximum supported: {'.'.join(map(str, MAX_PYTHON_VERSION))}"
            ),
            recommendations=[
                f"Use Python {'.'.join(map(str, MIN_PYTHON_VERSION))} to {'.'.join(map(str, MAX_PYTHON_VERSION))}",
                "Check for newer FLEXT Core version with broader Python support",
                "Use pyenv or conda to install compatible Python version",
            ],
        )

    return CompatibilityResult(
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


def get_available_features() -> list[str]:
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
            raise ValueError("Invalid version format")

    """
    try:
        parts = version.split(".")
        if len(parts) != 3:
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
# EXPORTS - Clean public API for version information
# =============================================================================

__all__ = [
    "AVAILABLE_FEATURES",
    "BUILD_TYPE",
    "MAX_PYTHON_VERSION",
    # Compatibility information
    "MIN_PYTHON_VERSION",
    "RELEASE_DATE",
    "RELEASE_NAME",
    "VERSION_MAJOR",
    "VERSION_MINOR",
    "VERSION_PATCH",
    "CompatibilityResult",
    # Version data structures
    "VersionInfo",
    # Version constants
    "__version__",
    "check_python_compatibility",
    "compare_versions",
    "get_available_features",
    "get_version_info",
    "get_version_string",
    # Version utilities
    "get_version_tuple",
    "is_feature_available",
    "validate_version_format",
]
