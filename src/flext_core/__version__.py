"""Version management and compatibility checking for FLEXT Core."""

import sys
from importlib.metadata import version as _pkg_version
from typing import NamedTuple

# from flext_core.typings import FlextTypes  # Avoid circular import

__version__: str = _pkg_version("flext-core")

# Version metadata for programmatic access
VERSION_MAJOR: int = 0
VERSION_MINOR: int = 9
VERSION_PATCH: int = 0

# Semantic version format constants
SEMVER_PARTS_COUNT: int = 3  # major.minor.patch

# Release information
RELEASE_NAME: str = "Foundation"
RELEASE_DATE: str = "2025-06-27"
BUILD_TYPE: str = "stable"

# Compatibility information
MIN_PYTHON_VERSION: tuple[int, int, int] = (3, 13, 0)
MAX_PYTHON_VERSION: tuple[int, int, int] = (3, 14, 0)

# Feature availability matrix
AVAILABLE_FEATURES: dict[str, bool] = {
    "core_validation": True,
    "dependency_injection": True,
    "domain_driven_design": True,
    "railway_programming": True,
    "enterprise_logging": True,
    "performance_tracking": True,
    "decorators": True,
    "type_safety": True,
    "configuration_management": True,
    "plugin_architecture": False,  # Future release
    "event_sourcing": False,  # Future release
    "distributed_tracing": False,  # Future release
}


# =============================================================================
# VERSION UTILITIES - Programmatic version handling and compatibility
# =============================================================================


class FlextVersionManager:
    """Single consolidated class for all version management functionality.

    Consolidates ALL version-related classes and operations into one class
    following FLEXT patterns. Provides version information, compatibility
    checking, and utility functions.
    """

    class VersionInfo(NamedTuple):
        """Structured version information nested inside consolidated class."""

        major: int
        minor: int
        patch: int
        release_name: str
        release_date: str
        build_type: str

    class CompatibilityResult:
        """Result of Python version compatibility check."""

        def __init__(
            self,
            *,
            is_compatible: bool,
            current_version: tuple[int, ...],
            required_version: tuple[int, ...],
            error_message: str,
            recommendations: list[str],
        ) -> None:
            self.is_compatible = is_compatible
            self.current_version = current_version
            self.required_version = required_version
            self.error_message = error_message
            self.recommendations = recommendations


# Backward compatibility aliases
FlextVersionInfo = FlextVersionManager.VersionInfo
FlextCompatibilityResult = FlextVersionManager.CompatibilityResult


def get_version_tuple() -> tuple[int, int, int]:
    """Get a version as tuple for programmatic comparison.

    Returns:
      Tuple containing (major, minor, patch) version components.

    """
    return VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH


def get_version_info() -> FlextVersionManager.VersionInfo:
    """Get comprehensive version information.

    Returns:
      FlextVersionManager.VersionInfo with complete version and metadata information.

    """
    return FlextVersionManager.VersionInfo(
        major=VERSION_MAJOR,
        minor=VERSION_MINOR,
        patch=VERSION_PATCH,
        release_name=RELEASE_NAME,
        release_date=RELEASE_DATE,
        build_type=BUILD_TYPE,
    )


def get_version_string() -> str:
    """Get formatted version string for display.

    Returns:
      Formatted version string with release information.

    """
    info = get_version_info()
    return f"{__version__} ({info.release_name})"


def check_python_compatibility() -> FlextCompatibilityResult:
    """Check Python version compatibility.

    Returns:
      FlextCompatibilityResult with compatibility status and recommendations.

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
                (
                    f"Use Python {'.'.join(map(str, MIN_PYTHON_VERSION))} "
                    f"to {'.'.join(map(str, MAX_PYTHON_VERSION))}"
                ),
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
    """Check if a specific feature is available in a current version.

    Enables feature-based conditional logic allowing graceful handling
    of version-dependent functionality with clear feature names.

    Args:
      feature_name: Name of feature to check availability

    Returns:
      True if feature is available, False otherwise.

    """
    return AVAILABLE_FEATURES.get(feature_name, False)


def get_available_features() -> list[str]:
    """Get a list of available features in the current version.

    Returns a list of feature names available in the current version enabling
    dynamic feature discovery and capability reporting.

    Returns:
      List of available feature names.

    """
    return [name for name, available in AVAILABLE_FEATURES.items() if available]


def compare_versions(version1: str, version2: str) -> int:
    """Compare two semantic version strings.

    Compares semantic version strings following SemVer 2.0.0 specification
    returning a standard comparison result for version ordering.

    Args:
      version1: First version string to compare
      version2: Second version string to compare

    Returns:
      -1 of version1 < version2, 0 if equal, 1 of version1 > version2.

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
    """Validate a semantic version string format.

    Validates version string against semantic versioning format requirements
    ensuring compliance with SemVer 2.0.0 specification.

    Args:
      version: Version string to validate

    Returns:
      True if a version format is valid, False otherwise.

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
        # Validation failed but maintain API contract
        return False
    else:
        return True


# =============================================================================
# EXPORTS - Clean public API
# =============================================================================

__all__: list[str] = [
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
