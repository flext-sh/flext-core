"""Version management and compatibility checking for FLEXT Core.

This module provides efficient version management functionality for the FLEXT ecosystem,
including semantic versioning, compatibility checking, feature availability tracking,
and programmatic version utilities following enterprise standards.

Module Organization:
    Version Constants: __version__, VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH
    Release Information: RELEASE_NAME, RELEASE_DATE, BUILD_TYPE
    Compatibility: MIN_PYTHON_VERSION, MAX_PYTHON_VERSION
    Feature Tracking: AVAILABLE_FEATURES dictionary with feature flags
    Manager Classes: FlextModels with nested utilities
    Utility Functions: Version comparison, validation, compatibility checking

Classes:
    FlextModels: Consolidated version management functionality
        └── VersionInfo(NamedTuple): Structured version information
            • major: int - Major version number
            • minor: int - Minor version number
            • patch: int - Patch version number
            • release_name: str - Human-readable release name
            • release_date: str - ISO formatted release date
            • build_type: str - Build type identifier
        └── CompatibilityResult: Python compatibility check results
            • is_compatible: bool - Compatibility status
            • current_version: tuple[int, ...] - Current Python version
            • required_version: tuple[int, ...] - Required Python version
            • error_message: str - Descriptive error message
            • recommendations: list[str] - Upgrade recommendations

Functions:
    get_version_tuple() -> tuple[int, int, int]
        Get semantic version as tuple for programmatic comparison

    get_version_info() -> FlextModels.VersionInfo
        Get efficient version information with metadata

    get_version_string() -> str
        Get formatted version string with release information

    check_python_compatibility() -> FlextCompatibilityResult
        Check current Python version compatibility with requirements

    is_feature_available(feature_name: str) -> bool
        Check if named feature is available in current version

    get_available_features() -> list[str]
        Get list of all available feature names

    compare_versions(version1: str, version2: str) -> int
        Compare two semantic version strings (-1, 0, 1)

    validate_version_format(version: str) -> bool
        Validate semantic version string format compliance

Constants:
    __version__: str - Package version from metadata
    VERSION_MAJOR/MINOR/PATCH: int - Semantic version components
    RELEASE_NAME: str - Human-readable release identifier
    RELEASE_DATE: str - ISO formatted release date
    BUILD_TYPE: str - Build type (stable, beta, alpha)
    MIN/MAX_PYTHON_VERSION: tuple[int, int, int] - Supported Python range
    AVAILABLE_FEATURES: dict[str, bool] - Feature availability matrix

Integration with FlextCore:
    >>> from flext_core import get_version_info, check_python_compatibility
    >>> from flext_core.core import FlextCore
    >>> # Version-aware initialization
    >>> core = FlextCore.get_instance()
    >>> version_info = get_version_info()
    >>> core.logger.info(
    ...     f"FLEXT Core {version_info.major}.{version_info.minor} initialized"
    ... )
    >>> # Compatibility validation
    >>> compat_result = check_python_compatibility()
    >>> if not compat_result.is_compatible:
    >>>     core.exceptions.raise_configuration_error(compat_result.error_message)
    >>> # Feature-based conditional logic
    >>> if is_feature_available("distributed_tracing"):
    >>>     core.observability.enable_distributed_tracing()

Feature Management Examples:
    >>> # Check enterprise features
    >>> enterprise_features = [
    ...     "enterprise_logging",
    ...     "performance_tracking",
    ...     "type_safety",
    ... ]
    >>> available = [f for f in enterprise_features if is_feature_available(f)]
    >>> print(f"Available enterprise features: {available}")
    >>> # Version comparison for migration
    >>> current = get_version_string()
    >>> target = "1.0.0"
    >>> if compare_versions(current, target) < 0:
    >>>     print(f"Migration available: {current} -> {target}")

Notes:
    - All version utilities follow SemVer 2.0.0 specification
    - Feature flags enable graceful degradation across versions
    - Compatibility checking supports automated environment validation
    - Version management integrates with FlextCore for centralized logging
    - Backward compatibility maintained through carefully designed aliases
    - Module avoids circular imports by defining constants locally

"""

from typing import ClassVar, NamedTuple

__version__: str = "0.9.0"


# =============================================================================
# VERSION UTILITIES - Programmatic version handling and compatibility
# =============================================================================


class FlextVersionManager:
    """Single consolidated class for all version management functionality.

    Consolidates ALL version-related classes and operations into one class
    following FLEXT patterns. Provides version information, compatibility
    checking, and utility functions.
    """

    # Version constants consolidated within the class
    VERSION_MAJOR: int = 0
    VERSION_MINOR: int = 9
    VERSION_PATCH: int = 0
    SEMVER_PARTS_COUNT: int = 3  # major.minor.patch

    # Release information
    RELEASE_NAME: str = "Foundation"
    RELEASE_DATE: str = "2025-06-27"
    BUILD_TYPE: str = "stable"

    MIN_PYTHON_VERSION: tuple[int, int, int] = (3, 13, 0)
    MAX_PYTHON_VERSION: tuple[int, int, int] = (3, 14, 0)

    # Feature availability matrix
    AVAILABLE_FEATURES: ClassVar[dict[str, bool]] = {
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

    @staticmethod
    def get_version_tuple() -> tuple[int, int, int]:
        """Get a version as tuple for programmatic comparison."""
        return (
            FlextVersionManager.VERSION_MAJOR,
            FlextVersionManager.VERSION_MINOR,
            FlextVersionManager.VERSION_PATCH,
        )

    @staticmethod
    def get_version_info() -> "FlextVersionManager.VersionInfo":
        """Get efficient version information."""
        return FlextVersionManager.VersionInfo(
            major=FlextVersionManager.VERSION_MAJOR,
            minor=FlextVersionManager.VERSION_MINOR,
            patch=FlextVersionManager.VERSION_PATCH,
            release_name=FlextVersionManager.RELEASE_NAME,
            release_date=FlextVersionManager.RELEASE_DATE,
            build_type=FlextVersionManager.BUILD_TYPE,
        )

    @staticmethod
    def get_version_string() -> str:
        """Get formatted version string for display."""
        info = FlextVersionManager.get_version_info()
        return f"{__version__} ({info.release_name})"

    @staticmethod
    def check_python_compatibility() -> "FlextVersionManager.CompatibilityResult":
        """Check Python version compatibility."""
        import sys

        current_version = sys.version_info[:3]

        if current_version < FlextVersionManager.MIN_PYTHON_VERSION:
            return FlextVersionManager.CompatibilityResult(
                is_compatible=False,
                current_version=current_version,
                required_version=FlextVersionManager.MIN_PYTHON_VERSION,
                error_message=(
                    f"Python {'.'.join(map(str, current_version))} is too old. "
                    f"Minimum required: {'.'.join(map(str, FlextVersionManager.MIN_PYTHON_VERSION))}"
                ),
                recommendations=[
                    f"Upgrade Python to {'.'.join(map(str, FlextVersionManager.MIN_PYTHON_VERSION))} or newer",
                    "Use pyenv or conda to manage multiple Python versions",
                    "Check FLEXT documentation for installation guides",
                ],
            )

        if current_version >= FlextVersionManager.MAX_PYTHON_VERSION:
            return FlextVersionManager.CompatibilityResult(
                is_compatible=False,
                current_version=current_version,
                required_version=FlextVersionManager.MAX_PYTHON_VERSION,
                error_message=(
                    f"Python {'.'.join(map(str, current_version))} is too new. "
                    f"Maximum supported: {'.'.join(map(str, FlextVersionManager.MAX_PYTHON_VERSION))}"
                ),
                recommendations=[
                    (
                        f"Use Python {'.'.join(map(str, FlextVersionManager.MIN_PYTHON_VERSION))} "
                        f"to {'.'.join(map(str, FlextVersionManager.MAX_PYTHON_VERSION))}"
                    ),
                    "Check for newer FLEXT Core version with broader Python support",
                    "Use pyenv or conda to install compatible Python version",
                ],
            )

        return FlextVersionManager.CompatibilityResult(
            is_compatible=True,
            current_version=current_version,
            required_version=FlextVersionManager.MIN_PYTHON_VERSION,
            error_message="",
            recommendations=[],
        )

    @staticmethod
    def is_feature_available(feature_name: str) -> bool:
        """Check if a specific feature is available in the current version."""
        return FlextVersionManager.AVAILABLE_FEATURES.get(feature_name, False)

    @staticmethod
    def get_available_features() -> list[str]:
        """Get a list of available features in the current version."""
        return [
            name
            for name, available in FlextVersionManager.AVAILABLE_FEATURES.items()
            if available
        ]

    @staticmethod
    def compare_versions(version1: str, version2: str) -> int:
        """Compare two semantic version strings."""

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

    @staticmethod
    def validate_version_format(version: str) -> bool:
        """Validate a semantic version string format."""
        try:
            parts = version.split(".")
            if len(parts) != FlextVersionManager.SEMVER_PARTS_COUNT:
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
# FLEXT PATTERN COMPLIANCE - All functionality within FlextVersionManager class
# Zero tolerance for loose CONSTANTS outside classes
# =============================================================================

__all__: list[str] = [
    "FlextVersionManager",
    "__version__",
]
