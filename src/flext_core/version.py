"""Version management utilities tracking the FLEXT-Core 1.0.0 programme.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar, NamedTuple

from flext_core.constants import FlextConstants

__version__: str = FlextConstants.Core.VERSION


class FlextVersionManager:
    """Single consolidated class for all version management functionality.

    The helpers drive documentation badges, compatibility checks, and feature
    gates tracked as part of the 1.0.0 release roadmap.
    """

    # Version constants consolidated within the class - derived from FlextConstants
    VERSION_MAJOR: int = int(FlextConstants.Core.VERSION.split(".")[0])
    VERSION_MINOR: int = int(FlextConstants.Core.VERSION.split(".")[1])
    VERSION_PATCH: int = int(FlextConstants.Core.VERSION.split(".")[2])
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
        "structured_logging": True,
        "performance_tracking": True,
        "decorators": True,
        "type_safety": True,
        "configuration_management": True,
        "plugin_architecture": False,  # Future release
        "event_sourcing": False,  # Future release
        "distributed_tracing": False,  # Future release
    }

    class VersionInfo(NamedTuple):
        """Structured version metadata used in release communications."""

        major: int
        minor: int
        patch: int
        release_name: str
        release_date: str
        build_type: str

    class CompatibilityResult:
        """Python compatibility outcome backing modernization guarantees."""

        def __init__(
            self,
            *,
            is_compatible: bool,
            current_version: tuple[int, ...],
            required_version: tuple[int, ...],
            error_message: str,
            recommendations: list[str],
        ) -> None:
            """Capture modernization-focused compatibility findings."""
            self.is_compatible = is_compatible
            self.current_version = current_version
            self.required_version = required_version
            self.error_message = error_message
            self.recommendations = recommendations

    @staticmethod
    def get_version_tuple() -> tuple[int, int, int]:
        """Return the semantic version tuple used across 1.0.0 tooling.

        Returns:
            tuple[int, int, int]: Version tuple (major, minor, patch)

        """
        return (
            FlextVersionManager.VERSION_MAJOR,
            FlextVersionManager.VERSION_MINOR,
            FlextVersionManager.VERSION_PATCH,
        )

    @staticmethod
    def get_version_info() -> FlextVersionManager.VersionInfo:
        """Return the structured release payload surfaced in documentation.

        Returns:
            FlextVersionManager.VersionInfo: Structured version metadata

        """
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
        """Render the version banner consumed by README and docs badges.

        Returns:
            str: Version string with release name

        """
        info = FlextVersionManager.get_version_info()
        return f"{__version__} ({info.release_name})"

    @staticmethod
    def get_available_features() -> list[str]:
        """List features flagged as ready for the 1.0.0 modernization cycle.

        Returns:
            list[str]: List of available feature names

        """
        return [
            name
            for name, available in FlextVersionManager.AVAILABLE_FEATURES.items()
            if available
        ]

    @staticmethod
    def compare_versions(version1: str, version2: str) -> int:
        """Compare semantic versions to inform upgrade guidance.

        Returns:
            int: -1 if version1 < version2, 0 if equal, 1 if version1 > version2

        """

        def parse_version(version: str) -> tuple[int, ...]:
            """Parse a version string into comparison-friendly tuple form.

            Returns:
                tuple[int, ...]: Version parts as integers

            """
            return tuple(int(part) for part in version.split("."))

        v1_tuple = parse_version(version1)
        v2_tuple = parse_version(version2)

        if v1_tuple < v2_tuple:
            return -1
        if v1_tuple > v2_tuple:
            return 1
        return 0

    @staticmethod
    def validate_version_format(version: object) -> bool:
        """Validate semantic version formatting for roadmap compliance.

        Returns:
            bool: True if version format is valid, False otherwise

        """
        try:
            if not isinstance(version, str):
                return False
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
