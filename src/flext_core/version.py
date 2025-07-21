"""Centralized version management system for FLEXT ecosystem.

This module provides the single source of truth for versioning across all FLEXT projects.
All FLEXT modules should import their version information from this central system.

Zero tolerance for duplicate version files across the workspace.
"""

from __future__ import annotations

from typing import NamedTuple


class VersionInfo(NamedTuple):
    """Version information structure."""

    major: int
    minor: int
    patch: int
    pre: str | None = None
    build: str | None = None

    def __str__(self) -> str:
        """Return version as string."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.pre:
            version += f"-{self.pre}"
        if self.build:
            version += f"+{self.build}"
        return version

# FLEXT Core Framework Version
FLEXT_CORE_VERSION = "0.7.0"
FLEXT_CORE_VERSION_INFO = VersionInfo(0, 7, 0)

# FLEXT Ecosystem Versions - Single source of truth
FLEXT_VERSIONS: dict[str, str] = {
    # Core framework
    "flext-core": "0.7.0",
    "flext-api": "0.7.0",
    "flext-auth": "0.7.0",
    "flext-cli": "0.7.0",
    "flext-grpc": "0.7.0",
    "flext-web": "0.7.0",
    "flext-observability": "0.7.0",
    "flext-plugin": "0.7.0",
    "flext-meltano": "0.7.0",

    # Database and Oracle integration
    "flext-db-oracle": "0.7.0",
    "flext-oracle-oic-ext": "0.7.0",
    "flext-oracle-wms": "0.7.0",

    # LDAP and LDIF
    "flext-ldap": "0.7.0",
    "flext-ldif": "0.7.0",

    # Singer Taps
    "flext-tap-ldap": "0.7.0",
    "flext-tap-ldif": "0.7.0",
    "flext-tap-oracle": "0.7.0",
    "flext-tap-oracle-oic": "0.7.0",
    "flext-tap-oracle-wms": "0.7.0",

    # Singer Targets
    "flext-target-ldap": "0.7.0",
    "flext-target-ldif": "0.7.0",
    "flext-target-oracle": "0.7.0",
    "flext-target-oracle-oic": "0.7.0",
    "flext-target-oracle-wms": "0.7.0",

    # dbt Adapters
    "flext-dbt-ldap": "0.7.0",
    "flext-dbt-ldif": "0.7.0",
    "flext-dbt-oracle": "0.7.0",
    "flext-dbt-oracle-wms": "0.7.0",

    # Quality and tools
    "flext-quality": "0.7.0",

    # Enterprise applications
    "algar-oud-mig": "1.0.0",
    "gruponos-meltano-native": "1.0.0",
}

def get_version(project_name: str) -> str:
    """Get version for a specific FLEXT project.

    Args:
        project_name: Name of the FLEXT project

    Returns:
        Version string for the project

    Raises:
        ValueError: If project name is not recognized

    """
    if project_name not in FLEXT_VERSIONS:
        raise ValueError(f"Unknown FLEXT project: {project_name}")
    return FLEXT_VERSIONS[project_name]

def get_version_info(project_name: str) -> VersionInfo:
    """Get version info for a specific FLEXT project.

    Args:
        project_name: Name of the FLEXT project

    Returns:
        VersionInfo tuple for the project

    """
    version_str = get_version(project_name)
    parts = version_str.split(".")
    return VersionInfo(
        major=int(parts[0]),
        minor=int(parts[1]),
        patch=int(parts[2]),
    )

def get_all_versions() -> dict[str, str]:
    """Get all FLEXT project versions.

    Returns:
        Dictionary mapping project names to version strings

    """
    return FLEXT_VERSIONS.copy()

def update_version(project_name: str, new_version: str) -> None:
    """Update version for a FLEXT project (development use only).

    Args:
        project_name: Name of the FLEXT project
        new_version: New version string

    Raises:
        ValueError: If project name is not recognized

    """
    if project_name not in FLEXT_VERSIONS:
        raise ValueError(f"Unknown FLEXT project: {project_name}")
    FLEXT_VERSIONS[project_name] = new_version

__all__ = [
    "FLEXT_CORE_VERSION",
    "FLEXT_CORE_VERSION_INFO",
    "FLEXT_VERSIONS",
    "VersionInfo",
    "get_all_versions",
    "get_version",
    "get_version_info",
    "update_version",
]
