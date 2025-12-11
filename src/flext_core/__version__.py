"""FlextVersion - Package Version and Metadata Information Module.

This module provides version and package metadata for flext-core using
importlib.metadata, extracting information from the package's metadata
including version, author, license, and other package details. Implements
structural typing via p.Version through duck typing,
providing a foundation for version management and package information access.

Scope: Package metadata extraction, version string and tuple representation,
PEP 440 semantic versioning compliance, metadata access for package information,
version comparison utilities, and package identity management.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from importlib.metadata import PackageMetadata, PackageNotFoundError, metadata
from typing import cast

from flext_core.typings import t


class FlextVersion:
    """Package version and metadata information for FLEXT ecosystem.

    Provides comprehensive package metadata extraction and version management
    using importlib.metadata. Implements structural typing via p.Version
    through duck typing (no inheritance required), serving as the foundation layer
    (Layer 0) for version information throughout the FLEXT ecosystem.

    Core Features:
    - PEP 440 semantic versioning compliance
    - Package metadata extraction from distribution metadata
    - Version string and tuple representations for different use cases
    - Graceful fallback handling for missing metadata fields
    - Type-safe metadata access with proper error handling
    - Zero external dependencies (stdlib only)

    Architecture:
    - Single class with nested metadata extraction logic
    - DRY principle applied through centralized metadata access
    - SOLID principles: Single Responsibility for version/metadata management
    - Railway pattern for error handling in metadata extraction
    - Structural typing for protocol compliance without inheritance

    Version Formats:
    - String format: "major.minor.patch" (e.g., "1.0.0") for display
    - Tuple format: (major, minor, patch, ...) for numeric comparison
    - Supports pre-release versions: "1.0.0rc1", "1.0.0a1", "1.0.0b1"
    - Compatible with PEP 440 semantic versioning specification

    Usage Examples:
        >>> from flext_core import __version__, __version_info__
        >>> print(f"flext-core {__version__}")  # "flext-core 1.0.0"
        >>> if __version_info__ >= (1, 0, 0):  # Version comparison
        ...     enable_new_feature()
        >>> assert __version_info__[0] >= 0  # Runtime version validation
    """

    # Package metadata extraction with error handling
    try:
        _metadata = metadata("flext-core")
    except PackageNotFoundError:
        # Create PackageMetadata-compatible dict and cast to PackageMetadata
        _metadata_dict: t.StringDict = {
            "Version": "0.0.0-dev",
            "Name": "flext-core",
            "Summary": "FLEXT core (metadata fallback)",
            "Author": "",
            "Author-Email": "",
            "License": "",
            "Home-Page": "",
        }
        # Cast dict to PackageMetadata for type compatibility
        _metadata = cast("PackageMetadata", _metadata_dict)

    # Core version information - extracted once at import time
    __version__ = _metadata["Version"]
    __version_info__ = tuple(
        int(part) if part.isdigit() else part for part in __version__.split(".")
    )

    # Package identity information
    __title__ = _metadata["Name"]
    __description__ = _metadata.get("Summary", "")
    __author__ = _metadata.get("Author", "")
    __author_email__ = _metadata.get("Author-Email", "")
    __license__ = _metadata.get("License", "")
    __url__ = _metadata.get("Home-Page", "")

    @classmethod
    def get_version_string(cls) -> str:
        """Get package version as human-readable string.

        Returns the package version in string format suitable for display
        and logging. Follows PEP 440 semantic versioning format.

        Returns:
            str: Version string (e.g., "1.0.0", "1.0.0rc1")

        Example:
            >>> version = FlextVersion.get_version_string()
            >>> print(f"Current version: {version}")
            Current version: 1.0.0

        """
        return cls.__version__

    @classmethod
    def get_version_info(cls) -> tuple[int | str, ...]:
        """Get package version as comparison-friendly tuple.

        Returns version information as a tuple for easy numeric comparison.
        Each element is either an integer (for numeric version parts) or
        string (for pre-release identifiers).

        Returns:
            tuple[int | str, ...]: Version tuple for comparison (e.g., (1, 0, 0))

        Example:
            >>> vinfo = FlextVersion.get_version_info()
            >>> if vinfo >= (1, 0, 0):
            ...     print("Version 1.0.0 or higher")

        """
        return cls.__version_info__

    @classmethod
    def is_version_at_least(cls, major: int, minor: int = 0, patch: int = 0) -> bool:
        """Check if current version meets minimum version requirement.

        Performs version comparison to determine if the current package version
        is at least the specified minimum version. Useful for feature gating
        and compatibility checks.

        Args:
            major: Minimum major version number
            minor: Minimum minor version number (default: 0)
            patch: Minimum patch version number (default: 0)

        Returns:
            bool: True if current version >= specified version

        Example:
            >>> if FlextVersion.is_version_at_least(1, 0, 0):
            ...     enable_new_api()

        """
        return cls.__version_info__ >= (major, minor, patch)

    @classmethod
    def get_package_info(cls) -> t.StringDict:
        """Get comprehensive package information dictionary.

        Returns all available package metadata in a structured dictionary
        format for programmatic access to package information.

        Returns:
            dict[str, str]: Package metadata dictionary containing:
                - name: Package name
                - version: Version string
                - description: Package description
                - author: Author name
                - author_email: Author email
                - license: License type
                - url: Homepage URL

        Example:
            >>> info = FlextVersion.get_package_info()
            >>> print(f"Package: {info['name']} v{info['version']}")

        """
        return {
            "name": cls.__title__,
            "version": cls.__version__,
            "description": cls.__description__,
            "author": cls.__author__,
            "author_email": cls.__author_email__,
            "license": cls.__license__,
            "url": cls.__url__,
        }


# Module-level exports for external access
__version__ = FlextVersion.__version__
__version_info__ = FlextVersion.__version_info__
__title__ = FlextVersion.__title__
__description__ = FlextVersion.__description__
__author__ = FlextVersion.__author__
__author_email__ = FlextVersion.__author_email__
__license__ = FlextVersion.__license__
__url__ = FlextVersion.__url__

__all__ = [
    "FlextVersion",
    "__author__",
    "__author_email__",
    "__description__",
    "__license__",
    "__title__",
    "__url__",
    "__version__",
    "__version_info__",
]
