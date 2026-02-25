"""Package version and metadata information.

Provides version information and package metadata for the flext-infra package
using standard library metadata extraction.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib.metadata import PackageMetadata, PackageNotFoundError, metadata


class FlextInfraVersion:
    """Package version and metadata information.

    Provides version information and package metadata using standard library
    metadata extraction.
    """

    _metadata: PackageMetadata | Mapping[str, str]
    try:
        _metadata = metadata("flext-infra")
    except PackageNotFoundError:
        _metadata = {
            "Version": "0.12.0-dev",
            "Name": "flext-infra",
            "Summary": "FLEXT infrastructure",
            "Author": "",
            "Author-Email": "",
            "License": "",
            "Home-Page": "",
        }

    __version__ = _metadata["Version"]
    __version_info__ = tuple(
        int(part) if part.isdigit() else part for part in __version__.split(".")
    )

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

        """
        return cls.__version_info__

    @classmethod
    def is_version_at_least(cls, major: int, minor: int = 0, patch: int = 0) -> bool:
        """Check if current version meets minimum version requirement.

        Performs version comparison to determine if the current package version
        is at least the specified minimum version.

        Args:
            major: Minimum major version number
            minor: Minimum minor version number (default: 0)
            patch: Minimum patch version number (default: 0)

        Returns:
            bool: True if current version >= specified version

        """
        return cls.__version_info__ >= (major, minor, patch)

    @classmethod
    def get_package_info(cls) -> Mapping[str, str]:
        """Get comprehensive package information dictionary.

        Returns all available package metadata in a structured mapping
        format for programmatic access to package information.

        Returns:
            Mapping[str, str]: Package metadata containing:
                - name: Package name
                - version: Version string
                - description: Package description
                - author: Author name
                - author_email: Author email
                - license: License type
                - url: Homepage URL

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


__version__ = FlextInfraVersion.__version__
__version_info__ = FlextInfraVersion.__version_info__
__title__ = FlextInfraVersion.__title__
__description__ = FlextInfraVersion.__description__
__author__ = FlextInfraVersion.__author__
__author_email__ = FlextInfraVersion.__author_email__
__license__ = FlextInfraVersion.__license__
__url__ = FlextInfraVersion.__url__

__all__ = [
    "FlextInfraVersion",
    "__author__",
    "__author_email__",
    "__description__",
    "__license__",
    "__title__",
    "__url__",
    "__version__",
    "__version_info__",
]
