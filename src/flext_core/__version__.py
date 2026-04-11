"""Package version and metadata information.

Provides version information and package metadata for the flext-core package
using standard library metadata extraction.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib.metadata import PackageMetadata, metadata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flext_core import t


class FlextVersion:
    """Package version and metadata information.

    Provides version information and package metadata using standard library
    metadata extraction.
    """

    _metadata: PackageMetadata | Mapping[str, str] = metadata("flext-core")
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
    def resolve_package_info(cls) -> t.StrMapping:
        """Get comprehensive package information dictionary.

        Returns:
            t.StrMapping: Metadata including name, version, author, license, and url.

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

    @classmethod
    def resolve_version_info(cls) -> tuple[int | str, ...]:
        """Get package version as comparison-friendly tuple.

        Args:
            None.

        Returns:
            tuple[int | str, ...]: Version tuple for comparison.

        Notes:
            Each element is either an integer (numeric version-segment) or a
            string (pre-release identifier).

        """
        return cls.__version_info__

    @classmethod
    def resolve_version_string(cls) -> str:
        """Get package version as human-readable string.

        Args:
            None.

        Returns:
            str: Semantic version string.

        Notes:
            This follows PEP 440 semantic versioning.

        """
        return cls.__version__

    @classmethod
    def version_at_least(cls, major: int, minor: int = 0, patch: int = 0) -> bool:
        """Check if current version meets minimum version requirement.

        Args:
            major: Major version to compare.
            minor: Minor version to compare.
            patch: Patch version to compare.

        Returns:
            bool: True if current version is greater or equal to provided version.

        """
        return cls.__version_info__ >= (major, minor, patch)


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
