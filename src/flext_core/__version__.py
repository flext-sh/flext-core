"""Package version and metadata information.

Provides version information and package metadata for the flext-core package
using standard library metadata extraction.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from email.utils import parseaddr
from importlib.metadata import PackageMetadata, metadata

from packaging.version import Version


class _FlextVersionMetadata:
    """MRO base that derives package metadata for each concrete facade."""

    _metadata: PackageMetadata
    __version__: str = ""
    __version_info__: tuple[int, int, int]
    __title__: str = ""
    __description__: str = ""
    __author__: str = ""
    __author_email__: str = ""
    __license__: str = ""
    __url__: str = ""

    @staticmethod
    def _resolve_author(package_metadata: PackageMetadata) -> tuple[str, str]:
        """Return the first normalized author identity from package metadata."""
        raw_email = package_metadata.get("Author-Email", "")
        author_name, author_email = parseaddr(raw_email, strict=True)
        if raw_email and not author_email:
            msg = f"invalid Author-Email package metadata: {raw_email!r}"
            raise ValueError(msg)
        return (
            package_metadata.get("Author", "") or author_name or author_email,
            author_email,
        )

    @staticmethod
    def _resolve_homepage(package_metadata: PackageMetadata) -> str:
        """Return the legacy Home-Page or labeled Homepage project URL."""
        if homepage := package_metadata.get("Home-Page", ""):
            return homepage
        for project_url in package_metadata.get_all("Project-URL") or ():
            label, separator, url = project_url.partition(",")
            if label.strip().casefold() != "homepage":
                continue
            if not separator or not url.strip():
                msg = f"invalid Homepage project URL metadata: {project_url!r}"
                raise ValueError(msg)
            return url.strip()
        return ""

    @staticmethod
    def _resolve_version_info(version: str) -> tuple[int, int, int]:
        """Return the exact three-component PEP 440 release tuple."""
        try:
            major, minor, patch = Version(version).release
        except ValueError as exc:
            msg = f"invalid three-part semantic version metadata: {version!r}"
            raise ValueError(msg) from exc
        return major, minor, patch

    @classmethod
    def _apply_metadata(cls) -> None:
        """Derive every public value once for the current MRO class."""
        package_metadata = cls._metadata
        cls.__version__ = package_metadata["Version"]
        cls.__version_info__ = cls._resolve_version_info(cls.__version__)
        cls.__title__ = package_metadata.get("Name", "")
        cls.__description__ = package_metadata.get("Summary", "")
        cls.__author__, cls.__author_email__ = cls._resolve_author(package_metadata)
        cls.__license__ = package_metadata.get(
            "License-Expression", ""
        ) or package_metadata.get("License", "")
        cls.__url__ = cls._resolve_homepage(package_metadata)

    def __init_subclass__(cls, **kwargs: str | float | bool | None) -> None:
        """Recompute derived attributes when a subclass overrides ``_metadata``."""
        _ = kwargs
        super().__init_subclass__()
        if "_metadata" in cls.__dict__:
            cls._apply_metadata()


class FlextVersion(_FlextVersionMetadata):
    """Package metadata facade recomputed for subclasses through MRO."""

    _metadata: PackageMetadata = metadata("flext-core")


__version__ = FlextVersion.__version__
__version_info__ = FlextVersion.__version_info__
__title__ = FlextVersion.__title__
__description__ = FlextVersion.__description__
__author__ = FlextVersion.__author__
__author_email__ = FlextVersion.__author_email__
__license__ = FlextVersion.__license__
__url__ = FlextVersion.__url__
__all__: list[str] = [
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
