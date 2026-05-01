"""Package version and metadata information.

Provides version information and package metadata for the flext-core package
using standard library metadata extraction.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from importlib.metadata import PackageMetadata, metadata

from flext_core._typings.base import FlextTypingBase as t
from flext_core._typings.pydantic import FlextTypesPydantic as tp


class FlextVersion:
    """Package version and metadata — SSOT base class.

    Subclasses override only ``_metadata``.  ``__init_subclass__``
    recomputes every derived attribute via MRO — zero duplication.
    """

    _metadata: PackageMetadata = metadata("flext-core")

    # -- Base-class derivation (inline; subclass derivation via __init_subclass__) --
    __version__: str = _metadata["Version"]
    __version_info__: t.VariadicTuple[int | str] = tuple(
        int(part) if part.isdigit() else part for part in __version__.split(".")
    )
    __title__: str = _metadata.get("Name", "")
    __description__: str = _metadata.get("Summary", "")
    __author__: str = _metadata.get("Author", "") or _metadata.get("Author-Email", "")
    __author_email__: str = _metadata.get("Author-Email", "")
    __license__: str = _metadata.get("License-Expression", "") or _metadata.get(
        "License", ""
    )
    __url__: str = _metadata.get("Home-Page", "") or _metadata.get("Project-URL", "")

    def __init_subclass__(cls, **kwargs: tp.JsonValue) -> None:
        """Recompute derived attributes when a subclass overrides ``_metadata``."""
        super().__init_subclass__(**kwargs)
        if "_metadata" in cls.__dict__:
            m = cls._metadata
            cls.__version__ = m["Version"]
            cls.__version_info__ = tuple(
                int(p) if p.isdigit() else p for p in cls.__version__.split(".")
            )
            cls.__title__ = m.get("Name", "")
            cls.__description__ = m.get("Summary", "")
            cls.__author__ = m.get("Author", "") or m.get("Author-Email", "")
            cls.__author_email__ = m.get("Author-Email", "")
            cls.__license__ = m.get("License-Expression", "") or m.get("License", "")
            cls.__url__ = m.get("Home-Page", "") or m.get("Project-URL", "")

    @classmethod
    def resolve_package_info(cls) -> t.MappingKV[str, str]:
        """Get comprehensive package information dictionary."""
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
    def resolve_version_info(cls) -> t.VariadicTuple[int | str]:
        """Get package version as comparison-friendly tuple."""
        return cls.__version_info__

    @classmethod
    def resolve_version_string(cls) -> str:
        """Get package version as human-readable string."""
        return cls.__version__

    @classmethod
    def version_at_least(cls, major: int, minor: int = 0, patch: int = 0) -> bool:
        """Check if current version meets minimum version requirement."""
        return cls.__version_info__ >= (major, minor, patch)


__version__ = FlextVersion.__version__
__version_info__ = FlextVersion.__version_info__
__title__ = FlextVersion.__title__
__description__ = FlextVersion.__description__
__author__ = FlextVersion.__author__
__author_email__ = FlextVersion.__author_email__
__license__ = FlextVersion.__license__
__url__ = FlextVersion.__url__
__all__: t.MutableSequenceOf[str] = [
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
