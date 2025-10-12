"""Package version and metadata information.

This module provides version and package metadata for flext-core using
importlib.metadata, extracting information from the package's metadata
including version, author, license, and other package details.

Exports:
    - __version__: Package version as string
    - __version_info__: Version as tuple for comparison
    - __title__: Package name from metadata
    - __description__: Package description
    - __author__: Package author name
    - __author_email__: Package author email
    - __license__: Package license
    - __url__: Package homepage URL

Usage:
    >>> from flext_core import __version__, __version_info__
    >>> print(f"flext-core version: {__version__}")
    >>> if __version_info__ >= (1, 0, 0):
    ...     print("Version 1.0.0 or higher")

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from importlib.metadata import metadata

_metadata = metadata("flext-core")
__version__ = _metadata["Version"]
__version_info__ = tuple(
    int(part) if part.isdigit() else part for part in __version__.split(".")
)
__title__ = _metadata["Name"]
__description__ = _metadata["Summary"]
__author__ = _metadata.get("Author")
__author_email__ = _metadata.get("Author-Email")
__license__ = _metadata.get("License")
__url__ = _metadata.get("Home-Page")

__all__ = [
    "__author__",
    "__author_email__",
    "__description__",
    "__license__",
    "__title__",
    "__url__",
    "__version__",
    "__version_info__",
]
