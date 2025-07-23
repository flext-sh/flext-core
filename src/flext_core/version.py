"""Version information for FLEXT Core.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Version information and utilities for the FLEXT Core library.
"""

from __future__ import annotations

from flext_core.constants import FlextConstants

__version__ = FlextConstants.VERSION


def get_version() -> str:
    """Get the current version string.

    Returns:
        The current version string

    """
    return __version__


def get_version_info() -> tuple[int, int, int]:
    """Get version information as a tuple of integers.

    Returns:
        Tuple of (major, minor, patch) version numbers

    """
    parts = __version__.split(".")
    major = int(parts[0])
    minor = int(parts[1])
    min_parts_for_patch = 3
    patch = int(parts[2]) if len(parts) > min_parts_for_patch - 1 else 0
    return (major, minor, patch)


__all__ = [
    "__version__",
    "get_version",
    "get_version_info",
]
