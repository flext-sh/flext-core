"""Version information for FLEXT Core.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

# Version information
__version__ = "0.8.0"
__version_info__ = (0, 8, 0)


def get_version() -> str:
    """Get version string."""
    return __version__


def get_version_info() -> tuple[int, int, int]:
    """Get version info tuple."""
    return __version_info__


__all__ = ["__version__", "__version_info__", "get_version", "get_version_info"]
