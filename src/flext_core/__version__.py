"""Version and package metadata using importlib.metadata."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, metadata

try:
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
except PackageNotFoundError:
    # Fallback for development when package is not installed
    __version__ = "0.9.9"
    __version_info__ = (0, 9, 9)
    __title__ = "flext-core"
    __description__ = (
        "Enterprise Foundation Framework - Modern Python 3.13 + Clean Architecture"
    )
    __author__ = "FLEXT Team"
    __author_email__ = "team@flext.sh"
    __license__ = "MIT"
    __url__ = "https://github.com/flext-sh/flext-core"

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
