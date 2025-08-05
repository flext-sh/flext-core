"""Version information for FLEXT Core.

This file reads version from pyproject.toml metadata.
All version references should import from this file.
"""

import importlib.metadata

# Get metadata from the installed package
_metadata = importlib.metadata.metadata("flext-core")
__version__ = _metadata["Version"]
__project__ = _metadata["Name"]
__description__ = _metadata["Summary"]
__author__ = _metadata["Author"]
__author_email__ = (
    _metadata["Author-email"].split("<")[1].rstrip(">")
    if "<" in _metadata.get("Author-email", "")
    else ""
)
__maintainer__ = _metadata.get("Maintainer", __author__)
__maintainer_email__ = _metadata.get("Maintainer-email", __author_email__)
__license__ = _metadata.get("License", "MIT")

# Parse version info
_parts = __version__.split(".")
__version_info__ = tuple(int(p) if p.isdigit() else p for p in _parts)
__version_tuple__ = __version_info__

# Fixed metadata
__copyright__ = "Copyright (c) 2025 FLEXT Team. All rights reserved."

# Build information (can be populated by CI/CD)
__build__ = ""
__commit__ = ""
__branch__ = ""

# All exported symbols
__all__ = [
    "__author__",
    "__author_email__",
    "__branch__",
    "__build__",
    "__commit__",
    "__copyright__",
    "__description__",
    "__license__",
    "__maintainer__",
    "__maintainer_email__",
    "__project__",
    "__version__",
    "__version_info__",
    "__version_tuple__",
]
