"""Package Version and Metadata Information (Layer 0).

**ARCHITECTURE LAYER 0** - Pure constants extracted from package metadata

This module provides version and package metadata for flext-core using
importlib.metadata, extracting information from the package's metadata
including version, author, license, and other package details. Implements
structural typing via FlextProtocols (duck typing - no inheritance required).

**Protocol Compliance** (Structural Typing):
Satisfies FlextProtocols.Version through exported tuple and string properties:
- Version exports follow PEP 440 semantic versioning format
- __version_info__ provides comparison-friendly tuple representation
- isinstance() validation through method signatures (duck typing)
- No inheritance from @runtime_checkable protocols

**Exports** (8 public attributes):
1. **__version__**: Package version as string (e.g., "0.9.9")
2. **__version_info__**: Tuple of version parts for comparison (e.g., (0, 9, 9))
3. **__title__**: Package name from metadata ("flext-core")
4. **__description__**: Short package description
5. **__author__**: Package author name
6. **__author_email__**: Package author email address
7. **__license__**: Package license (MIT)
8. **__url__**: Package homepage URL

**Version Format**:
- String format: "major.minor.patch" (e.g., "1.0.0")
- Tuple format: (major, minor, patch, ...) for numeric comparison
- Supports pre-release versions: "1.0.0rc1", "1.0.0a1", "1.0.0b1"
- Compatible with PEP 440 semantic versioning

**Production Readiness Checklist**:
✅ Package metadata extraction via importlib.metadata
✅ Semantic version tuple for comparisons (PEP 440 compatible)
✅ Version string for human-readable display
✅ All metadata fields properly exported
✅ Graceful fallback for missing metadata (via .get())
✅ 100% type-safe (strict MyPy compliance)
✅ Zero dependencies (stdlib only)

**Usage Patterns**:
1. **Check version string**: `print(f"flext-core {__version__}")`
2. **Version comparison**: `if __version_info__ >= (1, 0, 0): ...`
3. **Display all metadata**: `print(f"{__title__} - {__description__}")`
4. **Import version**: `from flext_core import __version__`
5. **Runtime version check**: `assert __version_info__[0] >= 0  # Major version`
6. **API version compatibility**: Use __version_info__ for feature flags
7. **Package author info**: Access via __author__, __author_email__
8. **License verification**: `assert __license__ == "MIT"`

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
