"""Package version and metadata information.

Provides version information and package metadata for the flext-core package.
``u.PackageVersion`` owns the derivation; this module only declares the
distribution whose metadata it reads.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from importlib.metadata import PackageMetadata, metadata

from flext_core._utilities import FlextUtilitiesProjectMetadata


class FlextVersion(FlextUtilitiesProjectMetadata.PackageVersion):
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
