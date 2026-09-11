# AUTO-GENERATED FILE — Regenerate with: make gen
"""Package version and metadata for flext-tests.

Subclass of ``FlextVersion`` — overrides only ``_metadata``.
All derived attributes (``__version__``, ``__title__``, etc.) are
computed automatically via ``FlextVersion.__init_subclass__``.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from importlib.metadata import PackageMetadata, metadata

from flext_core.__version__ import FlextVersion


class FlextTestsVersion(FlextVersion):
    """flext-tests version — MRO-derived from FlextVersion."""

    _metadata: PackageMetadata = metadata("flext-tests")


__version__ = FlextTestsVersion.__version__
__version_info__ = FlextTestsVersion.__version_info__
__title__ = FlextTestsVersion.__title__
__description__ = FlextTestsVersion.__description__
__author__ = FlextTestsVersion.__author__
__author_email__ = FlextTestsVersion.__author_email__
__license__ = FlextTestsVersion.__license__
__url__ = FlextTestsVersion.__url__
__all__: list[str] = [
    "FlextTestsVersion",
    "__author__",
    "__author_email__",
    "__description__",
    "__license__",
    "__title__",
    "__url__",
    "__version__",
    "__version_info__",
]
