# AUTO-GENERATED FILE — Regenerate with: make gen
"""Package version and metadata for flext-cli.

Subclass of ``FlextVersion`` — overrides only ``_metadata``.
All derived attributes (``__version__``, ``__title__``, etc.) are
computed automatically via ``FlextVersion.__init_subclass__``.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from importlib.metadata import PackageMetadata, metadata

from flext_core.__version__ import FlextVersion


class FlextCliVersion(FlextVersion):
    """flext-cli version — MRO-derived from FlextVersion."""

    _metadata: PackageMetadata = metadata("flext-cli")


__version__ = FlextCliVersion.__version__
__version_info__ = FlextCliVersion.__version_info__
__title__ = FlextCliVersion.__title__
__description__ = FlextCliVersion.__description__
__author__ = FlextCliVersion.__author__
__author_email__ = FlextCliVersion.__author_email__
__license__ = FlextCliVersion.__license__
__url__ = FlextCliVersion.__url__
__all__: list[str] = [
    "FlextCliVersion",
    "__author__",
    "__author_email__",
    "__description__",
    "__license__",
    "__title__",
    "__url__",
    "__version__",
    "__version_info__",
]
