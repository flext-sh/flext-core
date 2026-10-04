# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Decorators package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._decorators._base import FlextDecoratorsBase
    from flext_core._decorators._combined import FlextDecoratorsCombined
    from flext_core._decorators._logging import FlextDecoratorsLogging
    from flext_core._decorators._logging_payloads import FlextDecoratorsLoggingPayloads
    from flext_core._decorators._railway import FlextDecoratorsRailway
    from flext_core._decorators._runtime import FlextDecorators


__all__: tuple[str, ...] = (
    "FlextDecorators",
    "FlextDecoratorsBase",
    "FlextDecoratorsCombined",
    "FlextDecoratorsLogging",
    "FlextDecoratorsLoggingPayloads",
    "FlextDecoratorsRailway",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            "._base": ("FlextDecoratorsBase",),
            "._combined": ("FlextDecoratorsCombined",),
            "._logging": ("FlextDecoratorsLogging",),
            "._logging_payloads": ("FlextDecoratorsLoggingPayloads",),
            "._railway": ("FlextDecoratorsRailway",),
            "._runtime": ("FlextDecorators",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    ),
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
