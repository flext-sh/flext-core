# AUTO-GENERATED FILE — Regenerate with: make gen
"""Internal package for decorators.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._decorators.base import FlextDecoratorsBase
    from flext_core._decorators.combined import FlextDecoratorsCombined
    from flext_core._decorators.logging import FlextDecoratorsLogging
    from flext_core._decorators.railway import FlextDecoratorsRailway
    from flext_core._decorators.runtime import FlextDecoratorsRuntime

__all__: tuple[str, ...] = (
    "FlextDecoratorsBase",
    "FlextDecoratorsCombined",
    "FlextDecoratorsLogging",
    "FlextDecoratorsRailway",
    "FlextDecoratorsRuntime",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextDecoratorsBase": ".base",
        "FlextDecoratorsCombined": ".combined",
        "FlextDecoratorsLogging": ".logging",
        "FlextDecoratorsRailway": ".railway",
        "FlextDecoratorsRuntime": ".runtime",
    }),
    public_exports=__all__,
)
