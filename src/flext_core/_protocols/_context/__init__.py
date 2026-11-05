# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Protocols. Context package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import flext_core
from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._protocols._context.data import FlextProtocolsContextData
    from flext_core._protocols._context.export import FlextProtocolsContextExport
    from flext_core._protocols._context.scope import FlextProtocolsContextScope
    from flext_core._protocols._context.tokens import FlextProtocolsContextTokens


__all__: tuple[str, ...] = (
    "FlextProtocolsContextData",
    "FlextProtocolsContextExport",
    "FlextProtocolsContextScope",
    "FlextProtocolsContextTokens",
    "flext_core",
)

install_lazy_exports(
    __name__,
    globals(),
    {
        "FlextProtocolsContextData": ".data",
        "FlextProtocolsContextExport": ".export",
        "FlextProtocolsContextScope": ".scope",
        "FlextProtocolsContextTokens": ".tokens",
    },
    public_exports=__all__,
)
