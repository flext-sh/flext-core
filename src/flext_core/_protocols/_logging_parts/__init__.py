# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Protocols. Logging Parts package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._protocols._logging_parts.flextprotocolslogging_part_03 import (
        FlextProtocolsLogging,
    )


__all__: tuple[str, ...] = ("FlextProtocolsLogging",)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({"FlextProtocolsLogging": ".flextprotocolslogging_part_03"}),
    public_exports=__all__,
)
