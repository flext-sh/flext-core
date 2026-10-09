# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Protocols. Container Parts package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._protocols._container_parts.flextprotocolscontainer_part_03 import (
        FlextProtocolsContainer,
    )


__all__: tuple[str, ...] = ("FlextProtocolsContainer",)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({"FlextProtocolsContainer": ".flextprotocolscontainer_part_03"}),
    public_exports=__all__,
)
