# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Protocols. Context Parts package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._protocols._context_parts.flextprotocolscontext_part_03 import (
        FlextProtocolsContext,
    )


__all__: tuple[str, ...] = ("FlextProtocolsContext",)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            ".flextprotocolscontext_part_03": ("FlextProtocolsContext",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    ),
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
