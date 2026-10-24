# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Protocols. Context package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._protocols._context.flextprotocolscontext_part_01 import (
        FlextProtocolsContextPart01,
    )
    from flext_core._protocols._context.flextprotocolscontext_part_02 import (
        FlextProtocolsContextPart02,
    )
    from flext_core._protocols._context.flextprotocolscontext_part_03 import (
        FlextProtocolsContextPart03,
    )


__all__: tuple[str, ...] = (
    "FlextProtocolsContextPart01",
    "FlextProtocolsContextPart02",
    "FlextProtocolsContextPart03",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextProtocolsContextPart01": ".flextprotocolscontext_part_01",
        "FlextProtocolsContextPart02": ".flextprotocolscontext_part_02",
        "FlextProtocolsContextPart03": ".flextprotocolscontext_part_03",
    }),
    public_exports=__all__,
)
