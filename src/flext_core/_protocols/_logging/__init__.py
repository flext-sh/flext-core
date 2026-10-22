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
    from flext_core._protocols._logging_parts.flextprotocolslogging_part_01 import (
        FlextProtocolsLoggingPart01,
    )
    from flext_core._protocols._logging_parts.flextprotocolslogging_part_02 import (
        FlextProtocolsLoggingPart02,
    )
    from flext_core._protocols._logging_parts.flextprotocolslogging_part_03 import (
        FlextProtocolsLoggingPart03,
    )


__all__: tuple[str, ...] = (
    "FlextProtocolsLoggingPart01",
    "FlextProtocolsLoggingPart02",
    "FlextProtocolsLoggingPart03",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextProtocolsLoggingPart01": ".flextprotocolslogging_part_01",
        "FlextProtocolsLoggingPart02": ".flextprotocolslogging_part_02",
        "FlextProtocolsLoggingPart03": ".flextprotocolslogging_part_03",
    }),
    public_exports=__all__,
)
