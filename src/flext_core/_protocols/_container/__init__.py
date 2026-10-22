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
    from flext_core._protocols._container.flextprotocolscontainer_part_01 import (
        FlextProtocolsContainerPart01,
    )
    from flext_core._protocols._container.flextprotocolscontainer_part_02 import (
        FlextProtocolsContainerPart02,
    )
    from flext_core._protocols._container.flextprotocolscontainer_part_03 import (
        FlextProtocolsContainerPart03,
    )


__all__: tuple[str, ...] = (
    "FlextProtocolsContainerPart01",
    "FlextProtocolsContainerPart02",
    "FlextProtocolsContainerPart03",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextProtocolsContainerPart01": ".flextprotocolscontainer_part_01",
        "FlextProtocolsContainerPart02": ".flextprotocolscontainer_part_02",
        "FlextProtocolsContainerPart03": ".flextprotocolscontainer_part_03",
    }),
    public_exports=__all__,
)
