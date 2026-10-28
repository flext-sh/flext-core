# AUTO-GENERATED FILE — Regenerate with: make gen
"""Internal package for guards.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._guards.type_protocol_specs import (
        FlextUtilitiesGuardsTypeProtocolSpecs,
    )
    from flext_core._utilities._guards.type_protocol_string import (
        FlextUtilitiesGuardsTypeProtocolString,
    )

__all__: tuple[str, ...] = (
    "FlextUtilitiesGuardsTypeProtocolSpecs",
    "FlextUtilitiesGuardsTypeProtocolString",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextUtilitiesGuardsTypeProtocolSpecs": ".type_protocol_specs",
        "FlextUtilitiesGuardsTypeProtocolString": ".type_protocol_string",
    }),
    public_exports=__all__,
)
