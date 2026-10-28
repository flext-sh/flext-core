# AUTO-GENERATED FILE — Regenerate with: make gen
"""Internal package for enforcement.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._models._enforcement.base import FlextModelsEnforcementBase
    from flext_core._models._enforcement.catalog import FlextModelsEnforcementCatalog
    from flext_core._models._enforcement.inspection import (
        FlextModelsEnforcementInspection,
    )
    from flext_core._models._enforcement.params import FlextModelsEnforcementParams
    from flext_core._models._enforcement.resolution import (
        FlextModelsEnforcementResolution,
    )
    from flext_core._models._enforcement.sources import FlextModelsEnforcementSources

__all__: tuple[str, ...] = (
    "FlextModelsEnforcementBase",
    "FlextModelsEnforcementCatalog",
    "FlextModelsEnforcementInspection",
    "FlextModelsEnforcementParams",
    "FlextModelsEnforcementResolution",
    "FlextModelsEnforcementSources",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextModelsEnforcementBase": ".base",
        "FlextModelsEnforcementCatalog": ".catalog",
        "FlextModelsEnforcementInspection": ".inspection",
        "FlextModelsEnforcementParams": ".params",
        "FlextModelsEnforcementResolution": ".resolution",
        "FlextModelsEnforcementSources": ".sources",
    }),
    public_exports=__all__,
)
