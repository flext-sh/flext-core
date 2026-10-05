# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Models. Enforcement package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._models._enforcement._base import (
        FlextModelsEnforcementBase,
        FlextModelsEnforcementModelBase,
    )
    from flext_core._models._enforcement._catalog import FlextModelsEnforcementCatalog
    from flext_core._models._enforcement._inspection import (
        FlextModelsEnforcementInspection,
    )
    from flext_core._models._enforcement._params import FlextModelsEnforcementParams
    from flext_core._models._enforcement._resolution import (
        FlextModelsEnforcementResolution,
    )
    from flext_core._models._enforcement._sources import FlextModelsEnforcementSources


__all__: tuple[str, ...] = (
    "FlextModelsEnforcementBase",
    "FlextModelsEnforcementCatalog",
    "FlextModelsEnforcementInspection",
    "FlextModelsEnforcementModelBase",
    "FlextModelsEnforcementParams",
    "FlextModelsEnforcementResolution",
    "FlextModelsEnforcementSources",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextModelsEnforcementBase": "._base",
        "FlextModelsEnforcementCatalog": "._catalog",
        "FlextModelsEnforcementInspection": "._inspection",
        "FlextModelsEnforcementModelBase": "._base",
        "FlextModelsEnforcementParams": "._params",
        "FlextModelsEnforcementResolution": "._resolution",
        "FlextModelsEnforcementSources": "._sources",
    }),
    public_exports=__all__,
)
