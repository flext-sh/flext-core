# AUTO-GENERATED FILE — Regenerate with: make gen
"""Internal package for module.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._beartype._module.source_deferred import (
        FlextUtilitiesBeartypeModuleSourceDeferred,
    )
    from flext_core._utilities._beartype._module.source_provenance import (
        FlextUtilitiesBeartypeModuleSourceProvenance,
    )
    from flext_core._utilities._beartype._module.source_scan import (
        FlextUtilitiesBeartypeModuleSourceScan,
    )

__all__: tuple[str, ...] = (
    "FlextUtilitiesBeartypeModuleSourceDeferred",
    "FlextUtilitiesBeartypeModuleSourceProvenance",
    "FlextUtilitiesBeartypeModuleSourceScan",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextUtilitiesBeartypeModuleSourceDeferred": ".source_deferred",
        "FlextUtilitiesBeartypeModuleSourceProvenance": ".source_provenance",
        "FlextUtilitiesBeartypeModuleSourceScan": ".source_scan",
    }),
    public_exports=__all__,
)
