"""Runtime facade implementation building blocks.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from ._container import FlextRuntimeContainer
    from ._dependency import FlextRuntimeDependencyIntegration

__all__: tuple[str, ...] = (
    "FlextRuntimeContainer",
    "FlextRuntimeDependencyIntegration",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType(
        build_lazy_import_map(
            MappingProxyType({
                "._container": ("FlextRuntimeContainer",),
                "._dependency": ("FlextRuntimeDependencyIntegration",),
            }),
            alias_groups=MappingProxyType({}),
            sort_keys=False,
        )
    ),
    public_exports=__all__,
)
