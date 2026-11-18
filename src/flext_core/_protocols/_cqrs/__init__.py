# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Protocols. Cqrs package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._protocols._cqrs.pagination import FlextProtocolsCqrsPagination
    from flext_core._protocols._cqrs.part_01 import (
        FlextProtocolsCqrsFlextmodelscqrsPart01,
    )


__all__: tuple[str, ...] = (
    "FlextProtocolsCqrsFlextmodelscqrsPart01",
    "FlextProtocolsCqrsPagination",
)

install_lazy_exports(
    __name__,
    globals(),
    {
        "FlextProtocolsCqrsFlextmodelscqrsPart01": ".part_01",
        "FlextProtocolsCqrsPagination": ".pagination",
    },
    public_exports=__all__,
    deferred_members=("FlextProtocolsCqrsPagination",),
)
