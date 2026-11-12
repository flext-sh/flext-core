# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Protocols. Cqrs package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._protocols._cqrs.flextmodelscqrs_pagination import (
        FlextProtocolsCqrsFlextmodelscqrsPagination,
    )
    from flext_core._protocols._cqrs.flextmodelscqrs_part_01 import (
        FlextProtocolsCqrsFlextmodelscqrsPart01,
    )


__all__: tuple[str, ...] = (
    "FlextProtocolsCqrsFlextmodelscqrsPagination",
    "FlextProtocolsCqrsFlextmodelscqrsPart01",
)

install_lazy_exports(
    __name__,
    globals(),
    {
        "FlextProtocolsCqrsFlextmodelscqrsPagination": ".flextmodelscqrs_pagination",
        "FlextProtocolsCqrsFlextmodelscqrsPart01": ".flextmodelscqrs_part_01",
    },
    public_exports=__all__,
)
