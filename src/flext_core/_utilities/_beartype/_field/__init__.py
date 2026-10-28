# AUTO-GENERATED FILE — Regenerate with: make gen
"""Internal package for field.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._beartype._field.union_scan import (
        FlextUtilitiesBeartypeFieldUnionScan,
    )

__all__: tuple[str, ...] = ("FlextUtilitiesBeartypeFieldUnionScan",)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextUtilitiesBeartypeFieldUnionScan": ".union_scan",
    }),
    public_exports=__all__,
)
