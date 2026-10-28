# AUTO-GENERATED FILE — Regenerate with: make gen
"""Internal package for import.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._beartype._import.blacklist_visitor import (
        FlextUtilitiesBeartypeImportBlacklistVisitor,
    )

__all__: tuple[str, ...] = ("FlextUtilitiesBeartypeImportBlacklistVisitor",)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextUtilitiesBeartypeImportBlacklistVisitor": ".blacklist_visitor",
    }),
    public_exports=__all__,
)
