# AUTO-GENERATED FILE — Regenerate with: make gen
"""Internal package for library.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._beartype._library.visitor import (
        FlextUtilitiesBeartypeLibraryVisitor,
    )

__all__: tuple[str, ...] = ("FlextUtilitiesBeartypeLibraryVisitor",)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextUtilitiesBeartypeLibraryVisitor": ".visitor",
    }),
    public_exports=__all__,
)
