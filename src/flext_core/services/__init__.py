# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core.services package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core import install_lazy_exports

if TYPE_CHECKING:
    from flext_core.services.dispatcher import FlextServiceDispatcher
    from flext_core.services.registry import FlextServiceRegistry


__all__: tuple[str, ...] = ("FlextServiceDispatcher", "FlextServiceRegistry")

__getattr__ = install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextServiceDispatcher": ".dispatcher",
        "FlextServiceRegistry": ".registry",
    }),
    public_exports=__all__,
)[0]
