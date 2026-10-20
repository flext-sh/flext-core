# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Runtime package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._runtime._base import FlextRuntimeBase


__all__: tuple[str, ...] = ("FlextRuntimeBase",)

__getattr__ = install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({"FlextRuntimeBase": "._base"}),
    public_exports=__all__,
)[0]
