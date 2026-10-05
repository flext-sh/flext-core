# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Handlers Parts package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._handlers_parts.flexthandlers_part_07 import FlextHandlers


__all__: tuple[str, ...] = ("FlextHandlers",)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({"FlextHandlers": ".flexthandlers_part_07"}),
    public_exports=__all__,
)
