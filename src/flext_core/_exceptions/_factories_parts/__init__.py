# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Exceptions. Factories Parts package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._exceptions._factories_parts.flextexceptionsfactories_part_04 import (
        FlextExceptionsFactories,
    )


__all__: tuple[str, ...] = ("FlextExceptionsFactories",)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({"FlextExceptionsFactories": ".flextexceptionsfactories_part_04"}),
    public_exports=__all__,
)
