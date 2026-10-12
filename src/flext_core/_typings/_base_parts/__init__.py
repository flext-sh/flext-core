# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Typings. Base Parts package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._typings._base_parts.flexttypingbase_part_01 import FlextTypingBase


__all__: tuple[str, ...] = ("FlextTypingBase",)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({"FlextTypingBase": ".flexttypingbase_part_01"}),
    public_exports=__all__,
)
