# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities. Beartype. Class Visitor Parts. Parts package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._beartype._class_visitor_parts._parts.class_visitor_part_02_01 import (
        alias_first_violation,
    )


__all__: tuple[str, ...] = ("alias_first_violation",)

__getattr__ = install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({"alias_first_violation": ".class_visitor_part_02_01"}),
    public_exports=__all__,
)[0]
