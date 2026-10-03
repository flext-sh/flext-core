# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities. Beartype. Class Visitor Parts. Parts package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._beartype._class_visitor_parts._parts.class_visitor_part_02_01 import (
        alias_first_violation,
    )
    from flext_core._utilities._beartype._class_visitor_parts._parts.class_visitor_part_02_02 import (
        redundant_inner_violation,
        self_ref_violation,
    )


__all__: tuple[str, ...] = (
    "alias_first_violation",
    "redundant_inner_violation",
    "self_ref_violation",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            ".class_visitor_part_02_01": ("alias_first_violation",),
            ".class_visitor_part_02_02": (
                "redundant_inner_violation",
                "self_ref_violation",
            ),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    ),
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
