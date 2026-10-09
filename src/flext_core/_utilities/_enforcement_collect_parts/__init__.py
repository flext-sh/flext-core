# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities. Enforcement Collect Parts package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._enforcement_collect_parts.enforcement_collect_part_02 import (
        FlextUtilitiesEnforcementCollect,
    )


__all__: tuple[str, ...] = ("FlextUtilitiesEnforcementCollect",)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextUtilitiesEnforcementCollect": ".enforcement_collect_part_02",
    }),
    public_exports=__all__,
)
