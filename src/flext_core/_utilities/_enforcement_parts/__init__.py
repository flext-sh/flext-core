# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities. Enforcement Parts package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._enforcement_parts.enforcement_part_01 import (
        PREDICATE_BINDINGS,
    )
    from flext_core._utilities._enforcement_parts.enforcement_part_05 import (
        FlextUtilitiesEnforcement,
    )


__all__: tuple[str, ...] = ("PREDICATE_BINDINGS", "FlextUtilitiesEnforcement")

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "PREDICATE_BINDINGS": ".enforcement_part_01",
        "FlextUtilitiesEnforcement": ".enforcement_part_05",
    }),
    public_exports=__all__,
)
