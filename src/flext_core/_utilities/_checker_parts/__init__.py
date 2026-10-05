# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities. Checker Parts package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._checker_parts.checker_part_03 import (
        FlextUtilitiesChecker,
    )


__all__: tuple[str, ...] = ("FlextUtilitiesChecker",)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({"FlextUtilitiesChecker": ".checker_part_03"}),
    public_exports=__all__,
)
