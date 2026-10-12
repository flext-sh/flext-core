# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Constants. Enforcement Data package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._constants._enforcement_data.package_data import (
        FlextConstantsEnforcementData,
    )


__all__: tuple[str, ...] = ("FlextConstantsEnforcementData",)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({"FlextConstantsEnforcementData": ".package_data"}),
    public_exports=__all__,
)
