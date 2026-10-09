# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities. Beartype. Helpers Parts package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._beartype._helpers_parts.helpers_part_03 import (
        FlextUtilitiesBeartypeHelpers,
    )


__all__: tuple[str, ...] = ("FlextUtilitiesBeartypeHelpers",)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({"FlextUtilitiesBeartypeHelpers": ".helpers_part_03"}),
    public_exports=__all__,
)
