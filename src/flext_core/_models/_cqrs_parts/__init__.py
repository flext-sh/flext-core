# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Models. Cqrs Parts package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._models._cqrs_parts.flextmodelscqrs_part_01 import FlextModelsCqrs


__all__: tuple[str, ...] = ("FlextModelsCqrs",)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({"FlextModelsCqrs": ".flextmodelscqrs_part_01"}),
    public_exports=__all__,
)
