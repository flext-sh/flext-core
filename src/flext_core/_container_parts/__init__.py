# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Container Parts package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._container_parts.flextcontainertestingops_part_01 import (
        FlextContainerTestingOps,
    )


__all__: tuple[str, ...] = ("FlextContainerTestingOps",)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({"FlextContainerTestingOps": ".flextcontainertestingops_part_01"}),
    public_exports=__all__,
)
