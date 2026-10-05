# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Models. Container Parts package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._models._container_parts.flextmodelscontainer_part_04 import (
        FlextModelsContainer,
    )


__all__: tuple[str, ...] = ("FlextModelsContainer",)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({"FlextModelsContainer": ".flextmodelscontainer_part_04"}),
    public_exports=__all__,
)
