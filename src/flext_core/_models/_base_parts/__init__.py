# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Models. Base Parts package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._models._base_parts.flextmodelsbase_part_01 import (
        FlextModelsBasePart01,
    )
    from flext_core._models._base_parts.flextmodelsbase_part_03 import FlextModelsBase


__all__: tuple[str, ...] = ("FlextModelsBase", "FlextModelsBasePart01")

__getattr__ = install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextModelsBase": ".flextmodelsbase_part_03",
        "FlextModelsBasePart01": ".flextmodelsbase_part_01",
    }),
    public_exports=__all__,
)[0]
