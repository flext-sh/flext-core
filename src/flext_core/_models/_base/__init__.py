# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Models. Base package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._models._base.flextmodelsbase_part_01 import FlextModelsBasePart01
    from flext_core._models._base.flextmodelsbase_part_02 import FlextModelsBasePart02
    from flext_core._models._base.flextmodelsbase_part_03 import FlextModelsBasePart03


__all__: tuple[str, ...] = (
    "FlextModelsBasePart01",
    "FlextModelsBasePart02",
    "FlextModelsBasePart03",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextModelsBasePart01": ".flextmodelsbase_part_01",
        "FlextModelsBasePart02": ".flextmodelsbase_part_02",
        "FlextModelsBasePart03": ".flextmodelsbase_part_03",
    }),
    public_exports=__all__,
)
