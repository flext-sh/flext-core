# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Models. Container Parts package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._models._container_parts.flextmodelscontainer_part_01 import (
        FlextModelsContainerPart01,
    )
    from flext_core._models._container_parts.flextmodelscontainer_part_02 import (
        FlextModelsContainerPart02,
    )
    from flext_core._models._container_parts.flextmodelscontainer_part_03 import (
        FlextModelsContainerPart03,
    )
    from flext_core._models._container_parts.flextmodelscontainer_part_04 import (
        FlextModelsContainerPart04,
    )


__all__: tuple[str, ...] = (
    "FlextModelsContainerPart01",
    "FlextModelsContainerPart02",
    "FlextModelsContainerPart03",
    "FlextModelsContainerPart04",
)

__getattr__ = install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextModelsContainerPart01": ".flextmodelscontainer_part_01",
        "FlextModelsContainerPart02": ".flextmodelscontainer_part_02",
        "FlextModelsContainerPart03": ".flextmodelscontainer_part_03",
        "FlextModelsContainerPart04": ".flextmodelscontainer_part_04",
    }),
    public_exports=__all__,
)[0]
