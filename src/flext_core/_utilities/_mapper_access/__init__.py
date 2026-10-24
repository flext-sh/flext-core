# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities. Mapper Access package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._mapper_access.mapper_access_part_01 import (
        FlextUtilitiesMapperAccessPart01,
    )
    from flext_core._utilities._mapper_access.mapper_access_part_02 import (
        FlextUtilitiesMapperAccessPart02,
    )


__all__: tuple[str, ...] = (
    "FlextUtilitiesMapperAccessPart01",
    "FlextUtilitiesMapperAccessPart02",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextUtilitiesMapperAccessPart01": ".mapper_access_part_01",
        "FlextUtilitiesMapperAccessPart02": ".mapper_access_part_02",
    }),
    public_exports=__all__,
)
