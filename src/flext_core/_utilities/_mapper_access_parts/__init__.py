# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities. Mapper Access Parts package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._mapper_access_parts.mapper_access_part_02 import (
        FlextUtilitiesMapperAccess,
    )


__all__: tuple[str, ...] = ("FlextUtilitiesMapperAccess",)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({"FlextUtilitiesMapperAccess": ".mapper_access_part_02"}),
    public_exports=__all__,
)
