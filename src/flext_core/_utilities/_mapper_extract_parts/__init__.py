# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities. Mapper Extract Parts package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._mapper_extract_parts.mapper_extract_part_02 import (
        FlextUtilitiesMapperExtract,
    )


__all__: tuple[str, ...] = ("FlextUtilitiesMapperExtract",)

__getattr__ = install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({"FlextUtilitiesMapperExtract": ".mapper_extract_part_02"}),
    public_exports=__all__,
)[0]
