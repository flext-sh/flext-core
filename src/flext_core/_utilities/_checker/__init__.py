# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities. Checker Parts package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._checker_parts.checker_part_01 import (
        FlextUtilitiesCheckerPart01,
    )
    from flext_core._utilities._checker_parts.checker_part_02 import (
        FlextUtilitiesCheckerPart02,
    )
    from flext_core._utilities._checker_parts.checker_part_03 import (
        FlextUtilitiesCheckerPart03,
    )


__all__: tuple[str, ...] = (
    "FlextUtilitiesCheckerPart01",
    "FlextUtilitiesCheckerPart02",
    "FlextUtilitiesCheckerPart03",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextUtilitiesCheckerPart01": ".checker_part_01",
        "FlextUtilitiesCheckerPart02": ".checker_part_02",
        "FlextUtilitiesCheckerPart03": ".checker_part_03",
    }),
    public_exports=__all__,
)
