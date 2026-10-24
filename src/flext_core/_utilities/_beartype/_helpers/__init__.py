# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities. Beartype. Helpers Parts package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._beartype._helpers.helpers_part_01 import (
        FlextUtilitiesBeartypeHelpersPart01,
    )
    from flext_core._utilities._beartype._helpers.helpers_part_02 import (
        FlextUtilitiesBeartypeHelpersPart02,
    )
    from flext_core._utilities._beartype._helpers.helpers_part_03 import (
        FlextUtilitiesBeartypeHelpersPart03,
    )


__all__: tuple[str, ...] = (
    "FlextUtilitiesBeartypeHelpersPart01",
    "FlextUtilitiesBeartypeHelpersPart02",
    "FlextUtilitiesBeartypeHelpersPart03",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextUtilitiesBeartypeHelpersPart01": ".helpers_part_01",
        "FlextUtilitiesBeartypeHelpersPart02": ".helpers_part_02",
        "FlextUtilitiesBeartypeHelpersPart03": ".helpers_part_03",
    }),
    public_exports=__all__,
)
