# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Exceptions. Factories package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._exceptions._factories.flextexceptionsfactories_part_01 import (
        FlextExceptionsFactoriesPart01,
    )
    from flext_core._exceptions._factories.flextexceptionsfactories_part_02 import (
        FlextExceptionsFactoriesPart02,
    )
    from flext_core._exceptions._factories.flextexceptionsfactories_part_03 import (
        FlextExceptionsFactoriesPart03,
    )
    from flext_core._exceptions._factories.flextexceptionsfactories_part_04 import (
        FlextExceptionsFactoriesPart04,
    )


__all__: tuple[str, ...] = (
    "FlextExceptionsFactoriesPart01",
    "FlextExceptionsFactoriesPart02",
    "FlextExceptionsFactoriesPart03",
    "FlextExceptionsFactoriesPart04",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextExceptionsFactoriesPart01": ".flextexceptionsfactories_part_01",
        "FlextExceptionsFactoriesPart02": ".flextexceptionsfactories_part_02",
        "FlextExceptionsFactoriesPart03": ".flextexceptionsfactories_part_03",
        "FlextExceptionsFactoriesPart04": ".flextexceptionsfactories_part_04",
    }),
    public_exports=__all__,
)
