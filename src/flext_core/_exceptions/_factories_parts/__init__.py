# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Exceptions. Factories Parts package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._exceptions._factories_parts import (
        flextexceptionsfactories_part_01 as _owner_0,
    )

    FlextExceptionsFactoriesPart01 = _owner_0.FlextExceptionsFactoriesPart01
    from flext_core._exceptions._factories_parts import (
        flextexceptionsfactories_part_02 as _owner_1,
    )

    FlextExceptionsFactoriesPart02 = _owner_1.FlextExceptionsFactoriesPart02
    from flext_core._exceptions._factories_parts import (
        flextexceptionsfactories_part_03 as _owner_2,
    )

    FlextExceptionsFactoriesPart03 = _owner_2.FlextExceptionsFactoriesPart03
    from flext_core._exceptions._factories_parts import (
        flextexceptionsfactories_part_04 as _owner_3,
    )

    FlextExceptionsFactoriesPart04 = _owner_3.FlextExceptionsFactoriesPart04


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
