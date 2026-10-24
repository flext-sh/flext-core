# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities. Enforcement Parts package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._enforcement.enforcement_part_01 import (
        PREDICATE_BINDINGS,
    )
    from flext_core._utilities._enforcement.enforcement_part_02 import (
        FlextUtilitiesEnforcementPart02,
    )
    from flext_core._utilities._enforcement.enforcement_part_03 import (
        FlextUtilitiesEnforcementPart03,
    )
    from flext_core._utilities._enforcement.enforcement_part_04 import (
        FlextUtilitiesEnforcementPart04,
    )
    from flext_core._utilities._enforcement.enforcement_part_05 import (
        FlextUtilitiesEnforcementPart05,
    )


__all__: tuple[str, ...] = (
    "PREDICATE_BINDINGS",
    "FlextUtilitiesEnforcementPart02",
    "FlextUtilitiesEnforcementPart03",
    "FlextUtilitiesEnforcementPart04",
    "FlextUtilitiesEnforcementPart05",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "PREDICATE_BINDINGS": ".enforcement_part_01",
        "FlextUtilitiesEnforcementPart02": ".enforcement_part_02",
        "FlextUtilitiesEnforcementPart03": ".enforcement_part_03",
        "FlextUtilitiesEnforcementPart04": ".enforcement_part_04",
        "FlextUtilitiesEnforcementPart05": ".enforcement_part_05",
    }),
    public_exports=__all__,
)
