# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities. Enforcement Collect Parts package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._enforcement_collect import (
        enforcement_collect_part_01 as _owner_0,
    )

    FlextUtilitiesEnforcementCollectPart01 = (
        _owner_0.FlextUtilitiesEnforcementCollectPart01
    )
    from flext_core._utilities._enforcement_collect import (
        enforcement_collect_part_02 as _owner_1,
    )

    FlextUtilitiesEnforcementCollectPart02 = (
        _owner_1.FlextUtilitiesEnforcementCollectPart02
    )


__all__: tuple[str, ...] = (
    "FlextUtilitiesEnforcementCollectPart01",
    "FlextUtilitiesEnforcementCollectPart02",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextUtilitiesEnforcementCollectPart01": ".enforcement_collect_part_01",
        "FlextUtilitiesEnforcementCollectPart02": ".enforcement_collect_part_02",
    }),
    public_exports=__all__,
)
