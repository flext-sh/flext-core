# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities. Beartype. Class Visitor Parts package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._beartype._class_visitor import visitor_base as _owner_0
    from flext_core._utilities._beartype._class_visitor.alias_rules import (
        alias_first_violation,
    )

    FlextUtilitiesBeartypeClassVisitorPart01 = (
        _owner_0.FlextUtilitiesBeartypeClassVisitorPart01
    )
    from flext_core._utilities._beartype._class_visitor import (
        visitor_checks as _owner_1,
    )

    FlextUtilitiesBeartypeClassVisitorPart02 = (
        _owner_1.FlextUtilitiesBeartypeClassVisitorPart02
    )
    from flext_core._utilities._beartype._class_visitor import visitor as _owner_2

    FlextUtilitiesBeartypeClassVisitorPart03 = (
        _owner_2.FlextUtilitiesBeartypeClassVisitorPart03
    )


__all__: tuple[str, ...] = (
    "FlextUtilitiesBeartypeClassVisitorPart01",
    "FlextUtilitiesBeartypeClassVisitorPart02",
    "FlextUtilitiesBeartypeClassVisitorPart03",
    "alias_first_violation",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextUtilitiesBeartypeClassVisitorPart01": ".visitor_base",
        "FlextUtilitiesBeartypeClassVisitorPart02": ".visitor_checks",
        "FlextUtilitiesBeartypeClassVisitorPart03": ".visitor",
        "alias_first_violation": ".alias_rules",
    }),
    public_exports=__all__,
)
