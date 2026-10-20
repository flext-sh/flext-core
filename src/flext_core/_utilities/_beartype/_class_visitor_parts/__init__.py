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
    from flext_core._utilities._beartype._class_visitor_parts import (
        _parts as _parts,
        class_visitor_part_01 as _owner_0,
    )
    from flext_core._utilities._beartype._class_visitor_parts._parts import (
        alias_first_violation,
    )

    FlextUtilitiesBeartypeClassVisitorPart01 = (
        _owner_0.FlextUtilitiesBeartypeClassVisitorPart01
    )
    from flext_core._utilities._beartype._class_visitor_parts import (
        class_visitor_part_02 as _owner_1,
    )

    FlextUtilitiesBeartypeClassVisitorPart02 = (
        _owner_1.FlextUtilitiesBeartypeClassVisitorPart02
    )
    from flext_core._utilities._beartype._class_visitor_parts import (
        class_visitor_part_03 as _owner_2,
    )

    FlextUtilitiesBeartypeClassVisitorPart03 = (
        _owner_2.FlextUtilitiesBeartypeClassVisitorPart03
    )


__all__: tuple[str, ...] = (
    "FlextUtilitiesBeartypeClassVisitorPart01",
    "FlextUtilitiesBeartypeClassVisitorPart02",
    "FlextUtilitiesBeartypeClassVisitorPart03",
    "_parts",
    "alias_first_violation",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextUtilitiesBeartypeClassVisitorPart01": ".class_visitor_part_01",
        "FlextUtilitiesBeartypeClassVisitorPart02": ".class_visitor_part_02",
        "FlextUtilitiesBeartypeClassVisitorPart03": ".class_visitor_part_03",
        "_parts": "._parts",
        "alias_first_violation": "._parts.class_visitor_part_02_01",
    }),
    public_exports=__all__,
)
