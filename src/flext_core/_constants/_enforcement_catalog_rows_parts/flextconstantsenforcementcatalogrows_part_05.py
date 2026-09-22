"""Infrastructure enforcement catalog rows — extended rows (currently unused).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from flext_core import t


class FlextConstantsEnforcementCatalogInfraRowsExtended:
    """Extended infra detector rows placeholder (kept for import stability)."""

    INFRA_DETECTOR_ROWS_EXTENDED: ClassVar[
        tuple[tuple[str, str, str, str, t.VariadicTuple[str], bool, str], ...]
    ] = ()


__all__ = ["FlextConstantsEnforcementCatalogInfraRowsExtended"]
