"""Runtime enforcement engine MRO part.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._utilities._enforcement import enforcement_part_03


class FlextUtilitiesEnforcementPart04(
    enforcement_part_03.FlextUtilitiesEnforcementPart03
):
    """Runtime enforcement engine; the rule catalog is package data."""


__all__: list[str] = ["FlextUtilitiesEnforcementPart04"]
