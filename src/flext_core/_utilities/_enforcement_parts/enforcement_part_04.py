"""Runtime enforcement engine MRO part.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._utilities._enforcement_parts.enforcement_part_03 import (
    FlextUtilitiesEnforcement as FlextUtilitiesEnforcementPart03,
)


class FlextUtilitiesEnforcement(FlextUtilitiesEnforcementPart03):
    """Runtime enforcement engine; the rule catalog is package data."""


__all__: list[str] = ["FlextUtilitiesEnforcement"]
