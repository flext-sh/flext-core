"""Facade for FlextUtilitiesEnforcementCollect.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._utilities._enforcement_collect import enforcement_collect_part_02


class FlextUtilitiesEnforcementCollect(
    enforcement_collect_part_02.FlextUtilitiesEnforcementCollectPart02
):
    """Compose enforcement collection through its internal part hierarchy."""


__all__: list[str] = ["FlextUtilitiesEnforcementCollect"]
