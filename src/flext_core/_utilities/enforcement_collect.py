"""Facade for FlextUtilitiesEnforcementCollect.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._utilities._enforcement_collect_parts import (
    enforcement_collect_part_02 as _ec02,
)

FlextUtilitiesEnforcementCollect = _ec02.FlextUtilitiesEnforcementCollect

__all__: list[str] = ["FlextUtilitiesEnforcementCollect"]
