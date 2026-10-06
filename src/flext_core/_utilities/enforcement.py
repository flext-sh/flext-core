"""Facade for FlextUtilitiesEnforcement.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._utilities._enforcement_parts.enforcement_part_01 import (
    PREDICATE_BINDINGS,
)
from flext_core._utilities._enforcement_parts.enforcement_part_05 import (
    FlextUtilitiesEnforcement,
)

__all__: list[str] = ["PREDICATE_BINDINGS", "FlextUtilitiesEnforcement"]
