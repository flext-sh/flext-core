"""Facade for FlextUtilitiesChecker.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._utilities._checker_parts.checker_part_03 import (
    FlextUtilitiesCheckerPart03,
)


class FlextUtilitiesChecker(FlextUtilitiesCheckerPart03):
    """Compose the checker namespace through the internal part hierarchy."""


__all__: list[str] = ["FlextUtilitiesChecker"]
