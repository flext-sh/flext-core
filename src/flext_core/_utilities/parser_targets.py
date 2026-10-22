"""Facade for FlextUtilitiesParserTargets.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._utilities._parser_targets_parts.parser_targets_part_02 import (
    FlextUtilitiesParserTargetsPart02,
)


class FlextUtilitiesParserTargets(FlextUtilitiesParserTargetsPart02):
    """Compose parser targets through their internal part hierarchy."""


__all__: list[str] = ["FlextUtilitiesParserTargets"]
