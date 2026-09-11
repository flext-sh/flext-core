"""FlextCli protocol definitions - Structural typing contracts.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_cli._protocols._base_parts.flextcliprotocolsbase_part_05 import (
    FlextCliProtocolsBase as FlextCliProtocolsBasePart05,
)


class FlextCliProtocolsBase(FlextCliProtocolsBasePart05):
    """Public facade for FlextCliProtocolsBase."""


__all__: list[str] = ["FlextCliProtocolsBase"]
