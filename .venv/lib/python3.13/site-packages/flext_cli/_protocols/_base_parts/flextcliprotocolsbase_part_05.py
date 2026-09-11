"""FlextCli protocol definitions - Structural typing contracts.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_cli._protocols._base_parts.flextcliprotocolsbase_part_04 import (
    FlextCliProtocolsBase as FlextCliProtocolsBasePart04,
)

if TYPE_CHECKING:
    # Why (multi-agent): defer flext_cli import to break the __init__-time
    # circular import; t is annotation-only (PEP 563). Matches sibling part_03.
    from flext_cli import t


class FlextCliProtocolsBase(FlextCliProtocolsBasePart04):
    """Implementation part for FlextCliProtocolsBase."""

    @runtime_checkable
    class YamlModule(Protocol):
        """Protocol for YAML serialization module interface."""

        def dump(self, data: t.JsonPayload, *, default_flow_style: bool = True) -> str:
            """Dump data as YAML string."""
            ...


__all__: list[str] = ["FlextCliProtocolsBase"]
