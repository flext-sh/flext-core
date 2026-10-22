"""FlextProtocolsLoggingPart03 - logging and related infrastructure protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core._protocols._logging_parts.flextprotocolslogging_part_02 import (
    FlextProtocolsLoggingPart02,
)

if TYPE_CHECKING:
    from flext_core import t
    from flext_core._protocols import FlextProtocolsBase, FlextProtocolsResult


class FlextProtocolsLoggingPart03(FlextProtocolsLoggingPart02):
    @runtime_checkable
    class TextStream(Protocol):
        """Protocol for text-based output streams (stdout, stderr, file handles)."""

        mode: str
        name: str
        encoding: str

        def write(self, msg: str) -> int: ...

        def flush(self) -> None: ...

    type AccessibleData = (
        t.JsonPayload
        | FlextProtocolsBase.Model
        | t.MappingKV[str, t.JsonPayload | FlextProtocolsBase.Model | None]
        | FlextProtocolsResult.HasModelDump
        | FlextProtocolsLoggingPart03.ValidatorSpec
    )


__all__: list[str] = ["FlextProtocolsLoggingPart03"]
