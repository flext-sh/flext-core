"""FlextProtocolsContext - context and bootstrap protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from ._context import (
    FlextProtocolsContextBootstrap,
    FlextProtocolsContextNamespaces,
    FlextProtocolsContextOperations,
)


# mro-wkii.17.26 (codex): compose focused contracts and retire numbered parts.
class FlextProtocolsContext(
    FlextProtocolsContextBootstrap,
    FlextProtocolsContextNamespaces,
    FlextProtocolsContextOperations,
):
    """Context and runtime bootstrap protocol facade."""


__all__: tuple[str, ...] = ("FlextProtocolsContext",)
