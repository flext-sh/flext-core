"""CQRS handler foundation facade.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core._handlers.flexthandlers_discovery import FlextHandlersDiscovery

if TYPE_CHECKING:
    from flext_core import p


class FlextHandlers[MessageT_contra, ResultT](
    FlextHandlersDiscovery[MessageT_contra, ResultT]
):
    """Public CQRS handler facade."""

    def __init__(self, *, settings: p.HandlerConfig | None = None) -> None:
        """Initialize the public handler with its original configuration."""
        super().__init__(settings=settings)


h = FlextHandlers

__all__: list[str] = ["FlextHandlers", "h"]
