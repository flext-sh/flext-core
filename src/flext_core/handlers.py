"""CQRS handler foundation facade."""

from __future__ import annotations

from ._handlers_parts.flexthandlers_part_07 import FlextHandlers

h = FlextHandlers
__all__: list[str] = ["FlextHandlers", "h"]
