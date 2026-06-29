"""CQRS handler foundation facade."""

from __future__ import annotations

from ._handlers_parts import FlextHandlers

h = FlextHandlers
__all__: list[str] = ["FlextHandlers", "h"]
