"""Dependency injection container facade."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._container_parts.flextcontainer_part_05 import FlextContainer

if TYPE_CHECKING:
    from flext_core import t

__all__: t.MutableSequenceOf[str] = ["FlextContainer"]
