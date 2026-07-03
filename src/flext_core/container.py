"""Dependency injection container facade."""

from __future__ import annotations

from flext_core import t

from ._container_parts.flextcontainer_part_05 import FlextContainer

__all__: t.MutableSequenceOf[str] = ["FlextContainer"]
