"""Reusable service mixins facade."""

from __future__ import annotations

from ._mixins_parts.flextmixins_part_02 import FlextMixins

x = FlextMixins
__all__: list[str] = ["FlextMixins", "x"]
