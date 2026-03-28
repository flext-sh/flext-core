"""Backward-compat shim -- canonical location is _utilities/decorators.py."""

from __future__ import annotations

from flext_core._utilities.decorators import FlextDecorators

d = FlextDecorators
__all__ = ["FlextDecorators", "d"]
